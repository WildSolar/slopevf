"""Specular reflection of the module glass between adjacent rows."""

import numpy as np
import pytest

from slopevf import Array, PanelGlass, Settings, simulate
from slopevf.geometry import Geometry
from slopevf.reflection import fresnel_reflectance, view_kernel
from slopevf.tables import reflected_fractions

import raytrace


def test_fresnel():
    n = 1.52
    assert fresnel_reflectance(1.0, n) == pytest.approx(((n - 1) / (n + 1)) ** 2)     # 4.26 %
    assert fresnel_reflectance(np.cos(np.radians(60)), n) == pytest.approx(0.0925, abs=1e-3)
    assert fresnel_reflectance(0.0, n) == pytest.approx(1.0)
    c = np.cos(np.radians(np.linspace(0, 89, 90)))
    assert np.all(np.diff(fresnel_reflectance(c, n)) > 0)           # increases towards grazing
    assert PanelGlass(reflectance=lambda c: 0.02 * np.ones_like(c))(0.5) == pytest.approx(0.02)


def test_transmittance():
    g = PanelGlass(transmission=True)
    assert g.transmittance(1.0) == pytest.approx(1.0)
    assert g.transmittance(np.cos(np.radians(60))) == pytest.approx((1 - 0.0925) / (1 - 0.0426), abs=1e-3)
    assert g.transmittance(0.0) == 0.0
    np.testing.assert_array_equal(PanelGlass().transmittance([0.2, 0.9]), 1.0)     # off by default


def test_view_kernel():
    """G(s) of the 2D view factor: identity without weighting, hemispherical average of tau,
    and a direct 3D integration over a strip."""
    G1 = view_kernel(np.ones_like)
    s = np.linspace(-1, 1, 41)
    np.testing.assert_allclose(G1(s), s, atol=1e-6)
    tau = PanelGlass(transmission=True).transmittance
    G = view_kernel(tau)
    th = (np.arange(20000) + 0.5) * (np.pi / 2) / 20000
    hemi = np.sum(tau(np.cos(th)) * np.sin(2 * th)) * (np.pi / 2) / 20000
    assert G(1.0) == pytest.approx(hemi, rel=1e-4)
    assert G(-0.7) == pytest.approx(-G(0.7))
    # strip between view angles phi1, phi2 (cross-section), infinitely long
    phi1, phi2, n = -0.3, 0.9, 2000
    phi = phi1 + (np.arange(n) + 0.5) * (phi2 - phi1) / n
    om = -np.pi / 2 + (np.arange(n) + 0.5) * np.pi / n
    P, O = np.meshgrid(phi, om)
    c = np.cos(O) * np.cos(P)
    ref = np.sum(tau(c) * c * np.cos(O)) * (phi2 - phi1) / n * np.pi / n / np.pi
    assert 0.5 * (G(np.sin(phi2)) - G(np.sin(phi1))) == pytest.approx(ref, rel=1e-4)


def test_transmission_model():
    """R = 0 reproduces the model without glass; a pure beam is weighted with tau(theta)."""
    arr = Array(60, 20, 1.6, 1.8)
    w = dict(dni=[650.0, 300.0], ghi=[420.0, 250.0], solar_azimuth=[-30.0, 20.0], solar_elevation=[25.0, 40.0])
    s = Settings(sky="low", n_chi=30)
    plain = simulate(arr, w, albedo=0.6, settings=s)
    clear = simulate(arr, w, albedo=0.6, settings=s,
                     panel_glass=PanelGlass(reflectance=lambda c: 0 * c, transmission=True))
    for c in ("front_direct", "front_nearground", "front_farground", "back_direct", "back_nearground"):
        np.testing.assert_allclose(getattr(clear, c), getattr(plain, c), rtol=1e-4, atol=1e-6)
    # beam only (DHI = 0): front_direct scales with tau of the sun's incidence angle
    el, az = np.radians(25.0), np.radians(-30.0)
    wb = dict(dni=[650.0], ghi=[650.0 * np.sin(el)], solar_azimuth=[-30.0], solar_elevation=[25.0])
    glass = PanelGlass(front=False, back=False, transmission=True)
    pb = simulate(arr, wb, albedo=0.6, settings=s)
    tb = simulate(arr, wb, albedo=0.6, settings=s, panel_glass=glass)
    from slopevf.geometry import direction_vector, surface_normal
    cos_t = float(direction_vector(el, az) @ surface_normal(np.radians(60), 0.0))
    assert cos_t > 0
    lit = pb.front_direct[0] > 0
    np.testing.assert_allclose(tb.front_direct[0][lit] / pb.front_direct[0][lit], glass.transmittance(cos_t), rtol=1e-9)
    # diffuse and ground-reflected light is attenuated more than the near-normal beam
    t = simulate(arr, w, albedo=0.6, settings=s, panel_glass=glass)
    assert np.all(t.back_nearground.mean(axis=1) < 0.97 * plain.back_nearground.mean(axis=1))


def _brute_force(g, alpha, chi_lo, chi_hi, side, n_sub=200, n_rows=40):
    """Fraction of a receiving element reached by the specular reflection off the adjacent row."""
    chi = np.linspace(chi_lo, chi_hi, n_sub + 1)
    chi = 0.5 * (chi[1:] + chi[:-1])
    n_f = np.array([-np.sin(g.bp), np.cos(g.bp)])
    n_recv, mirror_row = (-n_f, +1) if side == "back" else (n_f, -1)
    n_mirror = -n_recv
    u = raytrace.sky_dir(alpha)
    if u @ n_mirror <= 0 or alpha >= np.pi - g.bg:
        return 0.0
    p = -u
    q = -(p - 2 * (p @ n_mirror) * n_mirror)                 # from receiver towards the mirror
    if q @ n_recv <= 0:
        return 0.0
    px, py = g.panel_point(chi, 0, side)
    P = np.stack([px, py], -1) + 1e-9 * n_recv
    A, B = raytrace.panel_segments(g, [mirror_row])
    # intersection with the mirror segment (thick panels: the face towards the receiver)
    m = 0 if (g.t == 0 or side == "back") else 1
    e = B[m] - A[m]
    w = A[m] - P
    den = q[0] * e[1] - q[1] * e[0]
    t = (w[:, 0] * e[1] - w[:, 1] * e[0]) / den
    s = (w[:, 0] * q[1] - w[:, 1] * q[0]) / den
    hit = (t > 0) & (s >= 0) & (s <= 1)
    Q = P + t[:, None] * q + 1e-9 * n_mirror
    others = np.array([i for i in range(-n_rows, n_rows + 1) if i != mirror_row])
    A2, B2 = raytrace.panel_segments(g, others)
    lit = ~raytrace.ray_hits(Q, np.broadcast_to(u, Q.shape), A2, B2)
    return float(np.mean(hit & lit))


@pytest.mark.parametrize("side", ["back", "front"])
def test_reflected_fractions_ray_tracing(array, side):
    g = Geometry(array)
    n_chi = 8
    width = 1 / n_chi
    chi = -0.5 + (np.arange(n_chi) + 0.5) * width
    alphas = np.linspace(0.02, np.pi - g.bg - 0.02, 41)
    rb, rf = reflected_fractions(g, alphas, chi, width)
    table = rb if side == "back" else rf
    for i, a in enumerate(alphas):
        for j, c in enumerate(chi):
            ref = _brute_force(g, a, c - width / 2, c + width / 2, side)
            assert table[i, j] == pytest.approx(ref, abs=0.011), (side, a, c)


def _first_hit(origin, direction, A, B):
    """Parameter t and segment index of the first intersection of each ray with segments A-B."""
    o, d = origin[:, None, :], direction[None, None, :]
    e = (B - A)[None, :, :]
    w = A[None, :, :] - o
    den = d[..., 0] * e[..., 1] - d[..., 1] * e[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[..., 0] * e[..., 1] - w[..., 1] * e[..., 0]) / den
        s = (w[..., 0] * d[..., 1] - w[..., 1] * d[..., 0]) / den
    t = np.where((t > 1e-9) & (s >= 0) & (s <= 1), t, np.inf)
    j = np.argmin(t, axis=1)
    return t[np.arange(len(origin)), j], j


@pytest.mark.parametrize("rows, receivers", [
    (None, ("front", "back")),
    ((2, 0), ("front", "back")),    # Totalp: front module on the last (uphill) of three rows
    ((0, 2), ("front", "back")),    # Totalp: back module on the first (downhill) of three rows
    ((1, 1), ("front", "back")),
])
@pytest.mark.parametrize("mirror", ["front", "back"])
def test_reflected_ground_ray_tracing(array, mirror, rows, receivers):
    """View factors to ground irradiated by reflected radiation, against 2D ray tracing
    (infinite rows, or a finite array with n_front rows in front and n_back rows behind)."""
    from slopevf.tables import reflected_ground_view_factors
    g = Geometry(array)
    n_f = np.array([-np.sin(g.bp), np.cos(g.bp)])
    n_m = n_f if mirror == "front" else -n_f
    offsets = np.arange(-60, 61) if rows is None else np.arange(-rows[0], rows[1] + 1)
    A, B = raytrace.panel_segments(g, offsets)
    chi = np.array([-0.3, 0.1, 0.4])
    alphas = np.linspace(0.05, np.pi - g.bg - 0.05, 9)
    table = reflected_ground_view_factors(g, alphas, chi, rows=rows)
    for i, a in enumerate(alphas):
        u = raytrace.sky_dir(a)
        if u @ n_m <= 0:
            continue
        p = -u
        r = p - 2 * (p @ n_m) * n_m
        for receiver in receivers:
            for j, c in enumerate(chi):
                if rows is None:
                    lo, hi = g.view_window(c, receiver)
                    lo, hi = max(lo, -40.0), min(hi, 40.0)
                    d = np.linspace(lo, hi, 8001)
                else:                                   # ray-traced visibility, wide range
                    d = np.linspace(-60.0, 60.0, 24001)
                dm = 0.5 * (d[1:] + d[:-1])
                if rows is not None:
                    vis = raytrace.ground_visible(g, c, dm, receiver, rows=offsets)
                G = np.stack([dm * np.cos(g.bg), dm * np.sin(g.bg)], -1)
                t, k = _first_hit(G, -r, A, B)          # back along the reflected ray
                Q = G - t[:, None] * r
                hit = np.isfinite(t) & ((-r) @ n_m < 0)  # arrives at the mirror side
                if g.t > 0:                              # thick: 4 segments per row, must be the mirror face
                    row_k = offsets[k // 4]
                    hit &= (k % 4) == (0 if mirror == "front" else 1)
                else:
                    row_k = offsets[k]
                with np.errstate(invalid="ignore"):          # rays without a hit (t = inf)
                    chi_q = ((Q - np.stack(g.panel_point(0.0, row_k, mirror), -1))
                             @ np.array([np.cos(g.bp), np.sin(g.bp)]))
                lit = hit & ~raytrace.ray_hits(Q + 1e-9 * n_m, np.broadcast_to(u, Q.shape), A, B)
                lit &= np.abs(chi_q) <= 0.5
                if rows is not None:
                    lit &= vis
                ds = np.abs(np.diff(g.sin_view_angle(c, d, receiver))) / 2
                ref = np.sum(ds[lit])
                assert table[(mirror, receiver)][i, j] == pytest.approx(ref, abs=3e-3), (mirror, receiver, a, c)


def test_vertical_panels_analytic():
    """Vertical panels on flat ground, sun in front (south) at elevation e, beam only.

    The mirror (front of the uphill row) is shaded below (h + 1/2) - rho tan(e); the reflected
    beam descends by rho tan(e) between the rows. The back side therefore receives
    DNI R(e) cos(e) on the band [h + 1/2 - 2 rho tan e, h + 1/2 - rho tan e] (heights).
    """
    h, rho, e = 1.0, 2.0, np.radians(12.0)
    arr = Array(90, 0, rho, h)
    dni = 800.0
    w = dict(dni=[dni], ghi=[dni * np.sin(e)], solar_azimuth=[0.0], solar_elevation=[np.degrees(e)])
    r = simulate(arr, w, albedo=0.0, panel_glass=PanelGlass(n=1.52, back=False),
                 settings=Settings(n_chi=400, sky="low"))
    z = h + r.chi                                         # height of the element centres
    top = h + 0.5 - rho * np.tan(e)
    bottom = h + 0.5 - 2 * rho * np.tan(e)
    expected = np.where((z > bottom) & (z < top), dni * fresnel_reflectance(np.cos(e)) * np.cos(e), 0.0)
    edge = (np.abs(z - top) < 1.5 / 400) | (np.abs(z - bottom) < 1.5 / 400)
    np.testing.assert_allclose(r.back_reflected[0][~edge], expected[~edge], atol=1e-9)
    assert np.allclose(r.back_direct, 0) and np.allclose(r.back_nearground, 0)


def test_reflection_components_and_totals():
    # flat ground: with the sun behind the array, the beam reflected off the back of the
    # downhill row descends only by rho tan(e) and reaches the front of the next row (on steep
    # slopes it largely passes below it)
    arr = Array(90, 0, 1.2, 1.5)
    w = dict(dni=[650.0, 600.0], ghi=[250.0, 250.0], solar_azimuth=[-20.0, 170.0], solar_elevation=[12.0, 10.0])
    s = Settings(sky="low", n_chi=40)
    plain = simulate(arr, w, albedo=0.5, settings=s)
    glass = simulate(arr, w, albedo=0.5, panel_glass=PanelGlass(), settings=s)
    assert np.all(plain.back_reflected == 0) and np.all(plain.front_reflected == 0)
    assert np.all(glass.back_reflected >= 0) and np.all(glass.front_reflected >= 0)
    # sun in front: reflection onto the back side; sun behind: onto the front side
    assert glass.back_reflected[0].mean() > 1.0 and glass.front_reflected[1].mean() > 1.0
    np.testing.assert_allclose(glass.back - glass.back_reflected, plain.back)
    np.testing.assert_allclose(glass.front - glass.front_reflected, plain.front)
    with pytest.raises(ValueError):
        simulate(arr, w, albedo=0.5, panel_glass=PanelGlass(),
                 settings=Settings.matlab(sky="low", n_chi=10, n_alpha=60, n_d=10, n_zeta=10, n_phi=10, n_sig=10))


@pytest.mark.parametrize("thickness", [0.0, 0.15])
@pytest.mark.parametrize("side", ["back", "front"])
def test_panel_view_factors(side, thickness):
    """Point-to-element view factors between facing faces: 2D differential view factor
    cos(theta_r) cos(theta_s) / (2 r) integrated along the facing face, and reciprocity."""
    from slopevf.viewfactors import panel_view_factors
    g = Geometry(Array(70, 20, 1.3, 1.6, thickness=thickness))
    edges = np.linspace(-0.5, 0.5, 9)
    chi = np.array([-0.4, 0.0, 0.35])
    V = panel_view_factors(g, chi, edges, side)
    n_f = np.array([-np.sin(g.bp), np.cos(g.bp)])
    n_r = -n_f if side == "back" else n_f
    row, face = (1, "front") if side == "back" else (-1, "back")
    for i, c in enumerate(chi):
        P = np.array(g.panel_point(c, 0, side))
        for j in range(edges.size - 1):
            s = np.linspace(edges[j], edges[j + 1], 4001)
            sm = 0.5 * (s[1:] + s[:-1])
            Q = np.stack(g.panel_point(sm, row, face), -1)
            d = Q - P
            r = np.hypot(d[:, 0], d[:, 1])
            cos_r, cos_s = (d @ n_r) / r, (-d @ -n_r) / r
            ref = np.sum(cos_r * cos_s / (2 * r)) * (s[1] - s[0])
            assert V[i, j] == pytest.approx(ref, rel=1e-6, abs=1e-9)
    # reciprocity between the back face of row 0 and the front face of row +1 (equal elements)
    n = 200
    e = np.linspace(-0.5, 0.5, n + 1)
    c = 0.5 * (e[1:] + e[:-1])
    np.testing.assert_allclose(panel_view_factors(g, c, e, "back"), panel_view_factors(g, c, e, "front").T,
                               atol=2e-5)


def test_diffuse_panel_reflection():
    """Single diffuse reflection off the facing face: linear in R_d, R_d * (E_source @ V^T),
    and for finite arrays the source is the adjacent row's own irradiance."""
    from slopevf.viewfactors import panel_view_factors
    arr = Array(90, 20, 1.2, 1.8, thickness=0.1)
    w = dict(dni=[650.0, 300.0], ghi=[420.0, 250.0], solar_azimuth=[-30.0, 20.0], solar_elevation=[25.0, 40.0])
    s = Settings(sky="low", n_chi=30)
    base = simulate(arr, w, albedo=0.6, settings=s, panel_glass=PanelGlass(transmission=True))
    r1 = simulate(arr, w, albedo=0.6, settings=s, panel_glass=PanelGlass(transmission=True, diffuse=0.05))
    r2 = simulate(arr, w, albedo=0.6, settings=s, panel_glass=PanelGlass(transmission=True, diffuse=0.1))
    assert np.all(base.back_diffuse == 0) and np.all(base.front_diffuse == 0)
    np.testing.assert_allclose(r2.back_diffuse, 2 * r1.back_diffuse)
    np.testing.assert_allclose(r1.back - r1.back_diffuse, base.back)
    np.testing.assert_allclose(r1.front - r1.front_diffuse, base.front)
    edges = np.linspace(-0.5, 0.5, 31)
    kernel = PanelGlass(transmission=True).view_kernel()
    Vb = panel_view_factors(Geometry(arr), base.chi, edges, "back", kernel)
    Vf = panel_view_factors(Geometry(arr), base.chi, edges, "front", kernel)
    np.testing.assert_allclose(r1.back_diffuse, 0.05 * base.front @ Vb.T)
    np.testing.assert_allclose(r1.front_diffuse, 0.05 * base.back @ Vf.T)
    assert r1.back_diffuse.mean() > 0.02 * base.back.mean()          # significant for the back side
    # finite array, first row of three: the source is the front of the middle row
    sf = Settings(sky="low", n_chi=30, finite_rows=(0, 2))
    first = simulate(arr, w, albedo=0.6, settings=sf, panel_glass=PanelGlass(diffuse=0.1))
    middle = simulate(arr, w, albedo=0.6, settings=Settings(sky="low", n_chi=30, finite_rows=(1, 1)),
                      panel_glass=PanelGlass())
    np.testing.assert_allclose(first.back_diffuse, 0.1 * middle.front @ panel_view_factors(
        Geometry(arr), first.chi, edges, "back").T)
    assert np.all(first.front_diffuse == 0)                          # outward side: no neighbour
    # only the selected sides reflect
    rb = simulate(arr, w, albedo=0.6, settings=s, panel_glass=PanelGlass(front=False, diffuse=0.1))
    assert np.all(rb.back_diffuse == 0) and np.all(rb.front_diffuse > 0)
    with pytest.raises(ValueError):
        PanelGlass(diffuse=1.2)
