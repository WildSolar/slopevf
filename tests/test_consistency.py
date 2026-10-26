"""Model-level consistency checks: view-factor closure, a limiting case and symmetries."""

import numpy as np
import pytest

from slopevf import Array, FarGround, Settings, simulate
from slopevf.geometry import Geometry
from slopevf.model import ArrayModel
from slopevf.sky import make_sky_grid
from slopevf.viewfactors import far_ground_view_factor, lit_ground_view_factor, strip_view_factor

CHI = np.array([-0.49, -0.4, -0.2, 0.0, 0.2, 0.4, 0.49])


def _sin_to(g, chi, x, y, side):
    """Sine of the view angle (measured from the panel normal) from the face of ``side``
    towards the point (x, y)."""
    px, py = g.panel_point(chi, 0, side)
    ux, uy = x - px, y - py
    return (np.cos(g.bp) * ux + np.sin(g.bp) * uy) / np.hypot(ux, uy)


def _row_extent(g, chi, row, side):
    """Range of view-angle sines covered by the cross-section of ``row`` (edges or corners)."""
    s = [_sin_to(g, chi, x, y, side) for x, y in g.corners(row)]
    return min(s), max(s)


# --- view-factor closure ---------------------------------------------------------------------

def test_front_closure(array):
    """Visible ground + downhill row + far ground + sky = 1 for every front-side point.

    In the cross-section the front hemisphere is swept from the panel plane downwards
    (S = -1) to the panel plane upwards (S = +1): near ground up to the line of sight to
    the lower edge Q of row -1, row -1 itself up to its upper edge T, the far ground up to
    the horizontal (if T lies below it) and the sky above. The ground and far-ground terms
    are taken from the implementation, the row and sky terms from the bare geometry (for
    panels of finite thickness, the silhouette corners of row -1 seen from the outer face).
    """
    g = Geometry(array)
    for chi in CHI:
        s_q, s_t = _row_extent(g, chi, -1, "front")
        s_h = -np.cos(g.bp)
        lo, hi = g.view_window(chi, "front")
        f_ground = strip_view_factor(g, chi, lo, hi, side="front")
        assert f_ground == pytest.approx(0.5 * (1 + s_q), abs=1e-12), chi
        f_row = 0.5 * (s_t - s_q)
        f_sky = 0.5 * (1 - max(s_t, s_h))
        f_far = far_ground_view_factor(g, chi)
        assert f_ground + f_row + f_far + f_sky == pytest.approx(1.0, abs=1e-12), chi


def test_back_window_edge(array):
    """The visible ground of the backside ends exactly at the line of sight to the lower edge
    (lowest silhouette corner) of row +1."""
    g = Geometry(array)
    for chi in CHI:
        s_b = _row_extent(g, chi, +1, "back")[0]
        lo, hi = g.view_window(chi, "back")
        assert strip_view_factor(g, chi, lo, hi, side="back") == pytest.approx(0.5 * (1 + s_b), abs=1e-12), chi


@pytest.mark.parametrize("side", ["front", "back"])
def test_lit_plus_shaded_ground(array, side):
    """Irradiated and shaded ground strips together make up the whole visible ground.

    The shaded strips are summed explicitly over 4000 row periods, independently of the
    implementation's exact range and tail approximation.
    """
    g = Geometry(array)
    a1, a2, a3, _ = g.alpha
    alphas = np.concatenate([np.linspace(max(a1, 0) + 0.02, a3 - 0.02, 9), [a2]])
    chi = CHI[1:-1]
    lit = lit_ground_view_factor(g, alphas, chi, side)
    lo, hi = g.view_window(chi, side)
    window = strip_view_factor(g, chi, lo, hi, side=side)
    for k, a in enumerate(alphas):
        d1, d2 = g.shadow_edges(a)
        s0, w = min(d1, d2), abs(d2 - d1)
        if w >= g.p:      # thick panels: shadows still overlap just above alpha_1
            np.testing.assert_array_equal(lit[k], 0.0)
            continue
        i = np.arange(-2000, 2001)[:, None]
        start = np.clip(s0 + i * g.p, lo, hi)
        end = np.clip(s0 + w + i * g.p, lo, hi)
        shaded = np.sum(strip_view_factor(g, chi[None, :], start, end, side=side), axis=0)
        np.testing.assert_allclose(lit[k] + shaded, window, atol=3e-4, err_msg=f"alpha={a}")
        if a == a2 and g.t == 0:   # light parallel to thin panels: no ground shadow at all
            np.testing.assert_allclose(lit[k], window, atol=1e-12)


# --- limiting case ---------------------------------------------------------------------------

@pytest.mark.parametrize("beta_p", [30, 60, 90])
def test_isolated_panel_isotropic_sky(beta_p):
    """Isolated panel high above horizontal ground under an isotropic sky (Liu-Jordan limit).

    front = DHI (1 + cos b)/2 + albedo DHI (1 - cos b)/2
    back  = DHI (1 - cos b)/2 + albedo DHI (1 + cos b)/2
    Remaining deviations stem from the panel's own shadow on the ground (vanishing with
    1/h) and the sky discretisation.
    """
    arr = Array(beta_p, 0, 400.0, 200.0)
    dhi, albedo = 200.0, 0.5
    grid = make_sky_grid("medium")
    model = ArrayModel(arr, FarGround(), grid, Settings(n_chi=10))
    L = np.full((grid.size, 1), dhi / np.pi)
    out = model.apply(L, np.array([albedo]), np.array([0.0]))
    c = np.cos(np.radians(beta_p))
    front = out["front_direct"] + out["front_nearground"] + out["front_farground"]
    back = out["back_direct"] + out["back_nearground"]
    np.testing.assert_allclose(out["front_direct"], dhi * (1 + c) / 2, rtol=2e-3)
    np.testing.assert_allclose(out["back_direct"], dhi * (1 - c) / 2, rtol=2e-3, atol=0.05)
    np.testing.assert_allclose(front, dhi * (1 + c) / 2 + albedo * dhi * (1 - c) / 2, rtol=5e-3)
    np.testing.assert_allclose(back, dhi * (1 - c) / 2 + albedo * dhi * (1 + c) / 2, rtol=5e-3)


# --- symmetries ------------------------------------------------------------------------------

WEATHER = dict(dni=[700.0, 300.0, 0.0], ghi=[600.0, 350.0, 180.0], solar_elevation=[25.0, 50.0, 30.0])


@pytest.mark.parametrize("sun_model", ["point", "buie"])
def test_mirror_symmetry(array, sun_model):
    """Sun positions mirrored about the slope azimuth give identical results."""
    if array.gamma != 0:
        pytest.skip("the sky grid is mirror symmetric about south only")
    s = Settings(sky="low", n_chi=20, sun_model=sun_model)
    az = np.array([-35.0, 70.0, -150.0])
    kw = dict(albedo=0.6, far_ground=FarGround(15, 0), albedo_fg=0.3, settings=s)
    a = simulate(array, dict(WEATHER, solar_azimuth=array.gamma + az), **kw)
    b = simulate(array, dict(WEATHER, solar_azimuth=array.gamma - az), **kw)
    for c in ("front", "back"):
        np.testing.assert_allclose(getattr(a, c), getattr(b, c), rtol=1e-9, atol=1e-9)


def test_rotation_invariance():
    """Rotating the slope, the sun and the far ground together leaves the result unchanged.

    The sky grid is not rotation invariant, so the diffuse part carries a small
    discretisation difference; the point-sun beam is exact.
    """
    s = Settings(sky="medium", n_chi=20)
    az = np.array([-35.0, 70.0, 150.0])
    base = dict(albedo=0.6, albedo_fg=0.3, settings=s)
    res = []
    for rot in (0.0, 37.0, -110.0):
        arr = Array(60, 20, 1.3, 1.2, gamma=rot)
        res.append(simulate(arr, dict(WEATHER, solar_azimuth=az + rot),
                            far_ground=FarGround(15, 160 + rot), **base))
    for r in res[1:]:
        for c in ("front", "back"):
            np.testing.assert_allclose(getattr(r, c), getattr(res[0], c), rtol=1e-3, atol=0.01)
