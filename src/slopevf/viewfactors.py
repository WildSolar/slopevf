"""View factors from points on the panel to the near and far ground.

Every ground strip and every boundary in the model is parallel to z', i.e.
the problem is two-dimensional. Integrating eqn. 26 over z' analytically,

    int dz / (r^2 + z^2)^2 = pi / (2 r^3),

reduces the view factor from a point to an infinitely long strip to the
classical 2D form

    F = 1/2 |sin(theta_2) - sin(theta_1)|,

with theta the angle between the panel normal and the line of sight to the
strip edges, measured in the cross-section. The same holds for the far-ground
wedge between the horizontal and the top edge of the row in front (eqn. 41).

The ``*_legacy`` functions reproduce the numerical integration of the MATLAB
implementation (VF_nearground.m, VF_farground.m).
"""

from __future__ import annotations

import numpy as np

from .geometry import Geometry


def strip_view_factor(geom: Geometry, chi, d1, d2, kernel=None, side=None):
    """View factor from panel point(s) ``chi`` to the ground strip between d1 and d2.

    Valid for either panel side as long as the strip lies within the visible
    window of that side. Arguments broadcast; infinite bounds are allowed. ``kernel`` is the
    transmission-weighted G(s) of :func:`slopevf.reflection.view_kernel` (None: G(s) = s).
    ``side`` selects the outer face of a panel of finite thickness.
    """
    return _vf(geom.sin_view_angle(chi, d1, side), geom.sin_view_angle(chi, d2, side), kernel)


def _vf(s1, s2, kernel=None):
    """1/2 |G(s2) - G(s1)| for the sines s of the view angles of two strip edges."""
    if kernel is not None:
        s1, s2 = kernel(s1), kernel(s2)
    return 0.5 * np.abs(s2 - s1)


def panel_view_factors(geom: Geometry, chi, edges, side: str, kernel=None):
    """View factors from points ``chi`` (C,) on ``side`` of row 0 to the elements of the facing
    face of the adjacent row, bounded by ``edges`` (E + 1,): back side -> front face of row +1,
    front side -> back face of row -1. Returns (C, E).

    The faces are parallel and see each other entirely, so every element is an unobstructed
    strip with F = 1/2 |G(s_2) - G(s_1)|, s the sine of the view angle to its edges.
    """
    chi = np.atleast_1d(np.asarray(chi, dtype=float))
    row, face = (1, "front") if side == "back" else (-1, "back")
    px, py = geom.panel_point(chi, 0, side)
    qx, qy = geom.panel_point(np.asarray(edges, dtype=float), row, face)
    ux, uy = qx[None, :] - px[:, None], qy[None, :] - py[:, None]
    s = (np.cos(geom.bp) * ux + np.sin(geom.bp) * uy) / np.hypot(ux, uy)
    if kernel is not None:
        s = kernel(s)
    return 0.5 * np.abs(np.diff(s, axis=1))


def outward(rows, side: str) -> bool:
    """Whether ``side`` of the evaluated row faces away from a finite array ``rows`` (no adjacent
    row on that side: front with n_front = 0, back with n_back = 0)."""
    return rows is not None and (rows[0] if side == "front" else rows[1]) == 0


def row_offsets(rows) -> np.ndarray:
    """Offsets of the existing rows relative to the evaluated row 0 for ``rows = (n_front, n_back)``:
    n_front rows in front of it (downhill, -1, -2, ...) and n_back rows behind it (uphill)."""
    n_front, n_back = rows
    return np.arange(-int(n_front), int(n_back) + 1)


def _lit_ground_finite(geom: Geometry, alpha, chi, side: str, rows, kernel=None):
    """Lit-ground view factor for a finite array of rows ``row_offsets(rows)``.

    The irradiated ground is the visible window minus the union of the shadows of the
    existing rows (outside the array the ground is unshaded). A side facing away from the array
    sees the ground out to infinity.
    """
    i = row_offsets(rows)
    i_min, i_max = i[0], i[-1]
    i = i[None, None, :]
    out = np.zeros((alpha.size, chi.size))
    active = (alpha > -geom.bg) & (alpha < np.pi - geom.bg)     # directions above the slope
    with np.errstate(divide="ignore", invalid="ignore"):
        d1, d2 = geom.shadow_edges(alpha)
        a, w = np.minimum(d1, d2), np.abs(d2 - d1)
    active &= np.isfinite(a) & np.isfinite(w)
    if not np.any(active):
        return out
    lo, hi = geom.view_window(chi, side, outward(rows, side))
    window = strip_view_factor(geom, chi, lo, hi, kernel, side)[None, :]
    aa, ww = a[:, None, None], w[:, None, None]
    merged = ww >= geom.p                                            # shadows overlap: one interval
    s = np.where(merged, aa + i_min * geom.p, aa + i * geom.p)
    e = np.where(merged, aa + ww + i_max * geom.p, aa + ww + i * geom.p)
    s, e = np.clip(s, lo[None, :, None], hi[None, :, None]), np.clip(e, lo[None, :, None], hi[None, :, None])
    c = chi[None, :, None]
    with np.errstate(invalid="ignore"):
        f = strip_view_factor(geom, c, s, e, kernel, side)
    f = np.where(e > s, f, 0.0)
    f[np.broadcast_to(merged, f.shape) & (i > i_min)] = 0.0         # merged: count once
    shaded = np.sum(f, axis=-1)
    out[active] = np.clip(window - shaded, 0.0, None)[active]
    return out


def lit_ground_view_factor(geom: Geometry, alpha, chi, side: str, extent: float = 50.0,
                           max_elements: int = 4_000_000, rows=None, kernel=None):
    """Aggregated view factor from panel points to all directly lit ground strips.

    Implements eqns. 27-32 in closed form for profile angles ``alpha`` (A,) and
    panel positions ``chi`` (C,); returns (A, C). Only meaningful for profile
    angles in cases (ii) and (iii); other angles return 0.

    Lit strips within ``extent`` (in panel heights, measured from where the
    panel plane meets the ground) are summed exactly. Beyond that, the view
    angle changes slowly over one row period and the remaining part of the
    window is weighted by the lit fraction of the ground.

    With ``rows = (n_front, n_back)``, the array is finite: only the evaluated row, n_front
    rows in front of it (downhill) and n_back rows behind it (uphill) cast shadows.
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    chi = np.atleast_1d(np.asarray(chi, dtype=float))
    if rows is not None:
        return _lit_ground_finite(geom, alpha, chi, side, rows, kernel)
    a4 = geom.alpha[3]
    p = geom.p
    out = np.zeros((alpha.size, chi.size))

    with np.errstate(divide="ignore", invalid="ignore"):
        d1, d2 = geom.shadow_edges(alpha)
        a = np.minimum(d1, d2)
        w = np.abs(d2 - d1)
        lit_frac = 1.0 - w / p
    # directions above the slope (alpha' < 0: terrain below the horizon); the shadows overlap
    # entirely below alpha_1 (and for most of the downhill wedge)
    active = (alpha > -geom.bg) & (alpha < a4) & (lit_frac > 0)
    if not np.any(active):
        return out

    lo, hi = geom.view_window(chi, side)
    d0 = geom.d_face(side)
    if side == "front":
        u, v = np.maximum(lo, d0 - extent), np.full_like(lo, d0)
        far_u, far_v = lo, np.full_like(lo, d0 - extent)
    else:
        u, v = np.full_like(hi, d0), np.minimum(hi, d0 + extent)
        far_u, far_v = np.full_like(hi, d0 + extent), hi
    far = far_v > far_u
    far_vf = np.where(far, strip_view_factor(geom, chi, far_u, np.where(far, far_v, far_u), kernel, side), 0.0)

    n_rows = int(np.ceil(extent / p)) + 2
    k = np.arange(n_rows)
    idx = np.flatnonzero(active)
    step = max(1, max_elements // (chi.size * n_rows))
    for s in range(0, idx.size, step):
        sel = idx[s:s + step]
        aa, ww = a[sel, None, None], w[sel, None, None]
        # first lit strip (b + (i-1)p, a + ip) that ends after u
        i0 = np.floor((u[None, :, None] - aa) / p) + 1
        i = i0 + k
        start = np.maximum(aa + ww + (i - 1) * p, u[None, :, None])
        end = np.minimum(aa + i * p, v[None, :, None])
        valid = end > start
        c = chi[None, :, None]
        f = _vf(geom.sin_view_angle(c, np.where(valid, start, 0.0), side),
                geom.sin_view_angle(c, np.where(valid, end, 0.0), side), kernel)
        out[sel] = np.sum(np.where(valid, f, 0.0), axis=-1) + lit_frac[sel, None] * far_vf[None, :]
    return out


def periodic_strips_view_factor(geom: Geometry, start, width, chi, side: str, extent: float = 50.0,
                                max_elements: int = 4_000_000, offsets=None, kernel=None,
                                outward: bool = False):
    """View factor from panel points to periodic ground strips [start + i p, start + width + i p].

    ``start`` and ``width`` are (A,) arrays (one strip per row period and case; width <= p,
    zero or NaN for none), ``chi`` is (C,); returns (A, C). Same exact summation and far-field
    approximation as :func:`lit_ground_view_factor`. With ``offsets`` (sequence of integer
    row offsets i), only these strips exist (finite array); ``outward``: the side faces away
    from the finite array (see :meth:`Geometry.view_window`).
    """
    start = np.atleast_1d(np.asarray(start, dtype=float))
    width = np.clip(np.nan_to_num(np.atleast_1d(np.asarray(width, dtype=float))), 0.0,
                    geom.p if offsets is None else np.inf)    # finite rows: overlapping bundles add up
    chi = np.atleast_1d(np.asarray(chi, dtype=float))
    p = geom.p
    out = np.zeros((start.size, chi.size))
    active = (width > 0) & np.isfinite(start)
    if not np.any(active):
        return out
    lo, hi = geom.view_window(chi, side, outward)
    if offsets is not None:
        i = np.asarray(offsets, dtype=float)[None, None, :]
        if i.size == 0:
            return out
        a, w = np.where(active, start, 0.0)[:, None, None], width[:, None, None]
        s = np.clip(a + i * p, lo[None, :, None], hi[None, :, None])
        e = np.clip(a + w + i * p, lo[None, :, None], hi[None, :, None])
        f = np.where(e > s, strip_view_factor(geom, chi[None, :, None], s, e, kernel, side), 0.0)
        return np.where(active[:, None], np.sum(f, axis=-1), 0.0)
    d0 = geom.d_face(side)
    if side == "front":
        u, v = np.maximum(lo, d0 - extent), np.full_like(lo, d0)
        far_u, far_v = lo, np.full_like(lo, d0 - extent)
    else:
        u, v = np.full_like(hi, d0), np.minimum(hi, d0 + extent)
        far_u, far_v = np.full_like(hi, d0 + extent), hi
    far = far_v > far_u
    far_vf = np.where(far, strip_view_factor(geom, chi, far_u, np.where(far, far_v, far_u), kernel, side), 0.0)
    frac = width / p
    n_rows = int(np.ceil(extent / p)) + 2
    k = np.arange(n_rows)
    idx = np.flatnonzero(active)
    step = max(1, max_elements // (chi.size * n_rows))
    for s0 in range(0, idx.size, step):
        sel = idx[s0:s0 + step]
        a, w = start[sel, None, None], width[sel, None, None]
        i0 = np.floor((u[None, :, None] - a - w) / p) + 1        # first strip ending after u
        i = i0 + k
        s = np.maximum(a + i * p, u[None, :, None])
        e = np.minimum(a + w + i * p, v[None, :, None])
        valid = e > s
        c = chi[None, :, None]
        f = _vf(geom.sin_view_angle(c, np.where(valid, s, 0.0), side),
                geom.sin_view_angle(c, np.where(valid, e, 0.0), side), kernel)
        out[sel] = np.sum(np.where(valid, f, 0.0), axis=-1) + frac[sel, None] * far_vf[None, :]
    return out


def far_ground_view_factor(geom: Geometry, chi, kernel=None, outward: bool = False):
    """View factor from front-side points to the far ground (closed form of eqn. 41).

    The far ground is the wedge between the horizontal and the line of sight to
    the top edge of the row in front (for finite thickness its upper silhouette corner);
    it is empty for points below that edge. ``outward`` (no row in front): the wedge between
    the horizontal and the downhill slope, i.e. the terrain below the horizontal beyond the
    ground in view (empty on horizontal ground).
    """
    chi = np.asarray(chi, dtype=float)
    if outward:
        return _vf(-np.cos(geom.bp), geom.sin_view_angle(chi, -np.inf, "front"), kernel)
    px, py = geom.panel_point(chi, 0, "front")
    tops = geom.corners(-1)[2:] if geom.t > 0 else [geom.panel_point(0.5, row=-1)]
    elev, s_top = [], []
    for tx, ty in tops:
        ux, uy = tx - px, ty - py
        elev.append(np.arctan2(uy, -ux))                 # elevation of the line of sight (downhill)
        s_top.append((np.cos(geom.bp) * ux + np.sin(geom.bp) * uy) / np.hypot(ux, uy))
    k = np.argmax(np.array(elev), axis=0)                # highest line of sight over the row
    e_top = np.choose(k, elev)
    s_top = np.choose(k, s_top)
    s_horizon = -np.cos(geom.bp)
    return np.where(e_top < 0, _vf(s_horizon, s_top, kernel), 0.0)


# --- reference integration of eqn. 26 (used by the tests) -----------------------------

def near_ground_view_factor_numeric(geom: Geometry, chi, d1, d2, n_d=2000, n_zeta=2000):
    """Midpoint-rule integration of eqn. 26 (back side sign convention) for a scalar strip."""
    mu, gam, h = geom.bp, geom.bg, geom.h
    d = d1 + (np.arange(n_d) + 0.5) * (d2 - d1) / n_d
    zeta = -np.pi / 2 + (np.arange(n_zeta) + 0.5) * np.pi / n_zeta
    D, T2 = np.meshgrid(d, np.tan(zeta) ** 2)
    num = (1 + T2) * (h * np.cos(mu) - D * np.sin(gam - mu)) * (chi * np.sin(mu - gam) + h * np.cos(gam))
    den = (D ** 2 + chi ** 2 - 2 * D * chi * np.cos(gam - mu) + h ** 2 + T2
           - 2 * h * (D * np.sin(gam) - chi * np.sin(mu))) ** 2
    return np.sum(num / den) / np.pi * (np.pi / n_zeta) * ((d2 - d1) / n_d)


# --- legacy (MATLAB) implementations ------------------------------------------------

def near_ground_view_factor_legacy(chi, d1, d2, mu, gamma, hr, n_d, n_u, max_elements=4_000_000):
    """Vectorised VF_nearground.m for many (chi, d1, d2) triples."""
    chi, d1, d2 = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (chi, d1, d2)))
    chi, d1, d2 = chi.ravel(), d1.ravel(), d2.ravel()
    out = np.zeros(chi.size)
    nd = np.ceil(n_d / (1 + np.abs(np.fmin(d1, d2))))
    todo = d1 != d2
    u_vec = np.linspace(-np.pi / 2, np.pi / 2, n_u + 1)
    u = 0.5 * (u_vec[:-1] + u_vec[1:])
    d_u = np.pi / n_u
    tu2 = np.tan(u)[None, :, None] ** 2
    for m in np.unique(nd[todo]):
        m_int = int(m)
        sel = np.flatnonzero(todo & (nd == m))
        step = max(1, max_elements // (m_int * n_u))
        frac = np.arange(m_int + 1) / m_int
        for s in range(0, sel.size, step):
            j = sel[s:s + step]
            a, b, c = d1[j, None], d2[j, None], chi[j, None]
            d_vec = a + (b - a) * frac
            d_vec[:, -1] = b[:, 0]
            D = (0.5 * (d_vec[:, :-1] + d_vec[:, 1:]))[:, None, :]
            d_d = ((b - a) / m_int)[:, :, None]
            cc = c[:, :, None]
            numint = (1 / np.pi * (1 + tu2) * (hr * np.cos(mu) - D * np.sin(gamma - mu))
                      * (cc * np.sin(mu - gamma) + hr * np.cos(gamma))
                      / (D ** 2 + cc ** 2 - 2 * D * cc * np.cos(gamma - mu) + hr ** 2 + tu2
                         - 2 * hr * (D * np.sin(gamma) - cc * np.sin(mu))) ** 2
                      * (d_u * d_d))
            out[j] = np.sum(numint, axis=(1, 2))
    return out


def far_ground_view_factor_legacy(chi, mu, gamma, rho, n_phi, n_sig):
    """Vectorised VF_farground.m (including its deviations from eqn. 41)."""
    chi = np.atleast_1d(np.asarray(chi, dtype=float))[:, None, None]
    phi_delim = np.linspace(-np.pi / 2, np.pi / 2, n_phi + 1)
    phi = ((phi_delim[1:] + phi_delim[:-1]) / 2)[None, :, None]
    d_phi = phi_delim[1] - phi_delim[0]
    sig_bottom = np.arctan((rho * np.tan(gamma) + (chi - 1 / 2) * np.sin(mu))
                           / (np.sqrt((2 * np.cos(mu) * chi - np.cos(mu) + 2 * rho) ** 2
                                      * (1 / np.cos(phi))) / 2))
    frac = np.arange(n_sig + 1) / n_sig
    sig_delim = sig_bottom * frac
    sig_delim[..., -1] = sig_bottom[..., 0]
    sig = (sig_delim[..., 1:] + sig_delim[..., :-1]) / 2
    d_sig = sig_delim[..., 1:2] - sig_delim[..., 0:1]
    term = 1 / (2 * np.pi) * (np.sin(mu) * np.cos(phi) * np.cos(sig) - np.cos(mu) * np.sin(sig)) * d_sig * d_phi
    return np.sum(np.maximum(0.0, term), axis=(1, 2))
