"""Independent brute-force 2D ray tracing of the array cross-section, used as test oracle."""

import numpy as np

from slopevf.geometry import Geometry


def panel_segments(geom: Geometry, rows):
    """Segments (A, B) of the panels of the given rows: one per row for thin panels, the four
    edges of the cross-section rectangle for panels of finite thickness."""
    rows = np.atleast_1d(np.asarray(rows))
    if geom.t == 0:
        ax, ay = geom.panel_point(-0.5, rows)
        bx, by = geom.panel_point(0.5, rows)
        return np.stack([ax, ay], -1), np.stack([bx, by], -1)
    A, B = [], []
    for r in rows:
        bf, bb, tf, tb = geom.corners(int(r))          # bottom-front, bottom-back, top-front, top-back
        for a, b in ((bf, tf), (bb, tb), (bf, bb), (tf, tb)):
            A.append(a)
            B.append(b)
    return np.array(A), np.array(B)


def _face(geom: Geometry, chi, side):
    """Points on the outer face of row 0, nudged off the surface by 1e-9."""
    px, py = geom.panel_point(chi, 0, side)
    n_f = np.array([-np.sin(geom.bp), np.cos(geom.bp)])
    n = n_f if side == "front" else -n_f
    return np.stack([px, py], -1) + 1e-9 * n


def ray_hits(origin, direction, A, B, t_max=np.inf, eps=1e-12):
    """True where rays (origin + t*direction, 0 < t < t_max) hit any segment A-B.

    origin, direction: (N, 2); A, B: (M, 2). Returns (N,) bool.
    """
    o = origin[:, None, :]
    d = direction[:, None, :]
    e = (B - A)[None, :, :]
    w = A[None, :, :] - o
    den = d[..., 0] * e[..., 1] - d[..., 1] * e[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[..., 0] * e[..., 1] - w[..., 1] * e[..., 0]) / den
        s = (w[..., 0] * d[..., 1] - w[..., 1] * d[..., 0]) / den
    hit = (np.abs(den) > 1e-15) & (t > eps) & (t < t_max - eps) & (s >= 0) & (s <= 1)
    return hit.any(axis=1)


def sky_dir(alpha):
    """Direction towards a sky point with profile angle alpha in the cross-section."""
    return np.array([-np.cos(alpha), np.sin(alpha)])


def lit_fraction(geom: Geometry, alpha, chi_lo, chi_hi, side, n_sub=200, n_rows=60):
    """Directly lit fraction of the panel element [chi_lo, chi_hi] of row 0."""
    chi = np.linspace(chi_lo, chi_hi, n_sub + 1)
    chi = 0.5 * (chi[1:] + chi[:-1])
    P = _face(geom, chi, side)
    u = sky_dir(alpha)
    n_f = np.array([-np.sin(geom.bp), np.cos(geom.bp)])
    facing = u @ (n_f if side == "front" else -n_f) > 0
    if not facing or alpha >= np.pi - geom.bg:
        return 0.0
    rows = np.array([i for i in range(-n_rows, n_rows + 1) if i != 0 or geom.t > 0])
    A, B = panel_segments(geom, rows)
    blocked = ray_hits(P, np.broadcast_to(u, P.shape), A, B)
    return float(np.mean(~blocked))


def ground_lit(geom: Geometry, alpha, d, n_rows=400, rows=None):
    """Whether ground points d are directly lit for profile angle alpha (rows -n_rows..n_rows,
    or the explicit row offsets ``rows``)."""
    d = np.atleast_1d(d)
    G = np.stack([d * np.cos(geom.bg), d * np.sin(geom.bg)], -1)
    if alpha >= np.pi - geom.bg or alpha <= -geom.bg:      # direction must lie above the slope
        return np.zeros(d.size, bool)
    A, B = panel_segments(geom, np.arange(-n_rows, n_rows + 1) if rows is None else rows)
    u = sky_dir(alpha)
    return ~ray_hits(G, np.broadcast_to(u, G.shape), A, B)


def ground_visible(geom: Geometry, chi, d, side, n_rows=60, rows=None):
    """Whether ground points d are visible from P(chi) on the given side of panel 0 (other rows
    -n_rows..n_rows, or the explicit row offsets ``rows``)."""
    d = np.atleast_1d(d)
    p0 = _face(geom, chi, side)
    G = np.stack([d * np.cos(geom.bg), d * np.sin(geom.bg)], -1)
    v = G - p0
    n_f = np.array([-np.sin(geom.bp), np.cos(geom.bp)])
    n = n_f if side == "front" else -n_f
    rows = range(-n_rows, n_rows + 1) if rows is None else rows
    rows = np.array([i for i in rows if i != 0 or geom.t > 0])     # thick: own stack included
    A, B = panel_segments(geom, rows)
    P = np.broadcast_to(p0, G.shape)
    blocked = ray_hits(np.ascontiguousarray(P), v, A, B, t_max=1.0)
    return (v @ n > 0) & ~blocked
