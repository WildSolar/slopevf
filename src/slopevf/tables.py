"""Multiplication tables: geometry-only relation between profile angle and panel position.

For a profile angle alpha' and a panel position chi, the tables give the
multipliers to the in-plane irradiances of eqns. 37/38 (paper section 2.6):

* ``front_direct`` / ``back_direct``: directly irradiated (unshaded) fraction;
* ``front_ground`` / ``back_ground``: view factor to the directly lit near ground;
* ``far_ground``: view factor to the far ground (front only, independent of alpha');
* ``back_reflected`` / ``front_reflected``: fraction of the panel element that receives the
  specular reflection, off the glass of the adjacent row, of radiation arriving from alpha'
  (front of the uphill row -> back side; back of the downhill row -> front side).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Geometry
from .viewfactors import (far_ground_view_factor, far_ground_view_factor_legacy,
                          lit_ground_view_factor, near_ground_view_factor_legacy, outward,
                          periodic_strips_view_factor, row_offsets)


@dataclass
class Tables:
    alpha: np.ndarray          # (n_alpha,) profile angles [rad]
    chi: np.ndarray            # (n_chi,) panel positions
    front_direct: np.ndarray   # (n_alpha, n_chi)
    back_direct: np.ndarray
    front_ground: np.ndarray
    back_ground: np.ndarray
    far_ground: np.ndarray     # (n_chi,)
    back_reflected: np.ndarray = None    # (n_alpha, n_chi), not available in legacy mode
    front_reflected: np.ndarray = None
    reflected_ground: dict = None        # {(mirror, receiver): (n_alpha, n_chi)}, see below
    # near ground irradiated by the terrain below the horizon (downhill wedge between the
    # horizontal and the slope), per unit terrain radiance; (n_chi,) each, see below
    front_terrain: np.ndarray = None
    back_terrain: np.ndarray = None


def terrain_ground_view_factors(geom: Geometry, chi, side, extent=50.0, rows=None, kernel=None, n=48):
    """Ground-reflected irradiation from the terrain below the horizon, per unit radiance.

    On sloped ground, the ground sees the downhill terrain below the horizon: profile angles
    alpha' in (-beta_g, 0). For a uniform radiance L of that terrain (the far-ground radiance),
    the unshaded ground receives L pi/2 sin(alpha' + beta_g) d alpha' from each band, and the
    panel sees the strips irradiated from alpha' with the lit-ground view factor. Returns
    sum over bands of (pi/2) sin(alpha' + beta_g) d alpha' F_lit(alpha', chi), shape (C,); the
    irradiance contribution is albedo * L * this. In infinite arrays the rows block most of the
    wedge; the ground beyond a finite array receives all of it.
    """
    chi = np.atleast_1d(np.asarray(chi, dtype=float))
    if geom.bg <= 0:
        return np.zeros(chi.size)
    da = geom.bg / n
    a = -geom.bg + (np.arange(n) + 0.5) * da
    w = 0.5 * np.pi * np.sin(a + geom.bg) * da
    F = lit_ground_view_factor(geom, a, chi, side, extent, rows=rows, kernel=kernel)
    return w @ F


def _lit_fraction(eta, chi, width):
    """Element-averaged lit fraction of elements [chi - width/2, chi + width/2] lit above eta."""
    return np.clip((chi + width / 2 + 0.5 - eta) / width, 0.0, 1.0)


def evaluate_tables(geom: Geometry, alpha, chi, width, extent=50.0, rows=None, kernel=None):
    """Table entries for arbitrary profile angles ``alpha`` (A,) at element centres ``chi`` (C,).

    Direct fractions are exact element averages; view factors are evaluated at
    the element centres. Returns four (A, C) arrays.
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    fd, bd = direct_fractions(geom, alpha, chi, width, rows)
    fg = lit_ground_view_factor(geom, alpha, chi, "front", extent, rows=rows, kernel=kernel)
    bg = lit_ground_view_factor(geom, alpha, chi, "back", extent, rows=rows, kernel=kernel)
    return fd, bd, fg, bg


def direct_fractions(geom: Geometry, alpha, chi, width, rows=None):
    """Element-averaged directly irradiated fractions (front, back), each (A, C).

    With ``rows`` (finite array), a side facing away from the array is not shaded by another row:
    it is fully irradiated whenever the direction lies in front of it and above the slope.
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    a1, a2, a3, a4 = geom.alpha
    al = alpha[:, None]
    c = chi[None, :]
    case1 = al < a1
    case2 = (al >= a1) & (al < a2)
    case3 = (al >= a2) & (al < a3)
    case4 = (al >= a3) & (al < a4)
    with np.errstate(divide="ignore", invalid="ignore"):
        fd = np.where(case1, _lit_fraction(geom.shaded_fraction_front(al), c, width),
                      np.where(case2, 1.0, 0.0))
        bd = np.where(case4, _lit_fraction(geom.shaded_fraction_back(al), c, width),
                      np.where(case3, 1.0, 0.0))
    if outward(rows, "front"):
        fd = np.broadcast_to(np.where(al < a2, 1.0, 0.0), fd.shape).copy()
    if outward(rows, "back"):
        bd = np.broadcast_to(np.where((al >= a2) & (al < a4), 1.0, 0.0), bd.shape).copy()
    return fd, bd


def reflected_fractions(geom: Geometry, alpha, chi, width, rows=None):
    """Element-averaged fractions receiving the specular reflection off the adjacent row.

    With ``rows`` (finite array), a side facing away from the array has no mirror (zero).
    Returns ``(back, front)``, each (A, C):

    * back:  back side of row 0, mirror = front side of row +1 (uphill), which is irradiated
      for profile angles in cases (i) and (ii) (partially in case (i));
    * front: front side of row 0, mirror = back side of row -1 (downhill), irradiated in
      cases (iii) and (iv) (partially in case (iv)).

    Mirror and receiver are parallel, so the reflected parallel bundle maps the irradiated
    part of the mirror [lo, 1/2] onto the receiver shifted by a constant k(alpha'):
    chi_mirror = chi + k.
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    a1, a2, a3, a4 = geom.alpha
    t = np.array([np.cos(geom.bp), np.sin(geom.bp)])
    n_f = np.array([-np.sin(geom.bp), np.cos(geom.bp)])
    # row i -> row i+1, from the back face of row i to the front face of row i+1 (thickness)
    D = np.array([geom.rho, geom.rho * np.tan(geom.bg)]) + geom.t * n_f
    u = np.stack([-np.cos(alpha), np.sin(alpha)], axis=-1)         # towards the sky point
    c = chi[None, :]
    out = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for side in ("back", "front"):
            if side == "back":     # mirror: front of row +1
                n_m, d, cases = n_f, D, ((alpha < a1), (alpha >= a1) & (alpha < a2))
                lo_partial = geom.shaded_fraction_front(alpha) - 0.5
            else:                  # mirror: back of row -1
                n_m, d, cases = -n_f, -D, ((alpha >= a3) & (alpha < a4), (alpha >= a2) & (alpha < a3))
                lo_partial = geom.shaded_fraction_back(alpha) - 0.5
            partial, full = cases
            lo = np.where(partial, np.maximum(lo_partial, -0.5), -0.5)
            lit = partial | full
            p = -u                                                    # propagation
            pr = p - 2 * (p @ n_m)[:, None] * n_m                     # reflected propagation
            q = -pr                                                   # from receiver towards mirror
            det = -q[:, 0] * t[1] + q[:, 1] * t[0]
            s = (-d[0] * t[1] + d[1] * t[0]) / det
            k = (q[:, 0] * d[1] - q[:, 1] * d[0]) / det
            ok = lit & (s > 0) & np.isfinite(k) & ((q @ n_m) < 0)       # receiver faces the mirror
            a_lo = (lo - k)[:, None]
            a_hi = (0.5 - k)[:, None]
            overlap = np.minimum(c + width / 2, a_hi) - np.maximum(c - width / 2, a_lo)
            frac = np.clip(overlap / width, 0.0, 1.0)
            out.append(np.where(ok[:, None] & (not outward(rows, side)), frac, 0.0))
    return out[0], out[1]


def reflected_ground_view_factors(geom: Geometry, alpha, chi, extent=50.0, rows=None, kernel=None):
    """View factors to the ground strips irradiated by radiation reflected off the module glass.

    The parallel bundle reflected by the irradiated part of a mirror that does not hit the
    adjacent row reaches the ground (if it descends relative to the slope) in one or two
    strips per row period. Returns a dict ``{(mirror, receiver): (A, C)}`` with mirror in
    {"front", "back"} (reflecting glass) and receiver in {"front", "back"} (panel side that
    views the irradiated ground).
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    a1, a2, a3, a4 = geom.alpha
    t = np.array([np.cos(geom.bp), np.sin(geom.bp)])
    n_f = np.array([-np.sin(geom.bp), np.cos(geom.bp)])
    D = np.array([geom.rho, geom.rho * np.tan(geom.bg)]) + geom.t * n_f     # back face -> next front face
    tg, cg, sg = np.tan(geom.bg), np.cos(geom.bg), np.sin(geom.bg)
    u = np.stack([-np.cos(alpha), np.sin(alpha)], axis=-1)
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for mirror in ("front", "back"):
            if mirror == "front":   # front of row 0, adjacent receiver: back of row -1
                n_m, d = n_f, -D
                partial, full = (alpha < a1), (alpha >= a1) & (alpha < a2)
                lo_partial = geom.shaded_fraction_front(alpha) - 0.5
            else:                   # back of row 0, adjacent receiver: front of row +1
                n_m, d = -n_f, D
                partial, full = (alpha >= a3) & (alpha < a4), (alpha >= a2) & (alpha < a3)
                lo_partial = geom.shaded_fraction_back(alpha) - 0.5
            lit = partial | full
            p_in = -u
            r = p_in - 2 * (p_in @ n_m)[:, None] * n_m                    # reflected propagation
            # ray chi_m t + s r = d + chi_r t  ->  chi_r = chi_m + kappa; the mirror points whose
            # rays hit the adjacent row are chi_m in [k - 1/2, k + 1/2] with k = -kappa
            det = r[:, 0] * t[1] - r[:, 1] * t[0]
            s_adj = (d[0] * t[1] - d[1] * t[0]) / det
            k = (r[:, 0] * d[1] - r[:, 1] * d[0]) / det
            hits_adj = (s_adj > 0) & np.isfinite(k)
            # mirror points whose rays hit the adjacent row: [k - 1/2, k + 1/2] for thin panels;
            # for finite thickness the projection of all four corners of its cross-section
            # (rays passing the receiving face can still hit the edges of the stack)
            if geom.t > 0:
                m0 = np.array(geom.panel_point(0.0, 0, mirror))
                cr = t[0] * r[:, 1] - t[1] * r[:, 0]                        # cross(t, r)
                chi_q = [((qx - m0[0]) * r[:, 1] - (qy - m0[1]) * r[:, 0]) / cr
                         for qx, qy in geom.corners(-1 if mirror == "front" else +1)]
                adj_lo, adj_hi = np.minimum.reduce(chi_q), np.maximum.reduce(chi_q)
            else:
                adj_lo, adj_hi = k - 0.5, k + 0.5
            # projection of mirror points onto the ground along r (affine in chi_m)
            descent = r[:, 1] - r[:, 0] * tg                                # < 0: reaches the ground

            def ground(chi_m):
                o = 0.5 * geom.t * (1.0 if mirror == "front" else -1.0)      # face offset
                qx, qy = chi_m * t[0] + o * n_f[0], geom.h + chi_m * t[1] + o * n_f[1]
                s = -(qy - qx * tg) / descent
                return (qx + s * r[:, 0]) * cg + (qy + s * r[:, 1]) * sg

            # The adjacent row shades the mirror and intercepts part of the reflected bundle. In a
            # finite array, the outermost row on that side has no adjacent row.
            if rows is None:
                variants = [(True, None)]
            else:
                offsets = row_offsets(rows)
                edge = offsets[0] if mirror == "front" else offsets[-1]
                variants = [(True, offsets[offsets != edge]), (False, [edge])]
            strips = []
            for neighbour, rows_i in variants:
                if neighbour:
                    lo = np.where(partial, np.maximum(lo_partial, -0.5), -0.5)
                    k_lo = np.where(hits_adj, adj_lo, np.inf)
                    k_hi = np.where(hits_adj, adj_hi, np.inf)
                else:
                    lo = np.full_like(alpha, -0.5)
                    k_lo = k_hi = np.full_like(alpha, np.inf)
                # remaining irradiated mirror intervals: [lo, min(1/2, k_lo)] and [max(lo, k_hi), 1/2]
                for c0, c1 in [(lo, np.minimum(0.5, k_lo)), (np.maximum(lo, k_hi), np.full_like(lo, 0.5))]:
                    ok = lit & (descent < 0) & (c1 > c0)
                    g0, g1 = ground(np.where(ok, c0, 0.0)), ground(np.where(ok, c1, 0.0))
                    strips.append((np.where(ok, np.minimum(g0, g1), np.nan),
                                   np.where(ok, np.abs(g1 - g0), 0.0), rows_i))

            for receiver in ("front", "back"):
                total = np.zeros((alpha.size, chi.size))
                for start, width, rows_i in strips:
                    total += periodic_strips_view_factor(geom, start, width, chi, receiver, extent,
                                                         offsets=rows_i, kernel=kernel,
                                                         outward=outward(rows, receiver))
                out[(mirror, receiver)] = total
    return out


def build_tables(geom: Geometry, n_alpha: int, n_chi: int, extent=50.0, rows=None, kernel=None,
                 reflection=True) -> Tables:
    """Tables on a uniform profile-angle grid [0, pi] and panel element centres.

    ``kernel``: transmission-weighted view-factor kernel of the receiving glass (ground view
    factors); ``reflection``: whether to build the reflected-ground view factors.
    """
    alpha = np.linspace(0.0, np.pi, n_alpha + 1)
    width = 1.0 / n_chi
    chi = -0.5 + (np.arange(n_chi) + 0.5) * width
    fd, bd, fg, bg = evaluate_tables(geom, alpha, chi, width, extent, rows, kernel)
    rb, rf = reflected_fractions(geom, alpha, chi, width, rows)
    tables = Tables(alpha, chi, fd, bd, fg, bg, far_ground_view_factor(geom, chi, kernel, outward(rows, "front")),
                    rb, rf)
    tables.front_terrain = terrain_ground_view_factors(geom, chi, "front", extent, rows, kernel)
    tables.back_terrain = terrain_ground_view_factors(geom, chi, "back", extent, rows, kernel)
    if reflection:
        tables.reflected_ground = reflected_ground_view_factors(geom, alpha, chi, extent, rows, kernel)
    return tables


def build_tables_legacy(geom: Geometry, n_sigma: int, n_chi: int, n_d: int, n_zeta: int,
                        max_rows: int, n_phi: int, n_sig: int) -> Tables:
    """Reproduce buildMultiplicationTable.m (vectorised)."""
    mu, gamma, rho_bar, h_bar = geom.bp, geom.bg, geom.rho, geom.h
    chi_vec = np.linspace(-0.5, 0.5, n_chi + 1)
    sigma_vec = np.linspace(0, 180, n_sigma + 1) / 180 * np.pi
    S, C = sigma_vec.size, chi_vec.size

    sc1 = np.arctan((np.sin(mu) - rho_bar * np.tan(gamma)) / (rho_bar - np.cos(mu)))
    sc2 = np.pi - mu
    sc3 = np.pi - np.arctan((np.sin(mu) + rho_bar * np.tan(gamma)) / (rho_bar + np.cos(mu)))
    sc4 = np.pi - gamma
    d_chi = abs(chi_vec[1] - chi_vec[0])

    sig = sigma_vec[:, None]
    ch = chi_vec[None, :]
    c1 = sig < sc1
    c2 = ~c1 & (sig < sc2)
    c3 = ~c1 & ~c2 & (sig < sc3)
    c4 = ~c1 & ~c2 & ~c3 & (sig < sc4)

    with np.errstate(all="ignore"):
        ratio = rho_bar * (np.tan(gamma) + np.tan(sig)) / (np.sin(mu) + np.tan(sig) * np.cos(mu))
        eta_f, eta_b = 1 - ratio, 1 + ratio
        fd = np.where(c1, np.maximum(0, np.minimum(1, (ch + 1 / 2 - eta_f) / d_chi)), 0.0)
        fd = np.where(c2, 1.0, fd)
        bd = np.where(c4, np.minimum(1, np.maximum(0, (ch + 1 / 2 - eta_b) / d_chi)), 0.0)
        bd = np.where(c3, 1.0, bd)

        two_chi = 2 * chi_vec
        df_min = ((h_bar * ((two_chi + 1) * np.cos(mu) + 2 * rho_bar)
                   + two_chi * rho_bar * (np.sin(mu) - np.tan(gamma) * np.cos(mu)))
                  / ((two_chi + 1) * np.sin(gamma - mu)))
        df_max = -h_bar / (np.cos(gamma) * (np.tan(mu) - np.tan(gamma)))
        db_min = df_max
        db_max = ((h_bar * ((two_chi + 1) * np.cos(mu) - 2 * rho_bar)
                   + two_chi * rho_bar * (np.tan(gamma) * np.cos(mu) - np.sin(mu)))
                  / ((two_chi + 1) * np.sin(gamma - mu)))
        tan_s = np.tan(sigma_vec)
        d_10 = (2 * h_bar - np.sin(mu) - tan_s * np.cos(mu)) / (2 * np.cos(gamma) * tan_s + 2 * np.sin(gamma))
        d_20 = (2 * h_bar + np.sin(mu) + tan_s * np.cos(mu)) / (2 * np.cos(gamma) * tan_s + 2 * np.sin(gamma))

    def clamp(i):
        return np.fmax(-max_rows, np.fmin(max_rows, i))

    # collect all ground intervals: (sigma index, chi index, sign, lo, hi)
    parts = []
    with np.errstate(all="ignore"):
        for case, (e_lo, e_hi) in ((c2[:, 0], (d_20, d_10)), (c3[:, 0], (d_10, d_20))):
            si = np.flatnonzero(case)
            if si.size == 0:
                continue
            lo_edge, hi_edge = e_lo[si, None], e_hi[si, None]   # strip i: (lo_edge+(i-1)p, hi_edge+ip)
            for side, dmin, dmax, sign in (("f", df_min[None, :], df_max, -1.0),
                                           ("b", db_min, db_max[None, :], 1.0)):
                i_max = clamp(np.ceil((dmax - lo_edge) * np.cos(gamma) / rho_bar))
                i_min = clamp(np.floor((dmin - hi_edge) * np.cos(gamma) / rho_bar) + 1)
                i_min, i_max = np.broadcast_arrays(i_min, i_max)
                count = np.maximum(0, i_max - i_min + 1).astype(int)
                s_idx = np.repeat(np.repeat(si, C), count.ravel())
                c_idx = np.repeat(np.tile(np.arange(C), si.size), count.ravel())
                start = np.repeat(i_min.ravel(), count.ravel())
                offs = np.arange(count.sum()) - np.repeat(np.cumsum(count.ravel()) - count.ravel(),
                                                          count.ravel())
                rows = start + offs
                lo_e = np.broadcast_to(lo_edge, i_min.shape).ravel()
                hi_e = np.broadcast_to(hi_edge, i_min.shape).ravel()
                dmn = np.broadcast_to(dmin, i_min.shape).ravel()
                dmx = np.broadcast_to(dmax, i_min.shape).ravel()
                rep = lambda x: np.repeat(x, count.ravel())  # noqa: E731
                lo = np.fmax(rep(lo_e) + (rows - 1) * rho_bar / np.cos(gamma), rep(dmn))
                hi = np.fmin(rep(hi_e) + rows * rho_bar / np.cos(gamma), rep(dmx))
                parts.append((side, s_idx, c_idx, sign, lo, hi))

    fg = np.zeros((S, C))
    bg = np.zeros((S, C))
    for side, s_idx, c_idx, sign, lo, hi in parts:
        vf = near_ground_view_factor_legacy(chi_vec[c_idx], lo, hi, mu, gamma, h_bar, n_d, n_zeta)
        np.add.at(fg if side == "f" else bg, (s_idx, c_idx), sign * vf)

    far = far_ground_view_factor_legacy(chi_vec, mu, gamma, rho_bar, n_phi, n_sig)
    return Tables(sigma_vec, chi_vec, fd, bd, fg, bg, far)
