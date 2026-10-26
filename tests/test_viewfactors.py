"""Closed-form view factors against numerical integration and brute force."""

import numpy as np
import pytest

from slopevf import Array
from slopevf.geometry import Geometry
from slopevf.viewfactors import (far_ground_view_factor, lit_ground_view_factor,
                                 near_ground_view_factor_numeric, strip_view_factor)

import raytrace


@pytest.mark.parametrize("chi", [-0.3, 0.0, 0.4])
def test_strip_vf_equals_eqn26(array, chi):
    """1/2 |sin t2 - sin t1| equals the double integral of eqn. 26 (both sides)."""
    g = Geometry(array)
    for side, sign in (("back", 1.0), ("front", -1.0)):
        lo, hi = g.view_window(chi, side)
        lo, hi = max(lo, g.d_plane - 8), min(hi, g.d_plane + 8)
        d1, d2 = lo + 0.2 * (hi - lo), lo + 0.7 * (hi - lo)
        num = sign * near_ground_view_factor_numeric(g, chi, d1, d2, n_d=3000, n_zeta=3000)
        assert strip_view_factor(g, chi, d1, d2) == pytest.approx(num, rel=2e-3)


def test_half_space_limits():
    """A vertical panel sees half of the space in front of it."""
    g = Geometry(Array(90, 0, 1e6, 1.0))
    # whole ground plane from the panel plane to -inf for a vertical panel on flat ground
    assert strip_view_factor(g, 0.0, -np.inf, g.d_plane) == pytest.approx(0.5)


@pytest.mark.parametrize("chi", [-0.45, -0.2, 0.1, 0.35, 0.49])
def test_far_ground_vf(array, chi):
    """Closed form equals a direct solid-angle integration of the far-ground wedge.

    Directions v = (-cos(s) cos(p), -sin(s), cos(s) sin(p)) with horizontal angle p
    and depression s; d(omega) = cos(s) ds dp. The wedge is bounded by the
    horizontal and the top edge of the row in front.
    """
    g = Geometry(array)
    px, py = g.panel_point(chi, 0, "front")
    # upper silhouette of row -1: its top corner seen highest from P (thin: the top edge)
    tops = g.corners(-1)[2:]
    tx, ty = tops[np.argmax([np.arctan2(y - py, px - x) for x, y in tops])]
    dh, dx = py - ty, px - tx
    n = 2000
    p = -np.pi / 2 + (np.arange(n) + 0.5) * np.pi / n
    s_max = np.arctan(np.maximum(dh, 0) * np.cos(p) / dx)
    F = 0.0
    for k in range(n):
        s = (np.arange(n) + 0.5) * s_max[k] / n
        cos_t = np.sin(g.bp) * np.cos(s) * np.cos(p[k]) - np.cos(g.bp) * np.sin(s)
        F += np.sum(np.maximum(cos_t, 0) * np.cos(s)) * s_max[k] / n * np.pi / n
    F /= np.pi
    assert far_ground_view_factor(g, chi) == pytest.approx(F, rel=1e-3, abs=1e-6)


def test_lit_ground_vf_brute_force(array):
    """Aggregated lit-ground view factor matches ray-traced lit/visible ground integration."""
    g = Geometry(array)
    a1, a2, a3, _ = g.alpha
    alphas = np.linspace(max(a1, 0) + 0.03, a3 - 0.03, 4)
    for side in ("front", "back"):
        for chi in (-0.35, 0.4):
            F = lit_ground_view_factor(g, alphas, np.array([chi]), side)[:, 0]
            lo, hi = g.view_window(chi, side)
            lo, hi = max(lo, -80.0), min(hi, 80.0)
            d = np.linspace(lo, hi, 16001)
            dm = 0.5 * (d[1:] + d[:-1])
            ds = np.abs(np.diff(g.sin_view_angle(chi, d, side))) / 2
            for a, f in zip(alphas, F):
                lit = raytrace.ground_lit(g, a, dm, n_rows=120)
                ref = np.sum(ds[lit])
                assert f == pytest.approx(ref, abs=2e-3), (side, chi, a)


@pytest.mark.parametrize("rows, sides", [
    ((2, 0), ("front",)),           # Totalp: front module on the last (uphill) of three rows
    ((0, 2), ("back",)),            # Totalp: back module on the first (downhill) of three rows
    ((1, 1), ("front", "back")),    # centre of three rows
    ((3, 1), ("front", "back")),
    ((0, 2), ("front",)),           # outward-facing sides: front of the first row,
    ((2, 0), ("back",)),            # back of the last row
    ((0, 0), ("front", "back")),    # single row
])
def test_lit_ground_vf_finite_rows(array, rows, sides):
    """Finite array: the lit-ground view factor matches 2D ray tracing of the visible and lit
    ground with only the existing rows, for all sun angles (including those where the infinite
    array leaves no ground lit), also for the sides facing away from the array."""
    g = Geometry(array)
    offsets = np.arange(-rows[0], rows[1] + 1)
    alphas = np.linspace(0.03, np.pi - g.bg - 0.03, 9)
    if g.bg > 0:      # downhill terrain below the horizon (alpha' in (-beta_g, 0))
        alphas = np.concatenate([np.linspace(-g.bg + 0.02, -0.02, 3), alphas])
    for side in sides:
        for chi in (-0.35, 0.4):
            F = lit_ground_view_factor(g, alphas, np.array([chi]), side, rows=rows)[:, 0]
            # reference: ray-traced visibility (independent of view_window) over a wide range
            # (fine near the panel: dense arrays have visible windows of a few panel heights)
            d = np.concatenate([np.linspace(-1e5, -400, 4001)[:-1],
                                np.linspace(-400, -60, 4001)[:-1], np.linspace(-60, -5, 11001)[:-1],
                                np.linspace(-5, 5, 50001), np.linspace(5, 60, 11001)[1:],
                                np.linspace(60, 400, 4001)[1:], np.linspace(400, 1e5, 4001)[1:]])
            dm = 0.5 * (d[1:] + d[:-1])
            vis = raytrace.ground_visible(g, chi, dm, side, rows=offsets)
            ds = np.abs(np.diff(g.sin_view_angle(chi, d, side))) / 2
            for a, f in zip(alphas, F):
                lit = raytrace.ground_lit(g, a, dm, rows=offsets)
                assert f == pytest.approx(np.sum(ds[vis & lit]), abs=2e-3), (rows, side, chi, a)


def test_lit_ground_vf_many_rows_is_infinite(array):
    """With many rows, the finite-array result approaches the infinite-row model."""
    g = Geometry(array)
    alphas = np.linspace(0.03, np.pi - g.bg - 0.03, 40)
    chi = np.array([-0.45, 0.0, 0.45])
    for side in ("front", "back"):
        inf = lit_ground_view_factor(g, alphas, chi, side)
        fin = lit_ground_view_factor(g, alphas, chi, side, rows=(400, 400))
        np.testing.assert_allclose(fin, inf, atol=2e-3)


def test_lit_ground_vf_extent_convergence(array):
    """The far-region approximation is accurate: results barely change with the exact range."""
    g = Geometry(array)
    alphas = np.linspace(0.05, np.pi - g.bg - 0.05, 60)
    chi = np.array([-0.49, -0.3, 0.0, 0.45])
    for side in ("front", "back"):
        a = lit_ground_view_factor(g, alphas, chi, side, extent=50.0)
        b = lit_ground_view_factor(g, alphas, chi, side, extent=2000.0)
        np.testing.assert_allclose(a, b, atol=2e-4)
