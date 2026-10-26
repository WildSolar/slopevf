"""Geometry (shading cases, shadows, view windows) against brute-force ray tracing."""

import numpy as np
import pytest

from slopevf import Array
from slopevf.geometry import Geometry, profile_angle
from slopevf.tables import evaluate_tables

import raytrace


def test_shading_angle_roundtrip():
    a = Array.from_shading_angle(60, 30, 15, 1.0)
    assert a.rho_bar == pytest.approx(1.183, abs=5e-4)   # paper section 4.2
    assert a.shading_angle == pytest.approx(15)
    assert Array.from_shading_angle(60, 0, 15, 1.0).rho_bar == pytest.approx(3.732, abs=5e-4)
    assert Array(60, 30, 1.04, 2.04).shading_angle == pytest.approx(26.2, abs=0.05)  # Fig. 7


def test_invalid_geometry():
    with pytest.raises(ValueError):
        Array(20, 20, 1.0, 1.0)          # beta_p must exceed beta_g
    with pytest.raises(ValueError):
        Array(60, 20, 1.0, 0.2)          # lower edge below ground


def test_profile_angle():
    # sun in the downhill direction: profile angle = elevation
    assert profile_angle(0.3, 0.2, 0.2) == pytest.approx(0.3)
    # sun on the uphill side: pi - elevation
    assert profile_angle(0.3, 0.2 + np.pi, 0.2) == pytest.approx(np.pi - 0.3)
    # sun along the rows: 90 deg
    assert profile_angle(0.3, 0.2 + np.pi / 2, 0.2) == pytest.approx(np.pi / 2)


def test_direct_fractions(array):
    """Element-averaged direct fractions of the tables match ray tracing."""
    g = Geometry(array)
    n_chi = 10
    width = 1 / n_chi
    chi = -0.5 + (np.arange(n_chi) + 0.5) * width
    alphas = np.concatenate([np.linspace(0.01, np.pi - g.bg - 0.01, 37), g.alpha[:3] + 1e-3])
    fd, bd, _, _ = evaluate_tables(g, alphas, chi, width)
    for i, a in enumerate(alphas):
        for j, c in enumerate(chi):
            ref_f = raytrace.lit_fraction(g, a, c - width / 2, c + width / 2, "front")
            ref_b = raytrace.lit_fraction(g, a, c - width / 2, c + width / 2, "back")
            assert fd[i, j] == pytest.approx(ref_f, abs=0.011), (a, c)
            assert bd[i, j] == pytest.approx(ref_b, abs=0.011), (a, c)


def test_ground_shadow(array):
    """Lit ground strips (periodic shadow of the panels) match ray tracing."""
    g = Geometry(array)
    a1, _, a3, _ = g.alpha
    d = np.linspace(-6, 6, 1201)
    for a in np.linspace(max(a1, 0) + 0.02, a3 - 0.02, 15):
        d1, d2 = g.shadow_edges(a)
        lo, w = min(d1, d2), abs(d2 - d1)
        shaded = np.mod(d - lo, g.p) < w
        ref = raytrace.ground_lit(g, a, d)
        # ignore samples right at a shadow edge
        edge = np.minimum(np.abs(np.mod(d - lo, g.p)), np.abs(np.mod(d - lo - w, g.p)))
        ok = (edge > 1e-6) & (g.p - edge > 1e-6)
        assert np.array_equal(~shaded[ok], ref[ok]), a


@pytest.mark.parametrize("side", ["front", "back"])
def test_view_windows(array, side):
    """The visible ground interval [lo, hi] matches ray tracing and (thin panels) the paper's
    formulas."""
    g = Geometry(array)
    d0 = g.d_face(side)
    for chi in (-0.4, -0.1, 0.2, 0.45):
        lo, hi = g.view_window(chi, side)
        span_lo = max(lo, d0 - 40) if side == "front" else d0 - 5
        span_hi = d0 + 5 if side == "front" else min(hi, d0 + 40)
        d = np.linspace(span_lo - 2, span_hi + 2, 2001)
        vis = raytrace.ground_visible(g, chi, d, side)
        inside = (d > lo + 1e-6) & (d < hi - 1e-6)
        outside = (d < lo - 1e-6) | (d > hi + 1e-6)
        assert vis[inside].all() and not vis[outside].any(), (chi, lo, hi)
        if g.t > 0:
            continue

        # paper eqns. 23-26
        mu, gam, rho, h = g.bp, g.bg, g.rho, g.h
        if side == "front":
            ref = ((h * ((2 * chi + 1) * np.cos(mu) + 2 * rho)
                    + 2 * chi * rho * (np.sin(mu) - np.tan(gam) * np.cos(mu)))
                   / ((2 * chi + 1) * np.sin(gam - mu)))
            assert lo == pytest.approx(ref)
        else:
            ref = ((h * ((2 * chi + 1) * np.cos(mu) - 2 * rho)
                    + 2 * chi * rho * (np.tan(gam) * np.cos(mu) - np.sin(mu)))
                   / ((2 * chi + 1) * np.sin(gam - mu)))
            assert hi == pytest.approx(ref)


def test_shading_fractions_match_paper(array):
    """Rewritten eta_f/eta_b and shadow edges equal eqns. 11, 13, 14 (thin panels)."""
    if array.thickness > 0:
        pytest.skip("paper equations hold for thin panels; thick panels are checked by ray tracing")
    g = Geometry(array)
    mu, gam, rho, h = g.bp, g.bg, g.rho, g.h
    a = np.linspace(0.05, np.pi - gam - 0.05, 50)
    a = a[np.abs(a - np.pi / 2) > 1e-3]
    ratio = rho * (np.tan(gam) + np.tan(a)) / (np.sin(mu) + np.tan(a) * np.cos(mu))
    ok = np.abs(np.sin(a + mu)) > 1e-3
    np.testing.assert_allclose(g.shaded_fraction_front(a)[ok], (1 - ratio)[ok], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(g.shaded_fraction_back(a)[ok], (1 + ratio)[ok], rtol=1e-10, atol=1e-10)
    den = 2 * np.cos(gam) * np.tan(a) + 2 * np.sin(gam)
    d1, d2 = g.shadow_edges(a)
    np.testing.assert_allclose(d1, (2 * h - np.sin(mu) - np.tan(a) * np.cos(mu)) / den, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(d2, (2 * h + np.sin(mu) + np.tan(a) * np.cos(mu)) / den, rtol=1e-9, atol=1e-9)
