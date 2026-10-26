"""Sky radiance: slopevf's wrapper around perezsky (corrected mode) and the legacy coefficients.

The Perez model itself (parameters, limits, efficacy, Radiance validation) is tested in the
perezsky package.
"""

import numpy as np
import perezsky
import pytest

from slopevf.geometry import direction_vector, surface_normal
from slopevf.sky import (PEREZ_COEFFICIENTS, PEREZ_COEFFICIENTS_LEGACY, _perez_parameters_legacy,
                         extraterrestrial_irradiance, make_sky_grid, sky_radiance)


def test_extraterrestrial_irradiance():
    # Earth-Sun distance: perihelion early January (+3.4 %), aphelion early July (-3.4 %)
    assert extraterrestrial_irradiance(3) == pytest.approx(1408.7, abs=1.0)
    assert extraterrestrial_irradiance(185) == pytest.approx(1315.5, abs=1.0)
    days = np.arange(1, 366)
    assert extraterrestrial_irradiance(days).mean() == pytest.approx(1361.0 * 1.00011, rel=1e-6)
    assert extraterrestrial_irradiance(100, solar_constant=1367.0) == pytest.approx(
        1367.0 / 1361.0 * extraterrestrial_irradiance(100))


def test_sky_brightness_uses_day_of_year():
    """A higher extraterrestrial irradiance (January) lowers the Perez sky brightness Delta."""
    g = make_sky_grid("low")
    args = (g, [300.0], [500.0], [0.0], [np.radians(40.0)])
    a = sky_radiance(*args, day_of_year=185).L
    b = sky_radiance(*args, day_of_year=3).L
    assert not np.allclose(a, b)
    # both remain normalised to the same diffuse horizontal irradiance
    np.testing.assert_allclose(g.horizontal_weight @ a, g.horizontal_weight @ b, rtol=1e-10)


@pytest.mark.parametrize("res", ["low", "medium", (1.0, 120)])
def test_grid_solid_angle(res):
    g = make_sky_grid(res)
    assert g.solid_angle.sum() == pytest.approx(2 * np.pi, rel=1e-12)
    assert g.horizontal_weight.sum() == pytest.approx(np.pi, rel=1e-4)
    np.testing.assert_allclose(np.linalg.norm(g.vec, axis=1), 1.0)


def test_grid_is_perezsky_grid():
    """Same cells as perezsky, converted to slopevf's frame (x = west, y = south)."""
    g = make_sky_grid("low")
    p = perezsky.make_sky_grid("low")
    np.testing.assert_allclose(g.vec[:, 0], -p.vec[:, 0], atol=1e-12)
    np.testing.assert_allclose(g.vec[:, 1], -p.vec[:, 1], atol=1e-12)
    np.testing.assert_allclose(g.vec[:, 2], p.vec[:, 2], atol=1e-12)
    assert np.all(np.abs(g.azimuth) <= np.pi)


def test_medium_grid_size():
    # the paper's validation grid has 20'672 cells of roughly 1 deg^2
    assert 19_000 < make_sky_grid("medium").size < 22_000


@pytest.mark.parametrize("sun_model", ["buie", "point"])
def test_energy_closure(sun_model):
    g = make_sky_grid("medium")
    dni = np.array([0.0, 50.0, 400.0, 850.0, 300.0])
    ghi = np.array([80.0, 150.0, 450.0, 800.0, 200.0])
    el = np.radians([30.0, 15.0, 40.0, 62.0, 3.0])
    az = np.radians([0.0, -60.0, 20.0, 5.0, 100.0])
    sky = sky_radiance(g, dni, ghi, az, el, sun_model=sun_model)
    h = g.horizontal_weight @ sky.L
    if sun_model == "point":
        h += dni * np.sin(el)
    np.testing.assert_allclose(h, ghi, rtol=1e-10)
    assert (sky.L >= 0).all()


def test_wrapper_conventions():
    """Irradiance on a plane is the same with slopevf's wrapper and with perezsky directly."""
    g = make_sky_grid("low")
    dni, ghi = np.array([700.0, 200.0]), np.array([500.0, 260.0])
    el, az_s = np.radians([30.0, 50.0]), np.radians([-60.0, 45.0])      # from south, + west
    L = sky_radiance(g, dni, ghi, az_s, el, sun_model="point").L
    for tilt, plane_az_s in ((90.0, -90.0), (60.0, 45.0)):              # facing east; south-west
        n = surface_normal(np.radians(tilt), np.radians(plane_az_s))
        mine = (g.solid_angle * np.maximum(g.vec @ n, 0)) @ L
        sky = perezsky.perez_sky(dni, 90 - np.degrees(el), np.degrees(az_s) + 180, ghi=ghi, grid="low")
        ref = sky.plane_irradiance(tilt, plane_az_s + 180, albedo=0.0)["sky"][:, 0]
        np.testing.assert_allclose(mine, ref, rtol=1e-10)
        # and the sun direction
        s = direction_vector(el, az_s)
        np.testing.assert_allclose(s[:, 2], sky.sun_vector[:, 2], atol=1e-12)
        np.testing.assert_allclose(s[:, :2], -sky.sun_vector[:, :2], atol=1e-12)


def test_perez_coefficients():
    np.testing.assert_array_equal(PEREZ_COEFFICIENTS, perezsky.LUMINANCE_1993)
    diff = np.argwhere(PEREZ_COEFFICIENTS != PEREZ_COEFFICIENTS_LEGACY)
    assert sorted(map(tuple, diff)) == [(0, 1, 3), (5, 2, 2), (6, 3, 3)]


def test_bright_sky_bounded():
    """Very bright overcast sky (Totalp 2025-03-16): the radiance near the horizon stays bounded
    (unlimited Perez parameters give b > 0 and a diverging horizon)."""
    raw = perezsky.perez_parameters(1.6, 681.3, 48.6, clamp=False, limit_gradation=False)
    assert raw.b[0] > 0
    s = sky_radiance(make_sky_grid("low"), [1.6], [682.4], [0.076], [np.radians(41.4)], sun_model="point")
    assert s.L.max() < 10 * 682.4 / np.pi


def test_legacy_overcast_bin():
    """MATLAB category-1 formula: exp(Delta (x1 + x2 Z)^x3) - 1 instead of exp((Delta (x1 + x2 Z))^x3) - x4."""
    z, dhi = np.array([0.6]), np.array([200.0])
    par = _perez_parameters_legacy(dhi, np.array([0.0]), z, np.array([1.2]))
    delta = 1.2 * 200 / 1361
    c = PEREZ_COEFFICIENTS[0, 2]
    assert par[2, 0] == pytest.approx(np.exp(delta * (c[0] + c[1] * 0.6) ** c[2]) - 1)
    corrected = perezsky.perez_parameters(0.0, 200.0, np.degrees(0.6), clamp=False, brightness_correction=False)
    assert corrected.c[0] != pytest.approx(par[2, 0], rel=1e-3)
