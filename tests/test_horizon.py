import numpy as np
import pytest

from slopevf import Array, FarGround, Horizon, Settings, simulate
from slopevf.model import ArrayModel
from slopevf.sky import make_sky_grid, sky_radiance

ARR = Array(60, 20, 1.3, 1.2)
WEATHER = dict(dni=[700.0, 300.0, 0.0], ghi=[600.0, 350.0, 180.0],
               solar_azimuth=[-35.0, 70.0, 150.0], solar_elevation=[25.0, 50.0, 30.0])
S = Settings(sky="low", n_chi=20)


def flat(elevation, **kw):
    az = np.arange(-180.0, 180.0, 5.0)
    return Horizon(az, np.full(az.shape, elevation), **kw)


def test_interpolation_and_clipping():
    h = Horizon([-90.0, 0.0, 90.0, 180.0], [10.0, 20.0, -3.0, 0.0])
    assert np.degrees(h.elevation_at(np.radians(-45.0))) == pytest.approx(15.0)
    assert np.degrees(h.elevation_at(np.radians(90.0))) == pytest.approx(0.0)      # negative -> 0
    # periodic: between 180 (= -180) and -90
    assert np.degrees(h.elevation_at(np.radians(-135.0))) == pytest.approx(5.0)
    assert np.degrees(h.elevation_at(np.radians(225.0))) == pytest.approx(5.0)


def test_from_file_conventions(tmp_path):
    f = tmp_path / "test.hor"
    np.savetxt(f, np.c_[np.arange(-179.5, 180.0, 1.0), np.linspace(0, 10, 360)])
    south = Horizon.from_file(f)
    north = Horizon.from_file(f, azimuth_convention="north")
    # azimuth a in the 'north' convention corresponds to a - 180 in the paper convention
    a = np.radians(30.5)
    assert north.elevation_at(a - np.pi) == pytest.approx(south.elevation_at(a))
    with pytest.raises(ValueError):
        Horizon.from_file(f, azimuth_convention="east")


@pytest.mark.parametrize("elev", [0.0, -1.5])
def test_zero_horizon_is_no_horizon(elev):
    kw = dict(albedo=0.6, far_ground=FarGround(15, 160), albedo_fg=0.3, settings=S)
    a = simulate(ARR, WEATHER, **kw)
    b = simulate(ARR, WEATHER, horizon=flat(elev), albedo_terrain=0.4, **kw)
    np.testing.assert_allclose(a.front, b.front)
    np.testing.assert_allclose(a.back, b.back)


def test_sun_behind_horizon_black_terrain():
    """Only beam irradiation (DHI = 0), sun behind a 40 deg horizon, black terrain: nothing arrives."""
    w = dict(dni=[700.0], ghi=[700.0 * np.sin(np.radians(25.0))], solar_azimuth=[-35.0], solar_elevation=[25.0])
    r = simulate(ARR, w, albedo=0.6, horizon=flat(40.0), albedo_terrain=0.0, settings=S)
    assert np.allclose(r.front, 0) and np.allclose(r.back, 0)
    r = simulate(ARR, w, albedo=0.6, horizon=flat(10.0), albedo_terrain=0.0, settings=S)
    assert r.front.mean() > 100


@pytest.mark.parametrize("terrain", ["horizontal", "facing"])
def test_matches_manual_masking(terrain):
    """simulate() equals a direct evaluation with the masked sky radiance replaced by terrain radiance."""
    h = Horizon([-180, -90, 0, 90], [5.0, 25.0, 12.0, 30.0], terrain=terrain)
    alb_t = 0.4
    r = simulate(ARR, WEATHER, albedo=0.6, horizon=h, albedo_terrain=alb_t, settings=S)

    grid = make_sky_grid("low")
    model = ArrayModel(ARR, FarGround(), grid, S)
    dni, ghi = np.array(WEATHER["dni"]), np.array(WEATHER["ghi"])
    el, az = np.radians(WEATHER["solar_elevation"]), np.radians(WEATHER["solar_azimuth"])
    L = sky_radiance(grid, dni, ghi, az, el, sun_model="point").L
    m = h.masked(grid.elevation, grid.azimuth)
    L[m] = h.terrain_radiance(grid.elevation[m], grid.azimuth[m], np.full(3, alb_t), dni, ghi, el, az)
    vis = ~h.masked(el, az)
    out = model.apply(L, np.full(3, 0.6), np.zeros(3), (dni * vis, el, az))
    np.testing.assert_allclose(r.front, out["front_direct"] + out["front_nearground"] + out["front_farground"],
                               rtol=1e-12)
    np.testing.assert_allclose(r.back, out["back_direct"] + out["back_nearground"], rtol=1e-12)


def test_facing_flat_terrain_equals_horizontal():
    """Facing terrain with zero tilt is horizontal terrain (if the sun is visible)."""
    kw = dict(albedo=0.6, albedo_terrain=0.4, settings=S)
    az = np.arange(-180.0, 180.0, 5.0)
    el = 8 + 6 * np.sin(np.radians(2 * az))                     # sun (25-50 deg) always above
    a = simulate(ARR, WEATHER, horizon=Horizon(az, el, terrain="horizontal"), **kw)
    b = simulate(ARR, WEATHER, horizon=Horizon(az, el, terrain="facing", terrain_tilt=0.0), **kw)
    np.testing.assert_allclose(a.front, b.front, rtol=1e-10)
    np.testing.assert_allclose(a.back, b.back, rtol=1e-10)


def test_horizon_reduces_irradiation_and_terrain_compensates():
    kw = dict(albedo=0.6, settings=S)
    none = simulate(ARR, WEATHER, **kw)
    black = simulate(ARR, WEATHER, horizon=flat(15.0), albedo_terrain=0.0, **kw)
    bright = simulate(ARR, WEATHER, horizon=flat(15.0), albedo_terrain=0.8, **kw)
    assert (black.front_mean < none.front_mean).all()
    assert (bright.front_mean > black.front_mean).all()


def test_default_terrain_model():
    h = Horizon([0.0, 90.0], [5.0, 10.0])
    assert h.terrain == "facing" and h.terrain_tilt == 30.0


def test_requires_terrain_albedo_and_no_legacy():
    with pytest.raises(ValueError):
        simulate(ARR, WEATHER, albedo=0.6, horizon=flat(10.0), settings=S)
    with pytest.raises(ValueError):
        simulate(ARR, WEATHER, albedo=0.6, horizon=flat(10.0), albedo_terrain=0.3,
                 settings=Settings.matlab(sky="low", n_chi=10, n_alpha=60, n_d=10, n_zeta=10, n_phi=10, n_sig=10))
