import warnings

import numpy as np
import pandas as pd
import pytest

from slopevf import Array, FarGround, Settings, prepare_weather, simulate
from slopevf.geometry import surface_normal
from slopevf.sky import make_sky_grid, sky_radiance

WEATHER = dict(dni=[550.0, 800.0, 100.0, 300.0], ghi=[200.0, 700.0, 180.0, 250.0],
               solar_azimuth=[-40.0, 5.0, 60.0, 120.0], solar_elevation=[11.0, 45.0, 30.0, 20.0])


def test_isolated_panel_matches_transposition():
    """With rows very far apart, front-side sky irradiance is a plain cosine-weighted sum."""
    arr = Array(35, 0, 60.0, 1.0, 10)
    s = Settings(sky="medium", n_chi=10, sun_model="buie")
    res = simulate(arr, WEATHER, albedo=0.0, settings=s)
    g = make_sky_grid("medium")
    L = sky_radiance(g, WEATHER["dni"], WEATHER["ghi"], np.radians(WEATHER["solar_azimuth"]),
                     np.radians(WEATHER["solar_elevation"])).L
    n = surface_normal(np.radians(35), np.radians(10))
    ref = (g.solid_angle * np.maximum(g.vec @ n, 0)) @ L
    # remaining differences: masking by far rows (~1 deg) and table interpolation
    np.testing.assert_allclose(res.front_direct.mean(axis=1), ref, rtol=0.01)


def test_point_and_buie_agree():
    arr = Array(60, 20, 1.125, 1.84, 0)
    kw = dict(albedo=0.79, far_ground=FarGround(20, 160), albedo_fg=0.3)
    a = simulate(arr, WEATHER, settings=Settings(sun_model="buie"), **kw)
    b = simulate(arr, WEATHER, settings=Settings(sun_model="point", sky="low"), **kw)
    c = simulate(arr, WEATHER, settings=Settings(sun_model="point", sky="medium"), **kw)
    # Buie spreads ~CSR of the beam over the 6 deg aureole, which matters when the sun
    # is close to the shading angle (time step 3); otherwise the two agree closely.
    np.testing.assert_allclose(a.front_mean, b.front_mean, rtol=0.03)
    np.testing.assert_allclose(a.back_mean, b.back_mean, rtol=0.03)
    # with a point sun, a coarse sky grid suffices
    np.testing.assert_allclose(b.front, c.front, rtol=2e-3, atol=0.1)
    np.testing.assert_allclose(b.back, c.back, rtol=2e-3, atol=0.1)


def test_table_resolution_convergence():
    arr = Array.from_shading_angle(60, 20, 15, 0.4)
    a = simulate(arr, WEATHER, albedo=0.79, settings=Settings(n_alpha=720, sky="low"))
    b = simulate(arr, WEATHER, albedo=0.79, settings=Settings(n_alpha=5760, sky="low"))
    # residual differences stem from the beam sitting in single sky cells near shading edges
    np.testing.assert_allclose(a.front, b.front, rtol=0.01, atol=2.0)
    np.testing.assert_allclose(a.back, b.back, rtol=0.01, atol=2.0)
    np.testing.assert_allclose(a.front_mean, b.front_mean, rtol=1e-3)
    np.testing.assert_allclose(a.back_mean, b.back_mean, rtol=1e-3)


def test_legacy_vs_new_close():
    """The corrected model stays close to the MATLAB one for the example configuration."""
    arr = Array(60, 20, 1.125, 1.84, 0)
    kw = dict(albedo=0.79)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        old = simulate(arr, WEATHER, settings=Settings.matlab(sky="low", n_chi=40), **kw)
    new = simulate(arr, WEATHER, settings=Settings(sky="low", n_chi=40), **kw)
    np.testing.assert_allclose(new.front_mean, old.front_mean, rtol=0.03)
    np.testing.assert_allclose(new.back_mean, old.back_mean, rtol=0.05)


def test_multiple_arrays_and_zero_sun():
    arrs = [Array(60, 20, 1.125, 1.84), Array(90, 10, 1.5, 1.2)]
    w = dict(WEATHER)
    w["solar_elevation"] = [11.0, -3.0, 30.0, 20.0]
    res = simulate(arrs, w, albedo=[0.8, 0.8, 0.3, 0.3])
    assert len(res) == 2
    assert (res[0].front[1] == 0).all()
    single = simulate(arrs[1], w, albedo=[0.8, 0.8, 0.3, 0.3])
    np.testing.assert_allclose(single.back, res[1].back)
    df = res[0].summary()
    assert {"front_mean", "back_min", "ghi_check"} <= set(df.columns)


def test_finite_rows():
    """Finite array: only the ground components change (more lit ground) on the sides facing an
    adjacent row, the outward-facing sides are unshaded, and legacy mode refuses the option."""
    arr = Array(90, 20, 1.2, 1.9)
    kw = dict(albedo=0.8, settings=Settings(sky="low", n_chi=40))
    inf = simulate(arr, WEATHER, **kw)
    last = simulate(arr, WEATHER, albedo=0.8, settings=Settings(sky="low", n_chi=40, finite_rows=(2, 0)))
    first = simulate(arr, WEATHER, albedo=0.8, settings=Settings(sky="low", n_chi=40, finite_rows=(0, 2)))
    assert np.isfinite(last.back).all() and np.isfinite(first.front).all()
    assert (first.front_direct >= inf.front_direct - 1e-9).all() and first.front.sum() > inf.front.sum()
    assert (last.back_direct >= inf.back_direct - 1e-9).all() and last.back.sum() > inf.back.sum()
    for c in ("front_direct", "front_farground"):
        np.testing.assert_allclose(getattr(last, c), getattr(inf, c))
    np.testing.assert_allclose(first.back_direct, inf.back_direct)
    assert (last.front_nearground >= inf.front_nearground - 1e-9).all()
    assert (first.back_nearground >= inf.back_nearground - 1e-9).all()
    assert first.back_nearground.sum() > inf.back_nearground.sum()
    with pytest.raises(ValueError):
        simulate(arr, WEATHER, albedo=0.8, settings=Settings.matlab(sky="low", finite_rows=(1, 1)))


@pytest.mark.parametrize("beta_g, thickness", [(0, 0.0), (20, 0.0), (20, 0.1)])
def test_single_row_is_isolated_panel(beta_g, thickness):
    """A single row (both sides outward-facing) equals an infinite array whose rows are so far
    apart that the neighbours only occupy the horizon: sky, beam, ground and far ground."""
    s = dict(sky="low", n_chi=20)
    kw = dict(albedo=0.6, far_ground=FarGround(beta_g, 0), albedo_fg=0.6)
    single = simulate(Array(70, beta_g, 1.5, 1.2, thickness=thickness), WEATHER,
                      settings=Settings(**s, finite_rows=(0, 0)), **kw)
    sparse = simulate(Array(70, beta_g, 2000.0, 1.2, thickness=thickness), WEATHER,
                      settings=Settings(**s, ground_extent=4000.0), **kw)
    for side in ("front", "back"):
        np.testing.assert_allclose(getattr(single, side).mean(axis=1), getattr(sparse, side).mean(axis=1), rtol=5e-3)


def test_thickness():
    """Finite thickness: continuous towards thin panels, lowers the ground-reflected back-side
    irradiance of a dense array (wider shadows), and is refused in legacy mode."""
    kw = dict(albedo=0.8, settings=Settings(sky="low", n_chi=40))
    thin = simulate(Array(90, 20, 1.2, 1.9), WEATHER, **kw)
    eps = simulate(Array(90, 20, 1.2, 1.9, thickness=1e-7), WEATHER, **kw)
    thick = simulate(Array(90, 20, 1.2, 1.9, thickness=0.16), WEATHER, **kw)
    for c in ("front", "back"):
        np.testing.assert_allclose(getattr(eps, c), getattr(thin, c), rtol=1e-4, atol=1e-4)
    assert thick.back_nearground.sum() < 0.97 * thin.back_nearground.sum()
    with pytest.raises(ValueError):
        simulate(Array(90, 20, 1.2, 1.9, thickness=0.1), WEATHER, albedo=0.8,
                 settings=Settings.matlab(sky="low", n_chi=10, n_alpha=60, n_d=10, n_zeta=10, n_phi=10, n_sig=10))
    with pytest.raises(ValueError):
        Array(60, 20, 1.2, 1.9, thickness=-0.1)


def test_date_dependent_extraterrestrial_irradiance():
    """The day of the year is taken from a DatetimeIndex or a day_of_year column."""
    arr = Array(60, 20, 1.125, 1.84)
    s = Settings(sky="low", n_chi=20)
    base = pd.DataFrame(WEATHER)
    jan = base.set_index(pd.date_range("2025-01-03 10:00", periods=4, freq="h"))
    jul = base.set_index(pd.date_range("2025-07-04 10:00", periods=4, freq="h"))
    r_jan = simulate(arr, jan, albedo=0.79, settings=s)
    r_jul = simulate(arr, jul, albedo=0.79, settings=s)
    r_col = simulate(arr, base.assign(day_of_year=3), albedo=0.79, settings=s)
    r_none = simulate(arr, base, albedo=0.79, settings=s)
    np.testing.assert_allclose(r_jan.front, r_col.front)
    assert not np.allclose(r_jan.front, r_jul.front)
    # without dates the solar constant is used: lies between perihelion and aphelion
    lo = np.minimum(r_jan.front_mean.to_numpy(), r_jul.front_mean.to_numpy())
    hi = np.maximum(r_jan.front_mean.to_numpy(), r_jul.front_mean.to_numpy())
    assert np.all((r_none.front_mean.to_numpy() >= lo - 1e-9) & (r_none.front_mean.to_numpy() <= hi + 1e-9))
    # the effect is small (sky distribution only, irradiation totals unchanged)
    np.testing.assert_allclose(r_jan.front_mean.to_numpy(), r_jul.front_mean.to_numpy(), rtol=0.02)


def test_prepare_weather_variants():
    base = prepare_weather(WEATHER)
    sin_el = np.sin(np.radians(base["solar_elevation"]))
    dhi = base["ghi"] - base["dni"] * sin_el
    w2 = prepare_weather(dict(DNI=base["dni"], DHI=dhi, solar_azimuth=base["solar_azimuth"],
                              solar_elevation=base["solar_elevation"]))
    np.testing.assert_allclose(w2["ghi"], base["ghi"])
    with pytest.raises(ValueError):
        prepare_weather(dict(ghi=[1.0]))


def test_solar_position_convention():
    pytest.importorskip("pvlib")
    times = pd.date_range("2025-06-21 00:00", periods=24 * 12, freq="5min", tz="UTC")
    w = prepare_weather(pd.DataFrame({"dni": 500.0, "ghi": 600.0}, index=times), latitude=46.84, longitude=9.81)
    noon = w["solar_elevation"].idxmax()
    assert abs(w.loc[noon, "solar_azimuth"]) < 3         # south at solar noon
    assert w["solar_azimuth"].iloc[6 * 12] < 0            # morning sun in the east (negative)
