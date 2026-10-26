"""Legacy mode reproduces the MATLAB implementation (reference data from tests/matlab)."""

import warnings

import numpy as np
import pytest
from scipy.io import loadmat

from slopevf import Array, FarGround, Settings, simulate
from slopevf.geometry import Geometry
from slopevf.sky import make_legacy_sky_grid, sky_radiance
from slopevf.tables import build_tables_legacy

from conftest import DATA

pytestmark = pytest.mark.skipif(not (DATA / "ref_sky.mat").exists(), reason="no MATLAB reference data")


def _load(name):
    return loadmat(DATA / name, squeeze_me=True, struct_as_record=False)


@pytest.fixture(scope="module")
def ref_sky():
    return _load("ref_sky.mat")


def _weather(irr):
    sp = irr.sunposition
    return dict(dni=np.asarray(irr.DNI, float), ghi=np.asarray(irr.GHI, float),
                solar_azimuth=np.asarray(sp.azimuth_deg, float),
                solar_elevation=np.asarray(sp.elevation_deg, float))


def _settings(so, **kw):
    return Settings.matlab(n_chi=so.n_chi, n_alpha=so.n_sigma, sky=so.hemispherePrecision, n_d=so.n_d,
                           n_zeta=so.n_zeta, n_phi=so.n_phi, n_sig=so.n_sig, max_rows=so.maxRowsConsidered, **kw)


def test_sky_grid(ref_sky):
    g = make_legacy_sky_grid(ref_sky["simopts"].hemispherePrecision)
    assert g.size == ref_sky["sky_area"].size
    np.testing.assert_allclose(g.solid_angle, ref_sky["sky_area"], rtol=1e-12)
    np.testing.assert_allclose(g.azimuth_deg, ref_sky["sky_azim"], atol=1e-10)
    np.testing.assert_allclose(g.elevation_deg, ref_sky["sky_elev"], atol=1e-12)


def test_sky_radiance(ref_sky):
    g = make_legacy_sky_grid(ref_sky["simopts"].hemispherePrecision)
    w = _weather(ref_sky["irradiation"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        L = sky_radiance(g, w["dni"], w["ghi"], np.radians(w["solar_azimuth"]),
                         np.radians(w["solar_elevation"])).L
    np.testing.assert_allclose(np.real(L), ref_sky["radiationMaps"].T, rtol=1e-9)


@pytest.mark.parametrize("k", [1, 2, 3])
def test_tables_and_results(ref_sky, k):
    ref = _load(f"ref_geom{k}.mat")
    c, so = ref["conf"], ref_sky["simopts"]
    arr = Array(c.mu_deg, c.gamma_deg, c.rho_bar, c.h_bar, c.alpha_g_deg)
    T = build_tables_legacy(Geometry(arr), so.n_sigma, so.n_chi, so.n_d, so.n_zeta,
                            so.maxRowsConsidered, so.n_phi, so.n_sig)
    MT = ref["MT"]
    for ours, theirs in (("front_direct", "front_GPF_mult"), ("back_direct", "back_GPB_mult"),
                         ("front_ground", "front_GG_mult"), ("back_ground", "back_GG_mult"),
                         ("far_ground", "front_GFG_mult")):
        np.testing.assert_allclose(getattr(T, ours), getattr(MT, theirs), atol=1e-12, err_msg=ours)

    irr = ref_sky["irradiation"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = simulate(arr, _weather(irr), albedo=irr.omega, far_ground=FarGround(c.gamma_fg_deg, c.alpha_fg_deg),
                       albedo_fg=irr.omega_fg, settings=_settings(so))
    for ours, theirs in (("front_direct", "fd"), ("back_direct", "bd"), ("front_nearground", "fng"),
                         ("back_nearground", "bng"), ("front_farground", "ffg")):
        np.testing.assert_allclose(getattr(res, ours), np.real(ref[theirs]), rtol=1e-9, atol=1e-9, err_msg=ours)
    np.testing.assert_allclose(res.ghi_check, ref["globalCheck"], rtol=1e-12)


def test_full_precision_example():
    """example_script.m settings (n_chi=100, n_sigma=180, 'high' sky, n_d=n_zeta=50)."""
    ref = _load("ref_full.mat")
    f = ref["full"]
    arr = Array(60, 20, 1.125, 1.84, 0)
    weather = dict(dni=[550.0], ghi=[200.0], solar_azimuth=[-40.0], solar_elevation=[11.0])
    res = simulate(arr, weather, albedo=0.79, far_ground=FarGround(20, 160), albedo_fg=0.3,
                   settings=_settings(f))
    for ours, theirs in (("front", "front"), ("back", "back"), ("front_farground", "ffg"),
                         ("back_nearground", "bng")):
        np.testing.assert_allclose(getattr(res, ours)[0], ref[theirs], rtol=1e-9, atol=1e-9, err_msg=ours)
