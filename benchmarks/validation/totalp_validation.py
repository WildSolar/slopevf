"""Validation against the adjustable miniature PV array at the Totalp site (winter 2024/25).

Measurement data and setup: data/totalp/ (see the README there). Columns: Timestamp; channel
(22: backside, 3: frontside); RA (in-plane row spacing, mm); MN_relHor (panel tilt, deg); HN
(ground slope, deg); Pnom (STC-normalised power); elevation, azimuth (sun, deg, 0 = south,
positive towards west); DNI, GHI (W/m^2, measured locally, i.e. including the local horizon).

Geometry: panel height 96 mm; horizontal row spacing rho = RA cos(HN) / 96; centre height
h_bar = 1.77 / cos(HN) (1.77 panel heights perpendicular to the sloped ground). Three rows; the
frontside module is on the last (uphill) row, i.e. two rows in front of it, the backside
module on the first (downhill) row, i.e. two rows behind it. The modules average over the full
row height (panel-mean irradiance). Albedo 0.79, far ground 0.5 with orientation 20 / 160 deg
(first paper).

Compared (prediction = panel-average irradiance / 1000 W/m^2 vs. Pnom):
  legacy       MATLAB implementation of the first paper (Settings.matlab()), infinite rows
  corrected    current model, infinite rows, non-reflective panels
  glass        corrected with Fresnel glass (n = 1.52, uncoated, both sides)
  array        current model for the actual three-row array: finite_rows = (2, 0) for the
               frontside module, (0, 2) for the backside module
  array_glass  array with Fresnel glass
  array_glass_horizon  array_glass with the local horizon (Totalp_Horizon.hor, terrain "facing",
               snow albedo); the measured DNI/GHI are used as inputs (local DNI = free DNI while
               the sun is visible)
  array_glass_tau  array_glass with the transmission of the module's own glass (incidence angle
               modifier tau(theta) / tau(0), Fresnel n = 1.52); Pnom is normalised to STC, i.e.
               normal incidence
  full         array_glass_tau with the local horizon
  full_thick   full with the module stack of 15 mm (Array(thickness=15/96))
  full_thick_a085  full_thick with albedo 0.85 (painted plate: 0.798 broadband per ASTM E903,
               about 0.85 weighted with the silicon response)
  plate        full_thick for the actual ground: the finite rows on the finite base plate, with
               snow-free (albedo 0.2) or snow-covered (0.85) surroundings beyond it, interpolated
               with the daily snow cover (data/totalp/snow_cover.csv). Ray-traced correction factors
               per configuration from benchmarks/raytrace/plate_check.py (run it first); the albedo
               of the horizon terrain follows the snow cover as well (the model is linear in it).
  plate_t10    plate with a 10 mm module stack instead of 15 mm (sensitivity)
  plate_dXX    plate with diffuse reflection of the module surfaces, reflectance 0.XX
               (PanelGlass(diffuse=...)); the diffuse component is linear in the reflectance and
               taken from one run with 0.1 (horizon terrain albedo 0.79, without the plate factor)
  plate_dfit   plate with the diffuse reflectance fitted to all measurements (least squares,
               front and back); the fitted value is printed

The statistics are given for the whole period and separately for the snow-free (snow cover 0) and
snow-covered (snow cover 1) days.

Not modelled: frames and gaps, support posts, cell temperature and spectral effects.

Usage:  python totalp_validation.py      (results in results/)
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from slopevf import Array, FarGround, Horizon, PanelGlass, Settings, simulate

HERE = Path(__file__).parent
DATA = HERE.parents[1] / "data" / "totalp" / "Totalp_Measurement_Data.csv"
RESULTS = HERE / "results"
H_MM = 96.0
# Sensitivity runs (environment variables; results get the suffix _<TOTALP_TAG>):
#   TOTALP_ALBEDO  albedo of the base plate and the uniform ground (default 0.79, broadband; 0.85 is the
#                  silicon-weighted value); needs raytrace/plate_check.py run with PLATE_ALBEDO = same
#   TOTALP_SUN=1   recompute the sun position at the true time (the measurement timestamps are the database
#                  clock with a drift of 4 - 5 min, see data/totalp/README.md)
#   TOTALP_DHI=1   scale the DHI input such that the model's horizontal irradiation including the local
#                  horizon equals the measured GHI (the measured GHI already contains the horizon, the
#                  model masks the free-horizon sky once more)
TAG = os.environ.get("TOTALP_TAG", "")
SUFFIX = f"_{TAG}" if TAG else ""
SUN_CORRECTION = os.environ.get("TOTALP_SUN") == "1"
DHI_CORRECTION = os.environ.get("TOTALP_DHI") == "1"
SITE = (46.84, 9.80, 2500.0)   # approximate latitude, longitude, altitude of the Totalp site
ALBEDO, ALBEDO_FG, FAR_GROUND = float(os.environ.get("TOTALP_ALBEDO", 0.79)), 0.5, FarGround(20, 160)
MIN_MEASUREMENTS = 20          # configurations with fewer measurements are excluded from the statistics
ROWS = {"front": (2, 0), "back": (0, 2)}
GLASS = PanelGlass(n=1.52)
GLASS_TAU = PanelGlass(n=1.52, transmission=True)
GLASS_DIFFUSE = PanelGlass(n=1.52, transmission=True, diffuse=0.1)
DIFFUSE_SCAN = (0.03, 0.05, 0.08)
THICKNESS = 15.0 / H_MM        # module stack: two modules back to back + 25 x 7 mm rod lying flat
THICKNESS_THIN = 10.0 / H_MM   # sensitivity: thinner modules (datasheet: 2 mm each)
HORIZON = Horizon.from_file(HERE.parents[1] / "data" / "totalp" / "Totalp_Horizon.hor")   # "facing" terrain
SNOW = HERE.parents[1] / "data" / "totalp" / "snow_cover.csv"
PLATE = HERE.parent / "raytrace" / "results" / (
    "plate_check.csv" if ALBEDO == 0.79 else f"plate_check_a{round(100 * ALBEDO):03d}.csv")
ALBEDO_FREE, ALBEDO_SNOW = 0.2, 0.85     # surroundings, as in plate_check.py


def _variants(side):
    """Simulation keyword arguments per variant for the module on the given side."""
    base = dict(sky="low", n_chi=50)
    return {
        "legacy": dict(settings=Settings.matlab(sky="medium", n_chi=50, n_d=100, n_zeta=100)),
        "corrected": dict(settings=Settings(**base)),
        "glass": dict(settings=Settings(**base), panel_glass=GLASS),
        "array": dict(settings=Settings(**base, finite_rows=ROWS[side])),
        "array_glass": dict(settings=Settings(**base, finite_rows=ROWS[side]), panel_glass=GLASS),
        "array_glass_horizon": dict(settings=Settings(**base, finite_rows=ROWS[side]), panel_glass=GLASS,
                                    horizon=HORIZON, albedo_terrain=ALBEDO),
        "array_glass_tau": dict(settings=Settings(**base, finite_rows=ROWS[side]), panel_glass=GLASS_TAU),
        "full": dict(settings=Settings(**base, finite_rows=ROWS[side]), panel_glass=GLASS_TAU,
                     horizon=HORIZON, albedo_terrain=ALBEDO),
        "full_thick": dict(settings=Settings(**base, finite_rows=ROWS[side]), panel_glass=GLASS_TAU,
                           horizon=HORIZON, albedo_terrain=ALBEDO, thickness=THICKNESS),
        "full_thick_a085": dict(settings=Settings(**base, finite_rows=ROWS[side]), panel_glass=GLASS_TAU,
                                horizon=HORIZON, albedo_terrain=ALBEDO, thickness=THICKNESS, albedo=0.85),
        # building blocks of plate / plate_t10 (terrain albedo 0 and 1)
        "_t15_terr0": dict(settings=Settings(**base, finite_rows=ROWS[side]), panel_glass=GLASS_TAU,
                           horizon=HORIZON, albedo_terrain=0.0, thickness=THICKNESS),
        "_t15_terr1": dict(settings=Settings(**base, finite_rows=ROWS[side]), panel_glass=GLASS_TAU,
                           horizon=HORIZON, albedo_terrain=1.0, thickness=THICKNESS),
        "_t10_terr0": dict(settings=Settings(**base, finite_rows=ROWS[side]), panel_glass=GLASS_TAU,
                           horizon=HORIZON, albedo_terrain=0.0, thickness=THICKNESS_THIN),
        "_t10_terr1": dict(settings=Settings(**base, finite_rows=ROWS[side]), panel_glass=GLASS_TAU,
                           horizon=HORIZON, albedo_terrain=1.0, thickness=THICKNESS_THIN),
        "_t15_d10": dict(settings=Settings(**base, finite_rows=ROWS[side]), panel_glass=GLASS_DIFFUSE,
                         horizon=HORIZON, albedo_terrain=ALBEDO, thickness=THICKNESS),
    }


SIMULATED = list(_variants("front"))
DERIVED = ["plate", "plate_t10"] + [f"plate_d{100 * r:02.0f}" for r in DIFFUSE_SCAN] + ["plate_dfit"]
VARIANTS = [v for v in SIMULATED if not v.startswith("_")] + DERIVED


def load():
    df = pd.read_csv(DATA, sep=";", decimal=",")
    df.columns = [c.strip() for c in df.columns]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["side"] = np.where(df["channel"] == 22, "back", "front")
    df["rho_bar"] = df["RA"] * np.cos(np.radians(df["HN"])) / H_MM
    df["h_bar"] = 1.77 / np.cos(np.radians(df["HN"]))
    valid = (df["MN_relHor"] > df["HN"]) & (df["MN_relHor"] <= 90)
    df = df[valid].reset_index(drop=True)
    df["ghi_in"] = df["GHI"]
    if SUN_CORRECTION:
        _correct_sun(df)
    if DHI_CORRECTION:
        _correct_dhi(df)
    return df


def _correct_sun(df):
    """Sun position at the true time. The timestamps are the database clock: UTC + 124 min until the end
    of DST on 27 Oct 2024, UTC + 64.5 min (65 min in March) in winter time, UTC + 125 min after 30 Mar
    2025 (civil time plus a clock drift of 4 - 5 min, determined from the GHI of the full-size station)."""
    import pvlib
    ts = df["Timestamp"]
    shift = np.where(ts < "2024-10-27 03:00", 124.0, np.where(ts < "2025-03-30 03:00", 64.5, 125.0))
    shift = np.where((ts >= "2025-03-01") & (ts < "2025-03-30 03:00"), 65.0, shift)
    utc = (ts - pd.to_timedelta(shift, unit="min")).dt.tz_localize("UTC")
    sp = pvlib.solarposition.get_solarposition(utc, SITE[0], SITE[1], altitude=SITE[2])
    d_az = (sp.azimuth.to_numpy() - 180.0) - df["azimuth"].to_numpy()
    d_el = sp.apparent_elevation.to_numpy() - df["elevation"].to_numpy()
    df.attrs["sun_shift"] = (float(np.median(d_az)), float(np.percentile(d_el, 5)), float(np.percentile(d_el, 95)))
    df["azimuth"] = sp.azimuth.to_numpy() - 180.0
    df["elevation"] = sp.apparent_elevation.to_numpy()


def _snow_albedo(df):
    snow = pd.read_csv(SNOW, index_col="date")["snow_cover"]
    cover = df.Timestamp.dt.strftime("%Y-%m-%d").map(snow).to_numpy()
    return ALBEDO_FREE + (ALBEDO_SNOW - ALBEDO_FREE) * cover


def _correct_dhi(df, n_iter=4):
    """Scale the DHI input such that the model's horizontal irradiation including the horizon equals the
    measured GHI. The sky of the model is normalised to the DHI it is given and then masked by the horizon;
    the measured GHI, however, already lacks the sky below the horizon (and contains the terrain reflection),
    so the horizon would be counted twice in the diffuse part. Fixed-point iteration per time step."""
    el = np.radians(df["elevation"].to_numpy())
    dni, ghi = df["DNI"].to_numpy(), df["GHI"].to_numpy()
    dhi = np.clip(ghi - dni * np.sin(el), 0.0, None)
    a_t = _snow_albedo(df)
    k = np.ones(len(df))
    probe = Array(60.0, 0.0, 3.0, 2.0)
    for it in range(n_iter):
        w = pd.DataFrame({"dni": dni, "ghi": dhi * k + dni * np.sin(el), "solar_azimuth": df["azimuth"].to_numpy(),
                          "solar_elevation": df["elevation"].to_numpy()}, index=pd.DatetimeIndex(df.Timestamp))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = simulate(probe, w, albedo=ALBEDO, far_ground=FAR_GROUND, albedo_fg=ALBEDO_FG, horizon=HORIZON,
                         albedo_terrain=a_t, settings=Settings(sky="low", n_chi=4))
        if it == 0:                                  # diagnostic: measured values used as free-horizon input
            df.attrs["ghi_model_ratio"] = np.where(ghi > 20.0, r.ghi_check / np.where(ghi > 20.0, ghi, 1.0), np.nan)
        beam = dni * np.sin(el)                     # not scaled (zero while the sun is behind the horizon)
        diff_model, diff_meas = r.ghi_check - beam, ghi - beam
        # the measured diffuse irradiation is a small difference of two large readings under clear sky:
        # hours below 5 W/m2 are left alone, and the factor is limited
        ok = (diff_model > 5.0) & (diff_meas > 5.0)
        k = np.where(ok, np.clip(k * np.clip(diff_meas / np.where(ok, diff_model, 1.0), 0.8, 1.25), 0.5, 1.5), k)
    df["ghi_in"] = dhi * k + dni * np.sin(el)
    df.attrs["dhi_factor"] = k


def predict(df):
    out = {v: np.full(len(df), np.nan) for v in SIMULATED}
    for (bp, bg, ra, side), g in df.groupby(["MN_relHor", "HN", "RA", "side"]):
        arr = Array(float(bp), float(bg), float(g.rho_bar.iloc[0]), float(g.h_bar.iloc[0]))
        w = pd.DataFrame({"dni": g.DNI.to_numpy(), "ghi": g.ghi_in.to_numpy(),
                          "solar_azimuth": g.azimuth.to_numpy(), "solar_elevation": g.elevation.to_numpy()},
                         index=pd.DatetimeIndex(g.Timestamp))
        for name, kw in _variants(side).items():
            kw = dict(kw)
            a = Array(float(bp), float(bg), arr.rho_bar, arr.h_bar, thickness=kw.pop("thickness", 0.0))
            albedo = kw.pop("albedo", ALBEDO)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = simulate(a, w, albedo=albedo, far_ground=FAR_GROUND, albedo_fg=ALBEDO_FG, **kw)
            out[name][g.index] = (r.front_mean if side == "front" else r.back_mean).to_numpy() / 1000.0
        print(f"{side:5s} tilt {bp:3.0f} slope {bg:2.0f} RA {ra:4.0f} (rho {arr.rho_bar:.2f}): "
              f"{len(g)} measurements", flush=True)
    for name in SIMULATED:
        if not name.startswith("_"):
            df[f"pred_{name}"] = out[name]
    _plate(df, out)
    return df


def _plate(df, out):
    """Derived variants plate / plate_t10: snow-dependent albedo of the surroundings and the horizon
    terrain, and the ray-traced factors for the finite plate (interpolated, the irradiance being
    linear in the albedo of the surroundings)."""
    snow = pd.read_csv(SNOW, index_col="date")["snow_cover"]
    df["snow_cover"] = df.Timestamp.dt.strftime("%Y-%m-%d").map(snow).to_numpy()
    a_s = ALBEDO_FREE + (ALBEDO_SNOW - ALBEDO_FREE) * df.snow_cover.to_numpy()
    f = pd.read_csv(PLATE).rename(columns={"beta_p": "MN_relHor", "beta_g": "HN"})
    f = df[["side", "MN_relHor", "HN", "RA"]].merge(f, how="left", on=["side", "MN_relHor", "HN", "RA"])
    factor = (f.plate_free + df.snow_cover.to_numpy() * (f.plate_snow - f.plate_free)).to_numpy()
    for name, t in (("plate", "t15"), ("plate_t10", "t10")):
        t0, t1 = out[f"_{t}_terr0"], out[f"_{t}_terr1"]
        df[f"pred_{name}"] = (t0 + a_s * (t1 - t0)) * factor
    # diffuse reflection of the module surfaces, per unit reflectance
    d1 = (out["_t15_d10"] - out["full_thick"]) / 0.1
    for r in DIFFUSE_SCAN:
        df[f"pred_plate_d{100 * r:02.0f}"] = df.pred_plate + r * d1
    ok = np.isfinite(df.pred_plate) & np.isfinite(d1)
    p, m, dd = df.pred_plate[ok].to_numpy(), df.Pnom[ok].to_numpy(), d1[ok]
    r_fit = float(np.sum(dd * (m - p)) / np.sum(dd * dd))
    df["pred_plate_dfit"] = df.pred_plate + r_fit * d1
    df.attrs["r_fit"] = r_fit


def summarise(df):
    rows = []
    for (bp, bg, ra, side), g in df.groupby(["MN_relHor", "HN", "RA", "side"]):
        row = dict(beta_p=bp, beta_g=bg, RA=ra, rho_bar=g.rho_bar.iloc[0], side=side, n=len(g))
        for name in VARIANTS:
            p, m = g[f"pred_{name}"], g["Pnom"]
            row[f"ratio_{name}"] = p.sum() / m.sum()
            row[f"nrmse_{name}_%"] = 100 * np.sqrt(np.mean((p - m) ** 2)) / m.mean()
        rows.append(row)
    s = pd.DataFrame(rows)
    used = s[s.n >= MIN_MEASUREMENTS]
    lines = [f"## Totalp validation ({len(df)} measurements, "
             f"{s[['beta_p', 'beta_g', 'RA']].drop_duplicates().shape[0]} configurations)",
             f"aggregated ratio prediction / measurement (mean, min .. max over configurations with "
             f">= {MIN_MEASUREMENTS} measurements) and NRMSE"]
    for side in ("front", "back"):
        for bp in sorted(used.beta_p.unique()):
            q = used[(used.side == side) & (used.beta_p == bp)]
            if q.empty:
                continue
            parts = [f"{name}: {q[f'ratio_{name}'].mean():.3f} ({q[f'ratio_{name}'].min():.3f}..{q[f'ratio_{name}'].max():.3f}), "
                     f"NRMSE {q[f'nrmse_{name}_%'].mean():.1f}%" for name in VARIANTS]
            lines.append(f"{side:5s} tilt {bp:3.0f} ({len(q):2d} cfg): " + " | ".join(parts))
    excluded = s[s.n < MIN_MEASUREMENTS]
    if len(excluded):
        lines.append("excluded: " + ", ".join(f"{r.side} {r.beta_p:.0f}/{r.beta_g:.0f}/{r.RA:.0f} (n={r.n})"
                                              for r in excluded.itertuples()))
    lines += ["", f"options: tag '{TAG}', plate/ground albedo {ALBEDO}, sun at true time {SUN_CORRECTION}, DHI normalised {DHI_CORRECTION}"]
    if SUN_CORRECTION:
        d = df.attrs["sun_shift"]
        lines.append(f"sun position change: azimuth median {d[0]:+.2f} deg, elevation 5th/95th percentile {d[1]:+.2f} / {d[2]:+.2f} deg")
    if DHI_CORRECTION:
        k = df.attrs["dhi_factor"]
        kd0 = (df.GHI - df.DNI * np.sin(np.radians(df.elevation))) / df.GHI
        lines.append(f"DHI scaling factor (diffuse part): mean {k.mean():.3f}, median {np.median(k):.3f}, 5-95% "
                     f"{np.percentile(k, 5):.3f} - {np.percentile(k, 95):.3f}; clear (diffuse fraction < 0.3) "
                     f"{k[(kd0 < 0.3).to_numpy()].mean():.3f}, overcast (> 0.7) {k[(kd0 > 0.7).to_numpy()].mean():.3f}")
        q = df.attrs["ghi_model_ratio"]
        c, o = (kd0 < 0.3).to_numpy(), (kd0 > 0.7).to_numpy()
        lines.append(f"model GHI incl. horizon / measured GHI with the measured values as free-horizon input: mean "
                     f"{np.nanmean(q):.4f}; clear {np.nanmean(q[c]):.4f}; overcast {np.nanmean(q[o]):.4f}; "
                     f"elevation < 15 deg {np.nanmean(q[(df.elevation < 15).to_numpy()]):.4f}")
    lines += ["", f"fitted diffuse reflectance of the module surfaces (plate_dfit): {df.attrs['r_fit']:.3f}",
              "", "## By snow cover of the surroundings (snow-free: cover 0, snow: cover 1)",
              "ratio prediction / measurement summed over all records (all sky / clear sky, diffuse fraction < 0.3)"]
    kd = (df.GHI - df.DNI * np.sin(np.radians(df.elevation))) / df.GHI
    used_cfg = set(map(tuple, used[["side", "beta_p", "beta_g", "RA"]].to_numpy()))
    ok = np.array([k in used_cfg for k in zip(df.side, df.MN_relHor, df.HN, df.RA)])
    for period, sel in (("snow-free", df.snow_cover == 0), ("snow", df.snow_cover == 1)):
        for side in ("front", "back"):
            for bp in sorted(used.beta_p.unique()):
                g = df[ok & sel & (df.side == side) & (df.MN_relHor == bp)]
                c = g[kd[g.index] < 0.3]
                parts = [f"{name}: {g[f'pred_{name}'].sum() / g.Pnom.sum():.3f} / "
                         f"{c[f'pred_{name}'].sum() / c.Pnom.sum():.3f}"
                         for name in ("full_thick", "plate", "plate_t10", "plate_d05", "plate_dfit")]
                lines.append(f"{period:9s} {side:5s} tilt {bp:3.0f} (n={len(g):4d}): " + " | ".join(parts))
    return s, "\n".join(lines)


def main():
    df = predict(load())
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / f"totalp_predictions{SUFFIX}.csv", index=False)
    s, text = summarise(df)
    s.to_csv(RESULTS / f"totalp_summary{SUFFIX}.csv", index=False)
    (RESULTS / f"totalp_summary{SUFFIX}.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
