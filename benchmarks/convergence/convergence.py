"""Numerical convergence of slopevf with respect to its discretisation parameters.

One parameter at a time is varied from the defaults (Settings()) and compared with a fine
reference for the same parameter, for a full TMY year (PVGIS, Davos) and four arrays:

  sky            sky grid resolution ('low' ... 'ultra'), for both sun models
  n_alpha        profile-angle intervals of the multiplication table
  n_chi          panel elements (only panel-level metrics are comparable)
  ground_extent  range of exact summation of the irradiated ground strips [panel heights]

Metrics (per array, worst case over the arrays is reported):
  year_mean      relative error of the yearly panel-average irradiation (front, back)
  year_min       relative error of the yearly sum of the hourly panel minimum (mismatch-relevant)
  hour_mean      max. relative error of the hourly panel average (hours with > 50 W/m2)
  hour_profile   max. absolute error of hourly element values [W/m2] (same n_chi only)

Usage:  python convergence.py        (results in results/)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from slopevf import Array, FarGround, Settings, prepare_weather, simulate

HERE = Path(__file__).parent
WORK = HERE / "work"
RESULTS = HERE / "results"
SITE = dict(latitude=46.838, longitude=9.813, altitude=1560)

ARRAYS = {
    "p60_g20": Array.from_shading_angle(60, 20, 15, 1.0),
    "p90_g30": Array.from_shading_angle(90, 30, 15, 1.2),
    "p30_g0": Array.from_shading_angle(30, 0, 20, 1.0),
    "p60_g20_low": Array.from_shading_angle(60, 20, 15, 0.4),
}
DEFAULT = Settings()

STUDIES = {
    # name: (list of variations, reference)
    "sky_point": ([{"sky": s} for s in ("low", "medium", "high")], {"sky": "ultra"}),
    "sky_buie": ([{"sky": s, "sun_model": "buie"} for s in ("low", "medium", "high")],
                 {"sky": "ultra", "sun_model": "buie"}),
    "n_alpha": ([{"n_alpha": n} for n in (180, 360, 720, 1440)], {"n_alpha": 5760}),
    "n_chi": ([{"n_chi": n} for n in (25, 50, 100, 200)], {"n_chi": 800}),
    "ground_extent": ([{"ground_extent": e} for e in (10.0, 25.0, 50.0, 100.0)], {"ground_extent": 1000.0}),
}


def weather():
    cache = WORK / "pvgis_tmy_davos.csv"
    if cache.exists():
        tmy = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        import pvlib
        tmy = pvlib.iotools.get_pvgis_tmy(SITE["latitude"], SITE["longitude"], map_variables=True)[0]
        WORK.mkdir(parents=True, exist_ok=True)
        tmy.to_csv(cache)
    w = prepare_weather(tmy[["dni", "ghi"]], **SITE)
    winter = w.index.month.isin([10, 11, 12, 1, 2, 3])
    w["albedo"] = np.where(winter, 0.7, 0.3)
    w["albedo_fg"] = np.where(winter, 0.3, 0.2)
    return w


def run(w, overrides):
    s = replace(DEFAULT, **overrides)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = simulate(list(ARRAYS.values()), w, far_ground=FarGround(20, 160), settings=s)
    return dict(zip(ARRAYS, res)), time.perf_counter() - t0


def metrics(res, ref, same_chi):
    out = {}
    for side in ("front", "back"):
        e_mean, e_min, e_hour, e_prof = [], [], [], []
        for name in ARRAYS:
            a, b = getattr(res[name], side), getattr(ref[name], side)
            ma, mb = a.mean(axis=1), b.mean(axis=1)
            e_mean.append(abs(ma.sum() / mb.sum() - 1))
            e_min.append(abs(a.min(axis=1).sum() / b.min(axis=1).sum() - 1))
            ok = mb > 50
            e_hour.append(np.max(np.abs(ma[ok] / mb[ok] - 1)) if ok.any() else 0.0)
            if same_chi:
                e_prof.append(np.max(np.abs(a - b)))
        out[f"{side}_year_mean_%"] = 100 * max(e_mean)
        out[f"{side}_year_min_%"] = 100 * max(e_min)
        out[f"{side}_hour_mean_%"] = 100 * max(e_hour)
        out[f"{side}_hour_profile_W"] = max(e_prof) if same_chi else np.nan
    return out


def main():
    w = weather()
    rows = []
    for study, (variations, reference) in STUDIES.items():
        ref, t_ref = run(w, reference)
        for ov in variations + [reference]:
            res, t = (ref, t_ref) if ov is reference else run(w, ov)
            same_chi = res[next(iter(ARRAYS))].chi.size == ref[next(iter(ARRAYS))].chi.size
            value = {k: v for k, v in ov.items() if k != "sun_model"}
            row = dict(study=study, value=str(next(iter(value.values()))),
                       reference=(ov is reference), seconds=t, **metrics(res, ref, same_chi))
            rows.append(row)
            print(f"{study:14s} {row['value']:>8s}{' (ref)' if row['reference'] else '      '} "
                  f"year F {row['front_year_mean_%']:.3f}% B {row['back_year_mean_%']:.3f}%  "
                  f"min F {row['front_year_min_%']:.3f}% B {row['back_year_min_%']:.3f}%  "
                  f"hour F {row['front_hour_mean_%']:.2f}% B {row['back_hour_mean_%']:.2f}%  "
                  f"[{t:5.1f} s]", flush=True)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "convergence.csv", index=False)
    return df


if __name__ == "__main__":
    main()
