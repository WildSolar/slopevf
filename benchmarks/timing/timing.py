"""Execution times of the main steps of slopevf (manuscript table "executiontimes").

Steps, as executed by ``slopevf.simulate``:

  1. build the sky discretisation (once)
  2. build the multiplication tables and sky-to-profile-angle mappings (per configuration)
  3. sky radiance for all daylight hours of the year (once, shared by all configurations)
  4. apply the tables (per configuration)

plus end-to-end ``simulate`` for 1, 10 and 100 configurations, for the sky resolutions 'medium'
(default) and 'low'. Each measurement is repeated and the fastest run is reported, since single
timings on a laptop fluctuate by up to +-30 %.

Weather: PVGIS TMY for Davos (the paper's site), downloaded with pvlib on first use and cached
in results/. Usage:  python timing.py [--repeat 3]   (writes results/timing.txt and timing.json)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import slopevf
from slopevf import Array, FarGround, Settings, simulate
from slopevf.model import ArrayModel, prepare_weather
from slopevf.sky import make_sky_grid, sky_radiance

HERE = Path(__file__).parent
RESULTS = HERE / "results"
LAT, LON, ALT = 46.838, 9.813, 2500.0
FAR_GROUND = FarGround(20, 160)
BASE = Array(60, 20, 1.125, 1.84)                       # example configuration of the first paper
CONFIGS = [Array(60, 20, rho, 1.84) for rho in np.linspace(1.0, 3.0, 10)]
CONFIGS_100 = [Array(60, 20, rho, 1.84) for rho in np.linspace(1.0, 3.0, 100)]   # ~6 GB of results


def weather():
    cache = RESULTS / "pvgis_tmy_davos.csv"
    if not cache.exists():
        import pvlib
        RESULTS.mkdir(exist_ok=True)
        pvlib.iotools.get_pvgis_tmy(LAT, LON, map_variables=True)[0].to_csv(cache)
    w = pd.read_csv(cache, index_col=0, parse_dates=True)[["dni", "ghi"]]
    return prepare_weather(w, LAT, LON, ALT)


def best(fn, repeat):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return min(times), out


def steps(df, sky_res, repeat):
    s = Settings(sky=sky_res)
    dni, ghi = df["dni"].to_numpy(float), df["ghi"].to_numpy(float)
    az, el = np.radians(df["solar_azimuth"].to_numpy(float)), np.radians(df["solar_elevation"].to_numpy(float))
    doy = df.index.dayofyear.to_numpy()
    valid = np.flatnonzero((el > 0) & (ghi > 0))
    chunks = [valid[i:i + s.chunk_size] for i in range(0, valid.size, s.chunk_size)]

    t_grid, grid = best(lambda: make_sky_grid(sky_res), repeat)
    t_tables, model = best(lambda: ArrayModel(BASE, FAR_GROUND, grid, s), repeat)

    def sky_all():
        return [sky_radiance(grid, dni[i], ghi[i], az[i], el[i], sun_model=s.sun_model,
                             day_of_year=doy[i], solar_constant=s.solar_constant).L for i in chunks]

    t_sky, skies = best(sky_all, repeat)
    alb = np.full(dni.size, 0.5)

    def apply_all():
        for i, L in zip(chunks, skies):
            model.apply(L, alb[i], alb[i], (dni[i], el[i], az[i]))

    t_apply, _ = best(apply_all, repeat)
    return dict(n_cells=grid.size, n_hours=int(valid.size), build_grid=t_grid, build_tables=t_tables,
                sky_radiance=t_sky, apply_tables=t_apply)


def end_to_end(df, sky_res, n_conf, repeat):
    kw = dict(albedo=0.5, far_ground=FAR_GROUND, albedo_fg=0.5, settings=Settings(sky=sky_res))
    configs = CONFIGS_100 if n_conf == 100 else CONFIGS[:n_conf]
    t, _ = best(lambda: simulate(configs, df, **kw), repeat)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    df = weather()
    simulate(BASE, df.iloc[:200], albedo=0.5, settings=Settings(sky="low"))     # warm-up (imports, caches)
    res = {"machine": f"{platform.processor()} ({os.cpu_count()} logical cores), {platform.system()} "
                      f"{platform.release()}, Python {platform.python_version()}, numpy {np.__version__}",
           "slopevf": getattr(slopevf, "__version__", "?"), "repeat": a.repeat}
    for sky_res in ("medium", "low"):
        r = steps(df, sky_res, a.repeat)
        r["simulate_1"] = end_to_end(df, sky_res, 1, a.repeat)
        r["simulate_10"] = end_to_end(df, sky_res, 10, a.repeat)
        r["simulate_100"] = end_to_end(df, sky_res, 100, a.repeat)
        res[sky_res] = r
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "timing.json").write_text(json.dumps(res, indent=2))
    lines = [f"# slopevf execution times (fastest of {a.repeat} runs)", res["machine"], ""]
    for sky_res in ("medium", "low"):
        r = res[sky_res]
        lines += [f"## sky '{sky_res}': {r['n_cells']} cells, {r['n_hours']} daylight hours, "
                  "721 profile angles, 100 panel elements",
                  f"build sky discretisation (once)            {r['build_grid']:8.3f} s",
                  f"build tables and mappings (per config)     {r['build_tables']:8.3f} s",
                  f"sky radiance, full year (once)             {r['sky_radiance']:8.3f} s",
                  f"apply tables, full year (per config)       {r['apply_tables']:8.3f} s",
                  f"simulate(), 1 configuration                {r['simulate_1']:8.3f} s",
                  f"simulate(), 10 configurations              {r['simulate_10']:8.3f} s",
                  f"simulate(), 100 configurations             {r['simulate_100']:8.3f} s", ""]
    text = "\n".join(lines)
    (RESULTS / "timing.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
