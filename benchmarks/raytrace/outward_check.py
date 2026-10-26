"""Outward-facing sides of a finite array against Radiance.

The sides of the outermost rows that face away from the array (the front of the first, downhill
row and the back of the last, uphill row) have no adjacent row: they are not shaded by another
row, see the ground out to infinity and, facing downhill, the terrain below the horizontal.
For three rows (layout of the Totalp miniature array), selected configurations and sampled
measurement time steps, slopevf (finite_rows=(0, 2) for the front of the first row, (2, 0) for
the back of the last row) is compared with Radiance for 600 H rows, thin and with a 15 mm stack,
on uniform ground (albedo 0.79) with the continuing slope as glow below the horizontal.

Usage:  python outward_check.py      (results in results/outward_check.txt)
"""

from __future__ import annotations

import json
import zlib

import numpy as np
import pandas as pd

from finite_array_check import (ALBEDO, DATA, LONG, RESULTS, THICKNESS, WORK, scene, sensor_rays,
                                slope_irradiance)
from radscene import build_octree, irradiance, radiance_env, sky_rad
from slopevf import Array, FarGround, Settings, simulate

N_STEPS = 8
ROWS = {"front": (0, 1, 2), "back": (-2, -1, 0)}            # evaluated row 0 is outermost
FINITE = {"front": (0, 2), "back": (2, 0)}
CONFIGS = [("front", 90, 0, 263), ("front", 90, 20, 115), ("front", 60, 20, 166), ("front", 60, 30, 137),
           ("back", 90, 0, 263), ("back", 90, 20, 115), ("back", 60, 20, 115), ("back", 90, 30, 115)]


def model(arr, side, m, thickness):
    a = Array(arr.beta_p, arr.beta_g, arr.rho_bar, arr.h_bar, thickness=thickness)
    r = simulate(a, dict(dni=[m.DNI], ghi=[m.GHI], solar_azimuth=[m.azimuth], solar_elevation=[m.elevation]),
                 albedo=ALBEDO, far_ground=FarGround(arr.beta_g, 0.0), albedo_fg=ALBEDO,
                 settings=Settings(sky="medium", n_chi=32, finite_rows=FINITE[side]))
    return float((r.front_mean if side == "front" else r.back_mean).iloc[0])


def main():
    env = radiance_env()
    cache_file = WORK / "cache.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    df = pd.read_csv(DATA, sep=";", decimal=",")
    rows = []
    for side, bp, bg, ra in CONFIGS:
        arr = Array(bp, bg, ra * np.cos(np.radians(bg)) / 96.0, 1.77 / np.cos(np.radians(bg)))
        q = df[(df.MN_relHor == bp) & (df.HN == bg) & (df.RA == ra)]
        rng = np.random.default_rng(zlib.crc32(f"outward|{side}|{bp}|{bg}|{ra}".encode()))
        for _, m in q.iloc[np.sort(rng.choice(len(q), size=N_STEPS, replace=False))].iterrows():
            dhi = max(m.GHI - m.DNI * np.sin(np.radians(m.elevation)), 0.0)
            sky = sky_rad(m.elevation, m.azimuth, m.DNI, dhi, slope_irradiance(m, bg), ALBEDO)
            row = dict(side=side, beta_p=bp, beta_g=bg, RA=ra, rho_bar=arr.rho_bar)
            for label, t in (("thin", 1e-5), ("thick", THICKNESS)):
                key = f"outward|{side}|{bp}|{bg}|{ra}|{m.Timestamp}|{LONG / 2}|t{t:.4f}"
                if key not in cache:
                    octree = build_octree(WORK, env, "scene", {"scene.rad": scene(arr, ROWS[side], LONG / 2, thickness=t),
                                                                "sky.rad": sky})
                    cache[key] = float(np.mean(irradiance(octree, sensor_rays(arr, side, LONG / 2, thickness=t),
                                                          WORK, env, n_proc=8)))
                    cache_file.write_text(json.dumps(cache))
                row[f"rt_{label}"] = cache[key]
                row[f"vf_{label}"] = model(arr, side, m, 0.0 if label == "thin" else THICKNESS)
            rows.append(row)
            print(side, bp, bg, ra, m.Timestamp, f"thin RT {row['rt_thin']:.1f} VF {row['vf_thin']:.1f} | "
                  f"thick RT {row['rt_thick']:.1f} VF {row['vf_thick']:.1f}", flush=True)
    r = pd.DataFrame(rows)
    s = r.groupby(["side", "beta_p", "beta_g", "RA", "rho_bar"]).apply(lambda g: pd.Series({
        "vf/rt thin": g.vf_thin.sum() / g.rt_thin.sum(), "vf/rt thick": g.vf_thick.sum() / g.rt_thick.sum(),
        "min sample": min((g.vf_thin / g.rt_thin).min(), (g.vf_thick / g.rt_thick).min()),
        "max sample": max((g.vf_thin / g.rt_thin).max(), (g.vf_thick / g.rt_thick).max())})).reset_index()
    text = ("## Outward-facing sides of a three-row array: slopevf / Radiance (600 H rows, sum over 8 time steps)\n"
            + s.round(4).to_string(index=False))
    (RESULTS / "outward_check.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
