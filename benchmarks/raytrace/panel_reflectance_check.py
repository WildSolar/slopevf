"""Diffuse reflectance of the module surfaces (Radiance sensitivity check).

slopevf and the ray-tracing benchmarks treat the panels as black apart from the specular Fresnel
reflection of the glass. Real modules also reflect diffusely: cells behind the glass, inactive
areas (backsheet between and around the cells), frames and the edges of the stack. Facing rows
exchange this radiation, which matters most for the backside at dense row spacing.

For selected Totalp configurations and sampled time steps, the backside (and one frontside)
irradiance is ray-traced with Lambertian panel surfaces of reflectance 0 and R_PANEL (infinitely
long rows, 15 mm stack, uniform ground albedo 0.79) and the ratio is reported.

Usage:  python panel_reflectance_check.py      (results in results/panel_reflectance.txt)
"""

from __future__ import annotations

import json
import zlib

import numpy as np
import pandas as pd

from finite_array_check import (ALBEDO, DATA, LONG, RESULTS, ROWS, THICKNESS, WORK, scene, sensor_rays,
                                slope_irradiance)
from radscene import build_octree, irradiance, radiance_env, sky_rad
from slopevf import Array

R_PANEL = 0.1
N_STEPS = 6
CONFIGS = [("back", 90, 0, 263), ("back", 90, 0, 688), ("back", 90, 20, 115), ("back", 90, 30, 115),
           ("back", 60, 20, 115), ("back", 60, 0, 688), ("front", 90, 20, 115)]
BLACK = "void plastic panel_mat\n0\n0\n5 0 0 0 0 0\n"


def main():
    env = radiance_env()
    cache_file = WORK / "cache.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    df = pd.read_csv(DATA, sep=";", decimal=",")
    df["side"] = np.where(df["channel"] == 22, "back", "front")
    rows = []
    for side, bp, bg, ra in CONFIGS:
        arr = Array(bp, bg, ra * np.cos(np.radians(bg)) / 96.0, 1.77 / np.cos(np.radians(bg)))
        q = df[(df.side == side) & (df.MN_relHor == bp) & (df.HN == bg) & (df.RA == ra)]
        rng = np.random.default_rng(zlib.crc32(f"{side}|{bp}|{bg}|{ra}".encode()))
        for _, m in q.iloc[np.sort(rng.choice(len(q), size=N_STEPS, replace=False))].iterrows():
            dhi = max(m.GHI - m.DNI * np.sin(np.radians(m.elevation)), 0.0)
            sky = sky_rad(m.elevation, m.azimuth, m.DNI, dhi, slope_irradiance(m, bg), ALBEDO)
            res = {}
            for r in (0.0, R_PANEL):
                key = f"{side}|{bp}|{bg}|{ra}|{m.Timestamp}|{LONG / 2}|module4|t{THICKNESS:.4f}|panelR{r}"
                if key not in cache:
                    rad = scene(arr, ROWS[side], LONG / 2, thickness=THICKNESS).replace(
                        BLACK, f"void plastic panel_mat\n0\n0\n5 {r} {r} {r} 0 0\n")
                    octree = build_octree(WORK, env, "scene", {"scene.rad": rad, "sky.rad": sky})
                    e = irradiance(octree, sensor_rays(arr, side, LONG / 2, thickness=THICKNESS), WORK, env, n_proc=8)
                    cache[key] = float(np.mean(e))
                    cache_file.write_text(json.dumps(cache))
                res[r] = cache[key]
            rows.append(dict(side=side, beta_p=bp, beta_g=bg, RA=ra, rho_bar=arr.rho_bar, e0=res[0.0], e1=res[R_PANEL]))
            print(side, bp, bg, ra, m.Timestamp, f"{res[0.0]:.1f} -> {res[R_PANEL]:.1f}", flush=True)
    r = pd.DataFrame(rows)
    s = r.groupby(["side", "beta_p", "beta_g", "RA", "rho_bar"]).apply(
        lambda g: g.e1.sum() / g.e0.sum()).rename(f"E(R={R_PANEL}) / E(R=0)").reset_index()
    text = (f"## Diffuse panel reflectance {R_PANEL} vs. black panels (Radiance, 600 H rows, 15 mm stack)\n"
            + s.round(3).to_string(index=False))
    (RESULTS / "panel_reflectance.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
