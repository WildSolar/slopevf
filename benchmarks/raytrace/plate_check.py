"""Finite base plate of the Totalp miniature array with snow-free or snow-covered surroundings (Radiance).

The white base plate (albedo 0.79) is 1830 mm long along the slope, centred on the middle row,
and about as wide as the rows (15.4 H). Beyond it, the ground is the natural surroundings: snow-free
(albedo 0.2) until mid-November 2024 and snow-covered (albedo 0.85) from 21 November 2024. Unlike
the plate between the rows, the surroundings beyond the row ends are not shaded by the rows.

For every measured configuration (>= 20 measurements) and N_STEPS sampled time steps, the
irradiance on the measuring module (15.4 H rows, 15 mm stack) is ray-traced for

  plate_free   plate + snow-free surroundings (albedo 0.2)
  plate_snow   plate + snow-covered surroundings (albedo 0.85)

and divided by slopevf for the same time steps in its idealisation (infinitely long rows, 15 mm
stack, uniform albedo 0.79, which agrees with Radiance within 0.1 % on the backside, see
finite_array_check.py). The factors correct the validation (totalp_validation.py) for the plate
and its surroundings; the irradiance is linear in the albedo of the surroundings, so partial snow
cover interpolates between them. The surroundings are approximated by the plane of the plate; the
terrain below the horizon glows with the albedo of the surroundings. Support posts and the sleds
beside the plate are not modelled.

Usage:  python plate_check.py      (results in results/plate_check.csv / .txt)
"""

from __future__ import annotations

import json
import zlib

import numpy as np
import pandas as pd

from finite_array_check import (DATA, RESULTS, ROW_LENGTH, ROWS, THICKNESS, WORK, scene, sensor_rays,
                                slope_irradiance, slopevf_irradiance)
from radscene import _polygon, _to_world, build_octree, irradiance, radiance_env, sky_rad
from slopevf import Array
from slopevf.geometry import Geometry

H_MM = 96.0
PLATE_HALF = 915.0 / H_MM          # along the slope, from the middle row
SURROUNDINGS = {"plate_free": 0.2, "plate_snow": 0.85}
PLATE_ALBEDO = 0.79
N_STEPS = 8                        # sampled measurement time steps per configuration
MIN_MEASUREMENTS = 20              # as in totalp_validation.py


def plate_scene(array: Array, offsets, albedo_s, margin=300.0):
    """Rows (15.4 H, 15 mm) on a plate centred on the middle row, framed by the surroundings."""
    rows = scene(array, offsets, ROW_LENGTH / 2, thickness=THICKNESS)
    rows = rows[:rows.rindex("ground_mat polygon ground")]       # drop the uniform ground (last object)
    g = Geometry(array)
    bg, rho = g.bg, g.rho
    yc = np.mean(offsets) * rho                        # middle row (horizontal coordinate)
    py = PLATE_HALF * np.cos(bg)                       # plate half-length, horizontal
    px = ROW_LENGTH / 2
    big = ROW_LENGTH / 2 + margin
    y0, y1 = yc - py, yc + py
    Y0, Y1 = yc - py - margin, yc + py + margin
    z = lambda y: y * np.tan(bg)
    rect = lambda xa, xb, ya, yb: np.array([[xa, ya, z(ya)], [xb, ya, z(ya)], [xb, yb, z(yb)], [xa, yb, z(yb)]])
    rad = [rows,
           f"void plastic plate_mat\n0\n0\n5 {PLATE_ALBEDO} {PLATE_ALBEDO} {PLATE_ALBEDO} 0 0\n",
           f"void plastic surr_mat\n0\n0\n5 {albedo_s} {albedo_s} {albedo_s} 0 0\n",
           _polygon("plate_mat", "plate", _to_world(rect(-px, px, y0, y1), array.gamma))]
    for name, r in (("s_down", rect(-big, big, Y0, y0)), ("s_up", rect(-big, big, y1, Y1)),
                    ("s_left", rect(-big, -px, y0, y1)), ("s_right", rect(px, big, y0, y1))):
        rad.append(_polygon("surr_mat", name, _to_world(r, array.gamma)))
    return "".join(rad)


def configurations(df):
    """Measured configurations (side, tilt, slope, RA) with at least MIN_MEASUREMENTS records."""
    valid = df[(df.MN_relHor > df.HN) & (df.MN_relHor <= 90)]
    n = valid.groupby(["side", "MN_relHor", "HN", "RA"]).size()
    return [tuple(k) for k, v in n.items() if v >= MIN_MEASUREMENTS]


def samples(df, side, bp, bg, ra):
    """N_STEPS time steps of a configuration, drawn reproducibly per configuration."""
    q = df[(df.side == side) & (df.MN_relHor == bp) & (df.HN == bg) & (df.RA == ra)]
    rng = np.random.default_rng(zlib.crc32(f"{side}|{bp}|{bg}|{ra}".encode()))
    return q.iloc[np.sort(rng.choice(len(q), size=min(N_STEPS, len(q)), replace=False))]


def main():
    env = radiance_env()
    cache_file = WORK / "cache.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    df = pd.read_csv(DATA, sep=";", decimal=",")
    df["side"] = np.where(df["channel"] == 22, "back", "front")
    rows = []
    for side, bp, bg, ra in configurations(df):
        arr = Array(bp, bg, ra * np.cos(np.radians(bg)) / 96.0, 1.77 / np.cos(np.radians(bg)))
        geo = (bp, bg, arr.rho_bar, arr.h_bar)
        for _, m in samples(df, side, bp, bg, ra).iterrows():
            el, az, dni, ghi = m.elevation, m.azimuth, m.DNI, m.GHI
            dhi = max(ghi - dni * np.sin(np.radians(el)), 0.0)
            res = {"slopevf": slopevf_irradiance(geo, side, m, THICKNESS)}
            e_slope = slope_irradiance(m, bg)
            for name, a_s in SURROUNDINGS.items():
                key = f"{side}|{bp}|{bg}|{ra}|{m.Timestamp}|plate{PLATE_HALF:.3f}|surr{a_s}|module4|t{THICKNESS:.4f}"
                if key not in cache:
                    rad = plate_scene(arr, ROWS[side], a_s)
                    sky = sky_rad(el, az, dni, dhi, e_slope, a_s)
                    octree = build_octree(WORK, env, "scene", {"scene.rad": rad, "sky.rad": sky})
                    e = irradiance(octree, sensor_rays(arr, side, ROW_LENGTH / 2, thickness=THICKNESS), WORK, env,
                                   n_proc=8)
                    cache[key] = float(np.mean(e))
                    cache_file.write_text(json.dumps(cache))
                res[name] = cache[key]
            rows.append(dict(side=side, beta_p=bp, beta_g=bg, RA=ra, rho_bar=arr.rho_bar, time=m.Timestamp,
                             elevation=el, kd=dhi / ghi if ghi > 0 else np.nan, **{f"e_{k}": v for k, v in res.items()}))
            print(f"{side} {bp}/{bg}/{ra} {m.Timestamp}: " + " / ".join(f"{v:.1f}" for v in res.values()) + " W/m2",
                  flush=True)
    r = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    r.to_csv(RESULTS / "plate_check_samples.csv", index=False)
    out = []
    for key, g in r.groupby(["side", "beta_p", "beta_g", "RA", "rho_bar"]):
        row = dict(zip(["side", "beta_p", "beta_g", "RA", "rho_bar"], key), n=len(g))
        for name in SURROUNDINGS:
            row[name] = g[f"e_{name}"].sum() / g.e_slopevf.sum()
        row["snow/free"] = g.e_plate_snow.sum() / g.e_plate_free.sum()
        out.append(row)
    s = pd.DataFrame(out)
    s.to_csv(RESULTS / "plate_check.csv", index=False)
    text = ("## Finite base plate (1830 mm, centred on the middle row) with surroundings, Radiance\n"
            "plate_free / plate_snow: E(Radiance: 15.4 H rows, 15 mm, plate 0.79 + surroundings 0.2 / 0.85) / "
            "E(slopevf: infinite rows, 15 mm, uniform 0.79)\n" + s.round(3).to_string(index=False))
    (RESULTS / "plate_check.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
