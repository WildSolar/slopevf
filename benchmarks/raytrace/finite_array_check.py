"""Finite row length and module thickness of the Totalp miniature array (Radiance).

slopevf is two-dimensional with infinitely thin panels: the rows are infinitely long. The Totalp
array has three rows of about 15.4 panel heights, and each row is a stack of about 15 mm
(0.16 H): two monofacial modules back to back with a central rod. Light entering past the row
ends raises the irradiance, mostly on the back side; the thickness widens the shadows on the
ground and moves the module surfaces off the row plane.

For selected configurations and measured time steps, the irradiance averaged over the measuring
module (module 4 of 7, i.e. the central seventh of the row, over the full height, on the outer
face of the stack) is ray-traced for

  long    600 H rows, thin panels (= slopevf's idealisation)
  finite  15.4 H rows, thin panels
  thick   15.4 H rows, 15 mm stack centred on the row plane

and the factors finite/long (length), thick/finite (thickness) and thick/long (total) are
formed as sum E / sum E over the sampled time steps.

Layout as at Totalp: the frontside module is on the last (uphill) row (rows -2, -1, 0), the
backside module on the first (downhill) row (rows 0, +1, +2). Black panels, Lambertian snow
(albedo 0.79) extending far beyond the array, slopevf's Perez sky; no horizon.

Usage:  python finite_array_check.py      (results in results/finite_array.csv / .txt)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from radscene import _box, _polygon, _to_world, build_octree, irradiance, radiance_env, sky_rad
from slopevf import Array
from slopevf.geometry import Geometry

HERE = Path(__file__).parent
WORK = HERE / "work" / "rowlength"
RESULTS = HERE / "results"
DATA = HERE.parents[1] / "data" / "totalp" / "Totalp_Measurement_Data.csv"
ALBEDO = 0.79
ROW_LENGTH = 15.4
LONG = 600.0
N_STEPS = 8                  # sampled measurement time steps per configuration
N_X, N_CHI = 7, 8            # sensors along the module and over the height
MODULE_HALF_WIDTH = ROW_LENGTH / 14   # measuring module: 4 of 7 along the row
THICKNESS = 15.0 / 96.0               # stack: two monofacial modules back to back + 25 x 7 mm rod lying flat
ROWS = {"front": (-2, -1, 0), "back": (0, 1, 2)}
CONFIGS = [  # (side, tilt, slope, RA [mm])
    ("back", 90, 0, 263), ("back", 90, 0, 688), ("back", 90, 10, 166), ("back", 90, 20, 115),
    ("back", 90, 20, 205), ("back", 90, 30, 115), ("back", 60, 0, 263), ("back", 60, 20, 115),
    ("back", 60, 30, 137), ("front", 90, 0, 263), ("front", 90, 20, 115), ("front", 60, 20, 166),
]


def scene(array: Array, offsets, half_length, thickness=1e-5, margin=300.0):
    g = Geometry(array)
    bp, bg, rho, h = g.bp, g.bg, g.rho, g.h
    t_vec = np.array([0.0, np.cos(bp), np.sin(bp)])
    n_f = np.array([0.0, -np.sin(bp), np.cos(bp)])
    rad = ["void plastic panel_mat\n0\n0\n5 0 0 0 0 0\n",
           f"void plastic ground_mat\n0\n0\n5 {ALBEDO} {ALBEDO} {ALBEDO} 0 0\n"]
    for i in offsets:
        centre = np.array([0.0, i * rho, h + i * rho * np.tan(bg)])
        origin = centre - 0.5 * t_vec - 0.5 * thickness * n_f + np.array([-half_length, 0, 0])
        rad.append(_box("panel_mat", f"row{i}", origin, [2 * half_length, 0, 0], t_vec, thickness * n_f,
                        array.gamma))
    y_lo, y_hi, xg = -2 * rho - margin, 2 * rho + margin, half_length + margin
    pts = np.array([[-xg, y_lo, y_lo * np.tan(bg)], [xg, y_lo, y_lo * np.tan(bg)],
                    [xg, y_hi, y_hi * np.tan(bg)], [-xg, y_hi, y_hi * np.tan(bg)]])
    rad.append(_polygon("ground_mat", "ground", _to_world(pts, array.gamma)))
    return "".join(rad)


def sensor_rays(array: Array, side, half_length, offset=0.001, thickness=1e-5):
    """Sensors on the central module of row 0 (same positions for finite and long rows)."""
    g = Geometry(array)
    t_vec = np.array([0.0, np.cos(g.bp), np.sin(g.bp)])
    n = np.array([0.0, -np.sin(g.bp), np.cos(g.bp)]) * (1.0 if side == "front" else -1.0)
    span = MODULE_HALF_WIDTH
    x = -span + (np.arange(N_X) + 0.5) * 2 * span / N_X
    chi = -0.5 + (np.arange(N_CHI) + 0.5) / N_CHI
    P = (np.array([0.0, 0.0, g.h]) + x[:, None, None] * np.array([1.0, 0, 0])
         + chi[None, :, None] * t_vec + (thickness / 2 + offset) * n)
    pw = _to_world(P, array.gamma).reshape(-1, 3)
    dw = np.broadcast_to(_to_world(n, array.gamma), pw.shape)
    return np.concatenate([pw, dw], axis=-1)


def samples(df, side, bp, bg, ra, rng):
    q = df[(df.side == side) & (df.MN_relHor == bp) & (df.HN == bg) & (df.RA == ra)]
    return q.iloc[np.sort(rng.choice(len(q), size=min(N_STEPS, len(q)), replace=False))]


CASES = {  # name: (row half length, panel thickness)
    "long": (LONG / 2, 1e-5),
    "finite": (ROW_LENGTH / 2, 1e-5),
    "thick": (ROW_LENGTH / 2, THICKNESS),
    "long_thick": (LONG / 2, THICKNESS),
}


def slopevf_irradiance(arr_args, side, m, thickness):
    """slopevf for the same scene: finite rows, thickness, ground beyond the array as far ground."""
    from slopevf import FarGround, Settings, simulate
    bp, bg, rho, h = arr_args
    arr = Array(bp, bg, rho, h, 0.0, thickness=thickness)
    w = dict(dni=[m.DNI], ghi=[m.GHI], solar_azimuth=[m.azimuth], solar_elevation=[m.elevation])
    rows = (2, 0) if side == "front" else (0, 2)
    r = simulate(arr, w, albedo=ALBEDO, far_ground=FarGround(bg, 0.0), albedo_fg=ALBEDO,
                 settings=Settings(sky="medium", n_chi=N_CHI * 4, finite_rows=rows))
    return float((r.front_mean if side == "front" else r.back_mean).iloc[0])


_GRID = None


def slope_irradiance(m, beta_g):
    """Irradiance on the unshaded slope (south-facing, tilt beta_g) with slopevf's sky: the terrain
    below the horizon is the continuing slope, glowing with albedo * this / pi (as in slopevf,
    whose far ground has the same orientation)."""
    global _GRID
    from slopevf.geometry import direction_vector, surface_normal
    from slopevf.sky import make_sky_grid, sky_radiance
    if _GRID is None:
        _GRID = make_sky_grid("medium")
    el, az = np.radians(m.elevation), np.radians(m.azimuth)
    L = sky_radiance(_GRID, [m.DNI], [m.GHI], [az], [el], sun_model="point").L[:, 0]
    n = surface_normal(np.radians(beta_g), 0.0)
    diffuse = (_GRID.solid_angle * np.maximum(_GRID.vec @ n, 0.0)) @ L
    return float(diffuse + m.DNI * max(direction_vector(el, az) @ n, 0.0))


def _factors(g, num, den):
    q = g[num] / g[den]
    return g[num].sum() / g[den].sum(), q.min(), q.max()


def main():
    env = radiance_env()
    WORK.mkdir(parents=True, exist_ok=True)
    cache_file = WORK / "cache.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    df = pd.read_csv(DATA, sep=";", decimal=",")
    df["side"] = np.where(df["channel"] == 22, "back", "front")
    rng = np.random.default_rng(1)
    rows = []
    for side, bp, bg, ra in CONFIGS:
        arr = Array(bp, bg, ra * np.cos(np.radians(bg)) / 96.0, 1.77 / np.cos(np.radians(bg)))
        rad = {name: scene(arr, ROWS[side], L, thickness=t) for name, (L, t) in CASES.items()}
        for _, m in samples(df, side, bp, bg, ra, rng).iterrows():
            el, az, dni, ghi = m.elevation, m.azimuth, m.DNI, m.GHI
            dhi = max(ghi - dni * np.sin(np.radians(el)), 0.0)
            # glow below the horizon: the continuing slope (radiance albedo * E_slope / pi)
            sky = sky_rad(el, az, dni, dhi, slope_irradiance(m, bg), ALBEDO)
            res = {}
            for name, (L, t) in CASES.items():
                key = (f"{side}|{bp}|{bg}|{ra}|{m.Timestamp}|{L}|module4" + ("" if t < 1e-3 else f"|t{t:.4f}")
                       + "|slopeglow")
                if key not in cache:
                    octree = build_octree(WORK, env, "scene", {"scene.rad": rad[name], "sky.rad": sky})
                    e = irradiance(octree, sensor_rays(arr, side, L, thickness=t), WORK, env, n_proc=8)
                    cache[key] = float(np.mean(e))
                    cache_file.write_text(json.dumps(cache))
                res[name] = cache[key]
            geo = (bp, bg, arr.rho_bar, arr.h_bar)
            res["slopevf"] = slopevf_irradiance(geo, side, m, 0.0)
            res["slopevf_thick"] = slopevf_irradiance(geo, side, m, THICKNESS)
            rows.append(dict(side=side, beta_p=bp, beta_g=bg, RA=ra, rho_bar=arr.rho_bar, time=m.Timestamp,
                             elevation=el, dni=dni, ghi=ghi, **{f"e_{k}": v for k, v in res.items()}))
            print(f"{side} {bp}/{bg}/{ra} {m.Timestamp}: " + " / ".join(f"{v:.1f}" for v in res.values())
                  + " W/m2", flush=True)
    r = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    r.to_csv(RESULTS / "finite_array_samples.csv", index=False)
    out = []
    for key, g in r.groupby(["side", "beta_p", "beta_g", "RA", "rho_bar"]):
        row = dict(zip(["side", "beta_p", "beta_g", "RA", "rho_bar"], key), n=len(g))
        for label, num, den in (("length", "e_finite", "e_long"), ("thickness", "e_thick", "e_finite"),
                                ("total", "e_thick", "e_long"),
                                ("slopevf_thin", "e_slopevf", "e_long"),
                                ("slopevf_thick", "e_slopevf_thick", "e_long_thick"),
                                ("thickness_long", "e_long_thick", "e_long"),
                                ("thickness_slopevf", "e_slopevf_thick", "e_slopevf")):
            row[label], row[f"{label}_min"], row[f"{label}_max"] = _factors(g, num, den)
        out.append(row)
    s = pd.DataFrame(out)
    s.to_csv(RESULTS / "finite_array.csv", index=False)
    mm = THICKNESS * 96
    cols = ["side", "beta_p", "beta_g", "RA", "rho_bar", "n", "length", "thickness", "total"]
    cols2 = ["side", "beta_p", "beta_g", "RA", "slopevf_thin", "slopevf_thick", "thickness_long",
             "thickness_slopevf"]
    text = ("## Finite array, central module (4 of 7), sampled Totalp time steps\n"
            f"length: E(15.4 H, thin) / E(600 H, thin); thickness: E(15.4 H, {mm:.0f} mm) / E(15.4 H, thin); "
            f"total: E(15.4 H, {mm:.0f} mm) / E(600 H, thin)\n" + s[cols].round(3).to_string(index=False)
            + "\n\n## slopevf (finite rows, ground beyond the array as far ground) vs. Radiance, 600 H rows\n"
            f"slopevf_thin: slopevf(thin) / E(600 H, thin); slopevf_thick: slopevf({mm:.0f} mm) / "
            f"E(600 H, {mm:.0f} mm); thickness effect E({mm:.0f} mm) / E(thin) in Radiance and slopevf\n"
            + s[cols2].round(3).to_string(index=False))
    (RESULTS / "finite_array.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
