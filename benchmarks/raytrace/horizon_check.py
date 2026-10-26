"""Verification of the horizon treatment against Radiance with a terrain around the array.

The horizon of the Totalp site (paper convention: 0 = south, positive towards west) is built
as terrain at a distance of R = 10'000 panel heights, much larger than the field, such that
all points of the array see the same horizon (as assumed by the model).

Level A ("wall", implementation check): a wall following the horizon line, glowing with
    exactly the terrain radiance assumed by slopevf (terrain model 'horizontal' or 'facing').
    It blocks the sun when the sun is behind the horizon and irradiates the ground of the array.
Level B ("terrain", terrain model check): Lambertian slopes of tilt BETA_T facing the array,
    rising to the horizon line, physically irradiated in Radiance (sun incl. cast shadows,
    Perez sky, interreflections). Compared with slopevf's 'horizontal' and 'facing' models.

Usage:  python horizon_check.py           (results in results/horizon_check.*)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import radscene as rs
import run_benchmark as rb
from slopevf import Array, FarGround, Horizon, Settings, simulate

HERE = Path(__file__).parent
HORIZON_FILE = HERE.parents[1] / "data" / "totalp" / "Totalp_Horizon.hor"
R = 10_000.0          # distance of the horizon line [panel heights]
ALBEDO_T = 0.5
BETA_T = 30.0         # tilt of the synthetic terrain slopes (level B) [deg]

GEOMETRIES = {
    "totalp_p60": Array(60, 30, 1.04, 2.04),
    "totalp_p90": Array(90, 30, 1.04, 2.04),
    "flat_p30": Array.from_shading_angle(30, 0, 15, 1.0),
}
# absolute sun positions (paper convention); the horizon at the sun azimuth is given for reference
SKIES = {
    "sun_visible": dict(el=15.0, az=-30.0, dni=750.0, dhi=80.0),     # horizon ~3 deg
    "sun_hidden": dict(el=10.0, az=90.0, dni=600.0, dhi=90.0),       # horizon ~14.5 deg
    "hazy": dict(el=20.0, az=-6.1, dni=125.9, dhi=346.9),
    "overcast": dict(el=30.0, az=0.0, dni=0.0, dhi=220.0),
}
RTRACE_B = ["-ab", "3", "-aa", "0", "-ad", "4096", "-as", "0", "-lw", "1e-6", "-lr", "8"]


def _world(az, el, dist):
    """Radiance coordinates (x east, y north, z up) of a point at azimuth az (paper convention)."""
    return np.stack([-np.sin(az) * dist, -np.cos(az) * dist, np.tan(el) * dist], axis=-1)


def _polygon(mod, name, pts):
    """Polygon with its front side facing the array at the origin (glow emits on the front only)."""
    pts = np.asarray(pts)
    normal = np.cross(pts[1] - pts[0], pts[2] - pts[1])
    if np.dot(normal, -pts.mean(axis=0)) < 0:
        pts = pts[::-1]
    return rs._polygon(mod, name, pts)


def horizon_profile():
    d = np.loadtxt(HORIZON_FILE)
    az = np.radians(d[:, 0])
    el = np.radians(np.maximum(d[:, 1], 0.0))
    return np.append(az, az[0] + 2 * np.pi), np.append(el, el[0])


def wall_rad(radiance_per_segment):
    """Level A: vertical wall at distance R from elevation 0 to the horizon, glowing per segment."""
    az, el = horizon_profile()
    txt = []
    for i in range(len(az) - 1):
        if el[i] <= 0 and el[i + 1] <= 0:
            continue
        lv = radiance_per_segment[i]
        txt.append(f"void glow wall_{i}\n0\n0\n4 {lv:.6f} {lv:.6f} {lv:.6f} 0\n")
        p = [_world(az[i], 0.0, R), _world(az[i + 1], 0.0, R),
             _world(az[i + 1], el[i + 1], R), _world(az[i], el[i], R)]
        txt.append(_polygon(f"wall_{i}", f"wall_poly_{i}", p))
    return "".join(txt)


def terrain_rad():
    """Level B: Lambertian slopes of tilt BETA_T facing the array, top edge on the horizon line."""
    az, el = horizon_profile()
    tb = np.tan(np.radians(BETA_T))
    txt = [f"void plastic terrain_mat\n0\n0\n5 {ALBEDO_T} {ALBEDO_T} {ALBEDO_T} 0 0\n"]
    for i in range(len(az) - 1):
        if el[i] <= 0 and el[i + 1] <= 0:
            continue
        top = [_world(az[i], el[i], R), _world(az[i + 1], el[i + 1], R)]
        base = []
        for a, e, t in ((az[i], el[i], top[0]), (az[i + 1], el[i + 1], top[1])):
            dist = R - t[2] / tb                       # horizontal distance of the slope foot
            base.append(_world(a, 0.0, dist))
        # two triangles, normals facing the array (counter-clockwise seen from the origin)
        txt.append(_polygon("terrain_mat", f"terrain_{i}_a", [base[0], base[1], top[1]]))
        txt.append(_polygon("terrain_mat", f"terrain_{i}_b", [base[0], top[1], top[0]]))
    return "".join(txt)


def segment_radiance(horizon: Horizon, sky):
    """Model terrain radiance at the azimuth of each wall segment (level A)."""
    az, el = horizon_profile()
    mid_az = 0.5 * (az[:-1] + az[1:])
    mid_el = np.full(mid_az.shape, 0.01)
    s = SKIES[sky]
    ghi = s["dhi"] + s["dni"] * np.sin(np.radians(s["el"]))
    return horizon.terrain_radiance(mid_el, mid_az, [ALBEDO_T], [s["dni"]], [ghi],
                                    np.radians([s["el"]]), np.radians([s["az"]]))[:, 0]


def rt_case(name, arr, sky, variant, env):
    key = f"{name}_{sky}_{variant}"
    cache = rb.WORK / "horizon" / f"{key}.json"
    if cache.exists():
        return np.array(json.loads(cache.read_text()))
    wd = rb.WORK / "horizon" / key
    wd.mkdir(parents=True, exist_ok=True)
    s = SKIES[sky]
    ghi = s["dhi"] + s["dni"] * np.sin(np.radians(s["el"]))
    if variant == "none":
        extra, params = "", rs.DEFAULT_RTRACE
    elif variant.startswith("wall"):
        model = variant.split("_", 1)[1]
        hz = Horizon.from_file(HORIZON_FILE, terrain="facing" if model.startswith("facing") else "horizontal",
                               terrain_tilt=BETA_T)
        extra, params = wall_rad(segment_radiance(hz, sky)), rs.DEFAULT_RTRACE
    elif variant == "terrain":
        extra, params = terrain_rad(), RTRACE_B
    else:
        raise ValueError(variant)
    oct_ = rs.build_octree(wd, env, "scene", {
        "sky.rad": rs.sky_rad(s["el"], s["az"], s["dni"], s["dhi"], ghi, rb.ALBEDO_FG),
        "scene.rad": rs.scene_rad(arr, "ideal", rb.ALBEDO, ghi, rb.ALBEDO_FG),
        "terrain.rad": extra or "# no terrain\n"})
    front, back = rs.sensors(arr, rb.CHI, rb.X_POS)
    e = rs.irradiance(oct_, np.concatenate([front.reshape(-1, 6), back.reshape(-1, 6)]), wd, env,
                      params, rb.N_PROC).reshape(2, len(rb.X_POS), rb.N_SENS).mean(axis=1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(e.tolist()))
    return e


def vf_case(arr, sky, terrain):
    s = SKIES[sky]
    ghi = s["dhi"] + s["dni"] * np.sin(np.radians(s["el"]))
    kw = {}
    if terrain is not None:
        kw = dict(horizon=Horizon.from_file(HORIZON_FILE, terrain=terrain, terrain_tilt=BETA_T),
                  albedo_terrain=ALBEDO_T)
    k = 15
    res = simulate(arr, dict(dni=[s["dni"]], ghi=[ghi], solar_azimuth=[s["az"]], solar_elevation=[s["el"]]),
                   albedo=rb.ALBEDO, far_ground=FarGround(0, 0), albedo_fg=rb.ALBEDO_FG,
                   settings=Settings(n_chi=k * rb.N_SENS, sky="medium"), **kw)
    idx = k * np.arange(rb.N_SENS) + k // 2
    return np.stack([res.front[0, idx], res.back[0, idx]])


def main():
    env = rs.radiance_env()
    rows = []
    for name, arr in GEOMETRIES.items():
        for sky in SKIES:
            t0 = time.perf_counter()
            rt = {v: rt_case(name, arr, sky, v, env)
                  for v in ("none", "wall_horizontal", "wall_facing", "terrain")}
            vf = {"none": vf_case(arr, sky, None), "horizontal": vf_case(arr, sky, "horizontal"),
                  "facing": vf_case(arr, sky, "facing")}
            row = dict(geometry=name, sky=sky)
            for s, side in enumerate(("front", "back")):
                m = lambda x: x[s].mean()  # noqa: E731
                row[f"{side}_horizon_effect_rt_%"] = 100 * (m(rt["terrain"]) / m(rt["none"]) - 1)
                row[f"{side}_A_none_%"] = 100 * (m(vf["none"]) / m(rt["none"]) - 1)
                # horizon-induced change of slopevf vs. that of the ray tracer (removes the
                # baseline deviation of each scene without horizon)
                for model, wall in (("horizontal", "wall_horizontal"), ("facing", "wall_facing")):
                    row[f"{side}_A_{model}_effect_%"] = 100 * ((m(vf[model]) / m(vf["none"]))
                                                              / (m(rt[wall]) / m(rt["none"])) - 1)
                row[f"{side}_A_horizontal_%"] = 100 * (m(vf["horizontal"]) / m(rt["wall_horizontal"]) - 1)
                row[f"{side}_A_facing_%"] = 100 * (m(vf["facing"]) / m(rt["wall_facing"]) - 1)
                row[f"{side}_B_horizontal_%"] = 100 * (m(vf["horizontal"]) / m(rt["terrain"]) - 1)
                row[f"{side}_B_facing_%"] = 100 * (m(vf["facing"]) / m(rt["terrain"]) - 1)
            rows.append(row)
            print(f"{name:11s} {sky:12s} A: F {row['front_A_horizontal_%']:+.2f}/{row['front_A_facing_%']:+.2f}%"
                  f" B {row['back_A_horizontal_%']:+.2f}/{row['back_A_facing_%']:+.2f}% | "
                  f"B: F {row['front_B_horizontal_%']:+.2f}/{row['front_B_facing_%']:+.2f}% "
                  f"B {row['back_B_horizontal_%']:+.2f}/{row['back_B_facing_%']:+.2f}% | "
                  f"horizon effect (RT, terrain) F {row['front_horizon_effect_rt_%']:+.1f}% "
                  f"[{time.perf_counter() - t0:.0f} s]", flush=True)
    df = pd.DataFrame(rows)
    rb.RESULTS.mkdir(exist_ok=True)
    df.to_csv(rb.RESULTS / "horizon_check.csv", index=False)
    lines = [f"## Horizon check (Totalp horizon, {len(df)} cases), deviation slopevf / Radiance - 1"]
    for side in ("front", "back"):
        for col, label in ((f"{side}_A_none_%", "no horizon (baseline)"),
                           (f"{side}_A_horizontal_%", "A: wall, horizontal model"),
                           (f"{side}_A_facing_%", "A: wall, facing model"),
                           (f"{side}_A_horizontal_effect_%", "A: horizon effect, horizontal"),
                           (f"{side}_A_facing_effect_%", "A: horizon effect, facing"),
                           (f"{side}_B_horizontal_%", "B: lit terrain vs horizontal model"),
                           (f"{side}_B_facing_%", "B: lit terrain vs facing model")):
            d = df[col]
            lines.append(f"{side:5s} {label:36s} mean {d.mean():+6.2f}%  |max| {d.abs().max():6.2f}%")
    text = "\n".join(lines)
    (rb.RESULTS / "horizon_check.txt").write_text(text)
    print(text)
    return df


if __name__ == "__main__":
    main()
