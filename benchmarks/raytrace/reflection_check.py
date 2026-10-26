"""Verification of the panel-to-panel reflection against Radiance.

The panels of the 'ideal' scenes (see radscene.py) are given a perfectly specular surface of
constant reflectance R on both sides (Radiance 'mirror', for which exact virtual light sources,
i.e. mirror images of the sun, are generated). slopevf uses the same constant reflectance
(PanelGlass(reflectance=...)), such that the geometry of the reflection is tested; the Fresnel
reflectance itself is verified analytically in the unit tests.

The reflected contribution in Radiance is the difference between the scenes with and without
mirror panels. It also contains paths that are not part of the model (reflection onto the
ground and from there onto the panels, reflection reaching rows further away).

Usage:  python reflection_check.py          (results in results/reflection_check.*)
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import radscene as rs
import run_benchmark as rb
from slopevf import Array, FarGround, PanelGlass, Settings, simulate

R = 0.05
GEOMETRIES = {
    "totalp_p90": Array(90, 30, 1.04, 2.04),
    "totalp_p60": Array(60, 30, 1.04, 2.04),
    "g20_p60": Array.from_shading_angle(60, 20, 15, 1.0),
    "flat_p90": Array.from_shading_angle(90, 0, 15, 1.0),
    "flat_p30": Array.from_shading_angle(30, 0, 15, 1.0),
}
SKIES = ["clear_front_low", "clear_front_high", "clear_behind", "hazy", "overcast"]
RTRACE = rs.DEFAULT_RTRACE + ["-dr", "1"]


def rt_case(name, arr, sky, refl, env):
    key = f"{name}_{sky}_R{refl}"
    cache = rb.WORK / "reflection" / f"{key}.json"
    if cache.exists():
        return np.array(json.loads(cache.read_text()))
    wd = rb.WORK / "reflection" / key
    wd.mkdir(parents=True, exist_ok=True)
    el, az, dni, dhi, ghi = rb.sky_inputs(sky, arr.gamma)
    oct_ = rs.build_octree(wd, env, "scene", {
        "sky.rad": rs.sky_rad(el, az, dni, dhi, ghi, rb.ALBEDO_FG),
        "scene.rad": rs.scene_rad(arr, "ideal", rb.ALBEDO, ghi, rb.ALBEDO_FG, panel_reflectance=refl)})
    front, back = rs.sensors(arr, rb.CHI, rb.X_POS)
    e = rs.irradiance(oct_, np.concatenate([front.reshape(-1, 6), back.reshape(-1, 6)]), wd, env,
                      RTRACE, rb.N_PROC).reshape(2, len(rb.X_POS), rb.N_SENS).mean(axis=1)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(e.tolist()))
    return e


def vf_case(arr, sky):
    el, az, dni, dhi, ghi = rb.sky_inputs(sky, arr.gamma)
    k = 15
    res = simulate(arr, dict(dni=[dni], ghi=[ghi], solar_azimuth=[az], solar_elevation=[el]),
                   albedo=rb.ALBEDO, far_ground=FarGround(0, 0), albedo_fg=rb.ALBEDO_FG,
                   panel_glass=PanelGlass(reflectance=lambda c: np.full_like(c, R)),
                   settings=Settings(n_chi=k * rb.N_SENS, sky="medium"))
    idx = k * np.arange(rb.N_SENS) + k // 2
    return dict(front=res.front[0, idx], back=res.back[0, idx],
                front_refl=res.front_reflected[0, idx], back_refl=res.back_reflected[0, idx])


def main():
    env = rs.radiance_env()
    rows, profiles = [], {}
    for name, arr in GEOMETRIES.items():
        for sky in SKIES:
            t0 = time.perf_counter()
            black, mirror = rt_case(name, arr, sky, 0.0, env), rt_case(name, arr, sky, R, env)
            vf = vf_case(arr, sky)
            row = dict(geometry=name, sky=sky)
            for s, side in enumerate(("front", "back")):
                rt_refl = mirror[s] - black[s]
                row[f"{side}_refl_rt"] = rt_refl.mean()
                row[f"{side}_refl_vf"] = vf[f"{side}_refl"].mean()
                row[f"{side}_total_rt"] = mirror[s].mean()
                row[f"{side}_total_vf"] = vf[side].mean()
                row[f"{side}_total_diff_%"] = 100 * (vf[side].mean() / mirror[s].mean() - 1)
                row[f"{side}_refl_share_rt_%"] = 100 * rt_refl.mean() / mirror[s].mean()
                profiles[(name, sky, side)] = pd.DataFrame(dict(chi=rb.CHI, rt=rt_refl, vf=vf[f"{side}_refl"]))
            rows.append(row)
            print(f"{name:11s} {sky:17s} back refl RT {row['back_refl_rt']:6.2f} VF {row['back_refl_vf']:6.2f} "
                  f"(share {row['back_refl_share_rt_%']:4.1f}%, total diff {row['back_total_diff_%']:+.2f}%) | "
                  f"front refl RT {row['front_refl_rt']:6.2f} VF {row['front_refl_vf']:6.2f} "
                  f"(total diff {row['front_total_diff_%']:+.2f}%) [{time.perf_counter() - t0:.0f} s]", flush=True)
    df = pd.DataFrame(rows)
    rb.RESULTS.mkdir(exist_ok=True)
    df.to_csv(rb.RESULTS / "reflection_check.csv", index=False)
    pd.concat({f"{n}|{s}|{d}": p for (n, s, d), p in profiles.items()}, names=["case"]).to_csv(
        rb.RESULTS / "reflection_check_profiles.csv")
    lines = [f"## Panel reflection check (constant R = {R}, {len(df)} cases)"]
    for side in ("front", "back"):
        d = df[f"{side}_total_diff_%"]
        e = df[f"{side}_refl_vf"] - df[f"{side}_refl_rt"]
        lines.append(f"{side:5s}: total (incl. reflection) |max| dev {d.abs().max():.2f}%, mean {d.mean():+.2f}%;  "
                     f"reflected component vf - rt: mean {e.mean():+.3f}, |max| {e.abs().max():.3f} W/m2;  "
                     f"reflected share (RT) max {df[f'{side}_refl_share_rt_%'].max():.1f}%")
    text = "\n".join(lines)
    (rb.RESULTS / "reflection_check.txt").write_text(text)
    print(text)
    return df


if __name__ == "__main__":
    main()
