"""Verification of the diffuse panel reflection against Radiance.

The panels of the 'ideal' scenes (see radscene.py) are given a Lambertian surface of reflectance
R_D on both sides (Radiance 'plastic'); slopevf uses PanelGlass(reflectance=0, diffuse=R_D), i.e.
no specular part. The diffuse contribution in Radiance is the difference between the scenes with
and without reflecting panels. It also contains paths that are not part of the model (multiple
reflections between the rows, panel -> ground -> panel, reflection reaching rows further away),
which are of second order in R_D.

Usage:  python diffuse_check.py          (results in results/diffuse_check.*)
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import radscene as rs
import run_benchmark as rb
from reflection_check import GEOMETRIES, SKIES
from slopevf import FarGround, PanelGlass, Settings, simulate

R_D = 0.1
RTRACE = ["-ab", "3", "-aa", "0", "-ad", "16384", "-as", "0", "-lw", "1e-7", "-lr", "8"]


def rt_case(name, arr, sky, refl, env):
    key = f"{name}_{sky}_D{refl}"
    cache = rb.WORK / "diffuse" / f"{key}.json"
    if cache.exists():
        return np.array(json.loads(cache.read_text()))
    wd = rb.WORK / "diffuse" / key
    wd.mkdir(parents=True, exist_ok=True)
    el, az, dni, dhi, ghi = rb.sky_inputs(sky, arr.gamma)
    oct_ = rs.build_octree(wd, env, "scene", {
        "sky.rad": rs.sky_rad(el, az, dni, dhi, ghi, rb.ALBEDO_FG),
        "scene.rad": rs.scene_rad(arr, "ideal", rb.ALBEDO, ghi, rb.ALBEDO_FG, panel_diffuse=refl)})
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
                   panel_glass=PanelGlass(reflectance=lambda c: np.zeros_like(c), diffuse=R_D),
                   settings=Settings(n_chi=k * rb.N_SENS, sky="medium"))
    idx = k * np.arange(rb.N_SENS) + k // 2
    return dict(front=res.front[0, idx], back=res.back[0, idx],
                front_diff=res.front_diffuse[0, idx], back_diff=res.back_diffuse[0, idx])


def main():
    env = rs.radiance_env()
    rows = []
    for name, arr in GEOMETRIES.items():
        for sky in SKIES:
            t0 = time.perf_counter()
            black, diffuse = rt_case(name, arr, sky, 0.0, env), rt_case(name, arr, sky, R_D, env)
            vf = vf_case(arr, sky)
            row = dict(geometry=name, sky=sky)
            for s, side in enumerate(("front", "back")):
                rt_diff = diffuse[s] - black[s]
                row[f"{side}_diff_rt"] = rt_diff.mean()
                row[f"{side}_diff_vf"] = vf[f"{side}_diff"].mean()
                row[f"{side}_total_rt"] = diffuse[s].mean()
                row[f"{side}_total_vf"] = vf[side].mean()
                row[f"{side}_total_diff_%"] = 100 * (vf[side].mean() / diffuse[s].mean() - 1)
                row[f"{side}_diff_share_rt_%"] = 100 * rt_diff.mean() / diffuse[s].mean()
            rows.append(row)
            print(f"{name:11s} {sky:17s} back diffuse RT {row['back_diff_rt']:6.2f} VF {row['back_diff_vf']:6.2f} "
                  f"(share {row['back_diff_share_rt_%']:4.1f}%, total diff {row['back_total_diff_%']:+.2f}%) | "
                  f"front RT {row['front_diff_rt']:6.2f} VF {row['front_diff_vf']:6.2f} "
                  f"(total diff {row['front_total_diff_%']:+.2f}%) [{time.perf_counter() - t0:.0f} s]", flush=True)
    df = pd.DataFrame(rows)
    rb.RESULTS.mkdir(exist_ok=True)
    df.to_csv(rb.RESULTS / "diffuse_check.csv", index=False)
    lines = [f"## Diffuse panel reflection check (Lambertian panels, R_D = {R_D}, {len(df)} cases)"]
    for side in ("front", "back"):
        d = df[f"{side}_total_diff_%"]
        e = df[f"{side}_diff_vf"] - df[f"{side}_diff_rt"]
        lines.append(f"{side:5s}: total (incl. reflection) |max| dev {d.abs().max():.2f}%, mean {d.mean():+.2f}%;  "
                     f"diffuse component vf - rt: mean {e.mean():+.3f}, |max| {e.abs().max():.3f} W/m2;  "
                     f"diffuse share (RT) max {df[f'{side}_diff_share_rt_%'].max():.1f}%")
    text = "\n".join(lines)
    (rb.RESULTS / "diffuse_check.txt").write_text(text)
    print(text)
    return df


if __name__ == "__main__":
    main()
