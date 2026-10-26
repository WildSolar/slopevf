"""Ray-tracing check of the ground-reflected component for the largest slopevf/infinite_sheds deviations.

The ground-reflected irradiance is obtained from two Radiance runs (albedo 0.5 and 0) with an
isotropic sky, using the scenes of ../raytrace (horizontal ground, non-reflective thin panels).

Usage:  python raytrace_check.py      (needs Radiance; set RADIANCE_DIR)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "raytrace"))

import radscene as rs           # noqa: E402
import run_benchmark as rb      # noqa: E402
from compare import ALBEDO, SKIES, pvlib_isotropic, slopevf_isotropic   # noqa: E402
from slopevf import Array       # noqa: E402
from slopevf.sky import make_sky_grid   # noqa: E402

CASES = [(60, 10, 0.6, "clear_front_high"), (90, 10, 0.6, "clear_front_high"),
         (30, 10, 0.6, "clear_front_low"), (45, 10, 0.6, "clear_front_high"),
         (90, 10, 0.6, "clear_front_low")]


def isotropic_sky(el, az, dni, dhi):
    """Uniform sky radiance DHI/pi, sun disc carrying DNI, black below the horizon."""
    from slopevf.geometry import direction_vector
    sun = direction_vector(np.radians(el), np.radians(az))
    sx, sy, sz = -sun[0], -sun[1], sun[2]           # Radiance: x = east, y = north
    ls = dhi / np.pi
    txt = (f"void glow sky_glow\n0\n0\n4 {ls:.6f} {ls:.6f} {ls:.6f} 0\n"
           "sky_glow source sky\n0\n0\n4 0 0 1 180\n")
    if dni > 0:
        lsun = dni / rs.SUN_SOLID_ANGLE
        txt += (f"void light solar\n0\n0\n3 {lsun:.6e} {lsun:.6e} {lsun:.6e}\n"
                f"solar source sun\n0\n0\n4 {sx:.8f} {sy:.8f} {sz:.8f} 0.533\n")
    return txt


def main():
    env = rs.radiance_env()
    grid = make_sky_grid("medium")
    chi = rb.CHI
    rows = []
    for bp, sh, h, sky in CASES:
        el, az, dni, dhi = SKIES[sky]
        arr = Array.from_shading_angle(bp, 0, sh, h)
        e = {}
        for alb in (ALBEDO, 0.0):
            wd = rb.WORK / "isotropic" / f"p{bp}_sh{sh}_h{h}_{sky}_a{alb}"
            wd.mkdir(parents=True, exist_ok=True)
            oct_ = rs.build_octree(wd, env, "scene", {
                "sky.rad": isotropic_sky(el, az, dni, dhi),
                "scene.rad": rs.scene_rad(arr, "field", max(alb, 1e-9), 0.0, 0.0)})
            front, back = rs.sensors(arr, chi, rb.X_POS)
            v = rs.irradiance(oct_, np.concatenate([front.reshape(-1, 6), back.reshape(-1, 6)]), wd, env,
                              rs.DEFAULT_RTRACE, rb.N_PROC).reshape(2, len(rb.X_POS), -1).mean(axis=(1, 2))
            e[alb] = v
        rt_ground = e[ALBEDO] - e[0.0]
        s = slopevf_isotropic(arr, grid, el, az, dni, dhi)
        p = pvlib_isotropic(arr, el, az, dni, dhi)
        rows.append(dict(beta_p=bp, alpha_sh=sh, h_bar=h, sky=sky,
                         front_ground_rt=rt_ground[0], front_ground_slopevf=s["front_nearground"],
                         front_ground_pvlib=p["front_ground"],
                         back_ground_rt=rt_ground[1], back_ground_slopevf=s["back_nearground"],
                         back_ground_pvlib=p["back_ground"]))
        print(pd.DataFrame(rows[-1:]).round(2).to_string(index=False), flush=True)
    df = pd.DataFrame(rows)
    for side in ("front", "back"):
        for m in ("slopevf", "pvlib"):
            df[f"{side}_{m}_vs_rt_%"] = 100 * (df[f"{side}_ground_{m}"] / df[f"{side}_ground_rt"] - 1)
    (HERE / "results").mkdir(exist_ok=True)
    df.to_csv(HERE / "results" / "infinite_sheds_raytrace_check.csv", index=False)
    print(df[["beta_p", "h_bar", "sky", "front_slopevf_vs_rt_%", "front_pvlib_vs_rt_%",
              "back_slopevf_vs_rt_%", "back_pvlib_vs_rt_%"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
