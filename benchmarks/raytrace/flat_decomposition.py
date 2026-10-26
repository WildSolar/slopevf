"""Decompose the deviation between slopevf and bifacial_radiance's defaults (flat ground).

Starting from the 'matched' variant of flat_bifacial_radiance.py (model assumptions, identical
sky, accurate rtrace), one bifacial_radiance default at a time is switched on:

  sky         gendaylit sky (perezlum.cal with horizon blending, gendaylit ground glow)
  reflect     module material 'black' with 1 % reflectance
  module      default module: 2 cm thick, 1 cm gaps between modules (and 'black' material)
  evaluation  bifacial_radiance's own evaluation: rtrace -i on the surface, 'high' settings
              (-ab 5 -aa .08 -ar 512 -ad 2048 -as 512, ambient caching)

and finally all of them together ('native'). Each variant is compared with slopevf for the
geometry actually built (ground disc at z = -0.01 m; module mid-plane for the 2 cm module).

Usage:  python flat_decomposition.py [geometry|sky ...]   (default: the cases with the largest
        native deviations). Results: results/flat_decomposition.csv
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import flat_bifacial_radiance as fb
import radscene as rs
import run_benchmark as rb
from slopevf import Array

DEFAULT_CASES = ["g0_p30|clear_front_low", "g0_p30|clear_behind", "g0_p30|overcast"]
VARIANTS = ["matched", "sky", "reflect", "module", "evaluation", "native"]


def run_variant(name, array, sky, variant, env):
    """Front and back irradiance at bifacial_radiance's sensor positions for one variant."""
    import bifacial_radiance as br
    wd = rb.WORK / "flat_decomp" / f"{variant}_{name}_{sky}"
    wd.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    use_gendaylit = variant in ("sky", "native")
    default_module = variant in ("module", "native")
    reflective = variant in ("reflect", "module", "native")
    own_eval = variant in ("evaluation", "native")
    try:
        demo = fb._quiet(br.RadianceObj, f"{name}_{variant}", path=str(wd))
        fb._quiet(demo.setGround, rb.ALBEDO)
        kw = {} if default_module else dict(z=1e-5, xgap=0.0)
        module = fb._quiet(demo.makeModule, name=f"mod_{variant}", x=fb.MODULE_X, y=1.0, **kw)
        fb._expand_module_file(Path(module.modulefile), non_reflective=not reflective)
        scene = fb._quiet(demo.makeScene, module, dict(
            tilt=array.beta_p, pitch=array.rho_bar, hub_height=array.h_bar, azimuth=180,
            nMods=fb.N_MODS, nRows=fb.N_ROWS))
        el, az, dni, dhi, ghi = rb.sky_inputs(sky, 0.0)
        if use_gendaylit:
            fb._quiet(demo.gendaylit2manual, dni, max(dhi, 1e-3), el, az)
            files = demo.getfilelist()
        else:
            g = rb.ALBEDO
            ground = (f"\nvoid plastic ground_mat\n0\n0\n5 {g} {g} {g} 0 0\n"
                      "\nground_mat ring groundplane\n0\n0\n8\n0 0 -.01\n0 0 1\n0 100\n")
            sky_file = Path("skies") / "slopevf_sky.rad"
            sky_file.write_text(rs.sky_rad(el, az, dni, dhi, ghi, rb.ALBEDO) + ground)
            Path("slopevf_sky.cal").write_text(rs.PEREZ_CAL)
            files = [f for f in demo.getfilelist() if not f.startswith("skies")] + [str(sky_file)]
        octfile = fb._quiet(demo.makeOct, files, octname=f"{name}_{sky}_{variant}")
        if own_eval:
            ana, front, back = fb.sensor_lines(demo, scene, 0.005)
            f, b = fb._quiet(ana.analysis, octfile, f"{name}_{sky}_{variant}", front, back, accuracy="high")
            return np.array(f["Wm2"]), np.array(b["Wm2"])
        ana, front, back = fb.sensor_lines(demo, scene, 0.001)
        rays = [np.concatenate([fb._scan_points(s) + np.array([dx, 0, 0, 0, 0, 0]) for dx in fb.X_SHIFT])
                for s in (front, back)]
        e = rs.irradiance(Path(octfile).resolve(), np.concatenate(rays), Path.cwd(), env,
                          rs.DEFAULT_RTRACE, rb.N_PROC).reshape(2, len(fb.X_SHIFT), fb.N_SENS).mean(axis=1)
        return e[0], e[1]
    finally:
        os.chdir(cwd)


def main(cases):
    env = fb._setup_env()
    geo = dict(fb.GEOMETRIES)
    cache = rb.WORK / "flat_decomp" / "cache.json"
    done = json.loads(cache.read_text()) if cache.exists() else {}
    rows = []
    for case in cases:
        name, sky = case.split("|")
        arr = geo[name]
        mask = fb.edge_mask(arr, sky)
        for variant in VARIANTS:
            key = f"{case}|{variant}"
            if key not in done:
                f, b = run_variant(name, arr, sky, variant, env)
                done[key] = [f.tolist(), b.tolist()]
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(done))
            dz = 0.01 + (0.01 if variant in ("module", "native") else 0.0)   # ground disc, module mid-plane
            a = Array(arr.beta_p, arr.beta_g, arr.rho_bar, arr.h_bar + dz, arr.gamma)
            vf = fb.slopevf_profile(a, sky, rb.ALBEDO)
            row = dict(case=case, variant=variant)
            for s, side in enumerate(("front", "back")):
                keep = ~mask[side]
                rt = np.array(done[key][s])[keep]
                row[f"{side}_rt_mean"] = rt.mean()
                row[f"{side}_diff_%"] = 100 * (vf[s][keep].mean() / rt.mean() - 1)
            rows.append(row)
            print(f"{case:28s} {variant:10s} front {row['front_diff_%']:+6.2f}%  back {row['back_diff_%']:+6.2f}%",
                  flush=True)
    df = pd.DataFrame(rows)
    rb.RESULTS.mkdir(exist_ok=True)
    df.to_csv(rb.RESULTS / "flat_decomposition.csv", index=False)
    return df


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT_CASES)
