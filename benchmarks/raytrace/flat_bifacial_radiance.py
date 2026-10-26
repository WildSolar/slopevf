"""Flat-ground cross-check of slopevf against bifacial_radiance's own scene builder.

Two variants per case:

  native   bifacial_radiance as a typical user would run it: gendaylit sky (perezlum.cal,
           with horizon blending and its ground glow), default module (2 cm thick, 1 cm gaps
           between modules), sensors 5 mm off the surface, rtrace settings 'low' and 'high'.
  matched  bifacial_radiance's scene builder, ground and sensor positions, but modules
           practically infinitely thin and without gaps, sensors 1 mm off the surface, the
           slopevf sky (see radscene.sky_rad) and the accurate rtrace settings of the
           sloped benchmark (averaged over 8 positions along the row).

slopevf is evaluated at bifacial_radiance's sensor positions chi = -1/2 + k/(N+1).

Usage:  python flat_bifacial_radiance.py           (results in results/flat_bifacial_radiance.csv)

Each finished case is cached in work/flat/cache.json; a rerun skips cached cases. To pause a
running job cleanly, create the file work/flat/STOP: the script stops before the next case.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import radscene as rs
import run_benchmark as rb
from slopevf import Array, FarGround, Settings, simulate
from slopevf.geometry import Geometry, profile_angle

N_SENS = 20
CHI = -0.5 + np.arange(1, N_SENS + 1) / (N_SENS + 1)
X_SHIFT = np.linspace(-0.8, 0.8, 8)          # positions along the centre module (row direction)
MODULE_X = 2.0                               # module length along the row [m] (H = 1 m)
N_MODS, N_ROWS = 50, 31                      # native: 100 m long rows within the 100 m ground disc
MATCHED_FIELD = dict(n_mods=100, n_rows=41, ground_radius=300.0)   # matched: as large as the sloped benchmark

GEOMETRIES = [
    ("g0_p30", Array.from_shading_angle(30, 0, 15, 1.0)),
    ("g0_p60", Array.from_shading_angle(60, 0, 15, 1.0)),
    ("g0_p90", Array.from_shading_angle(90, 0, 15, 1.0)),
    ("g0_p60_h0.5", Array.from_shading_angle(60, 0, 15, 0.5)),
    ("g0_p60_sh25", Array.from_shading_angle(60, 0, 25, 1.0)),
]


def _setup_env():
    env = rs.radiance_env()
    os.environ["PATH"], os.environ["RAYPATH"] = env["PATH"], env.get("RAYPATH", "")
    return env


def _quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def _expand_module_file(path: Path, non_reflective=False):
    """Replace the module's '!genbox ... | xform ...' command by the polygons it generates.

    bifacial_radiance's scene file arrays the module with 'xform -a', which re-runs the
    module command once per copy. On Windows every run spawns a shell (1-4 s each), i.e.
    hours for a large field. Evaluating the command once gives identical geometry.

    With ``non_reflective``, the module material 'black' (1 % reflectance in bifacial_radiance)
    is replaced by a perfectly black material, matching the model's assumption.
    """
    env = rs.radiance_env()
    txt = rs.run(["xform", str(path)], Path.cwd(), env).decode()
    if non_reflective:
        txt = ("void plastic black_ideal\n0\n0\n5 0 0 0 0 0\n\n"
               + txt.replace("\nblack polygon", "\nblack_ideal polygon"))
    path.write_text(txt)


def build_scene(name, array, wd, thin, n_mods=N_MODS, n_rows=N_ROWS):
    """bifacial_radiance RadianceObj with module and scene (horizontal ground)."""
    import bifacial_radiance as br
    demo = _quiet(br.RadianceObj, name, path=str(wd))
    _quiet(demo.setGround, rb.ALBEDO)
    kw = dict(z=1e-5, xgap=0.0) if thin else {}
    module = _quiet(demo.makeModule, name=f"mod_{'thin' if thin else 'default'}", x=MODULE_X, y=1.0, **kw)
    _expand_module_file(Path(module.modulefile), non_reflective=thin)
    scene_dict = dict(tilt=array.beta_p, pitch=array.rho_bar, hub_height=array.h_bar, azimuth=180,
                      nMods=n_mods, nRows=n_rows)
    scene = _quiet(demo.makeScene, module, scene_dict)
    return demo, scene


def sensor_lines(demo, scene, offset):
    import bifacial_radiance as br
    ana = _quiet(br.AnalysisObj)
    front, back = _quiet(ana.moduleAnalysis, scene, sensorsy=N_SENS, frontsurfaceoffset=offset,
                         backsurfaceoffset=offset)
    return ana, front, back


def _scan_points(scan):
    """(N, 6) sensor rays (position + outward surface normal) from a bifacial_radiance scan.

    bifacial_radiance traces with 'rtrace -i': the scan direction points *at* the module and
    the irradiance is evaluated on the surface that is hit. Here the irradiance is evaluated
    at the sensor point itself ('rtrace -I'), so the direction is reversed to the outward normal.
    """
    k = np.arange(scan["Ny"])
    pos = np.stack([scan["xstart"] + k * scan["xinc"], scan["ystart"] + k * scan["yinc"],
                    scan["zstart"] + k * scan["zinc"]], axis=-1)
    d = -np.array([float(v) for v in scan["orient"].split()])
    return np.concatenate([pos, np.broadcast_to(d, pos.shape)], axis=-1)


def run_native(name, array, sky, accuracy):
    wd = rb.WORK / "flat" / f"native_{name}_{sky}"
    wd.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    try:
        demo, scene = build_scene(name, array, wd, thin=False)
        el, az, dni, dhi, ghi = rb.sky_inputs(sky, 0.0)
        _quiet(demo.gendaylit2manual, dni, max(dhi, 1e-3), el, az)
        octfile = _quiet(demo.makeOct, demo.getfilelist())
        ana, front, back = sensor_lines(demo, scene, 0.005)
        f, b = _quiet(ana.analysis, octfile, f"{name}_{sky}_{accuracy}", front, back, accuracy=accuracy)
        return np.array(f["Wm2"]), np.array(b["Wm2"])
    finally:
        os.chdir(cwd)


def run_matched(name, array, sky, env):
    wd = rb.WORK / "flat" / f"matched_{name}_{sky}"
    wd.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    try:
        demo, scene = build_scene(name, array, wd, thin=True, n_mods=MATCHED_FIELD["n_mods"],
                                  n_rows=MATCHED_FIELD["n_rows"])
        el, az, dni, dhi, ghi = rb.sky_inputs(sky, 0.0)
        # slopevf sky + bifacial_radiance's ground material and ground disc
        g = rb.ALBEDO
        ground = (f"\nvoid plastic ground_mat\n0\n0\n5 {g} {g} {g} 0 0\n"
                  f"\nground_mat ring groundplane\n0\n0\n8\n0 0 -.01\n0 0 1\n0 {MATCHED_FIELD['ground_radius']}\n")
        sky_file = Path("skies") / "slopevf_sky.rad"
        sky_file.write_text(rs.sky_rad(el, az, dni, dhi, ghi, rb.ALBEDO_FG_FLAT) + ground)
        Path("slopevf_sky.cal").write_text(rs.PEREZ_CAL)
        files = [f for f in demo.getfilelist() if not f.startswith("skies")] + [str(sky_file)]
        octfile = _quiet(demo.makeOct, files, octname=f"{name}_{sky}_matched")
        ana, front, back = sensor_lines(demo, scene, 0.001)
        rays = []
        for scan in (front, back):
            pts = _scan_points(scan)
            shifted = [pts + np.array([dx, 0, 0, 0, 0, 0]) for dx in X_SHIFT]   # row runs along x
            rays.append(np.concatenate(shifted))
        e = rs.irradiance(Path(octfile).resolve(), np.concatenate(rays), Path.cwd(), env,
                          rs.DEFAULT_RTRACE, rb.N_PROC)
        e = e.reshape(2, len(X_SHIFT), N_SENS).mean(axis=1)
        return e[0], e[1]
    finally:
        os.chdir(cwd)


def slopevf_profile(array, sky, albedo_fg):
    el, az, dni, dhi, ghi = rb.sky_inputs(sky, 0.0)
    n = 21 * 50
    res = simulate(array, dict(dni=[dni], ghi=[ghi], solar_azimuth=[az], solar_elevation=[el]),
                   albedo=rb.ALBEDO, far_ground=FarGround(0, 0), albedo_fg=albedo_fg,
                   settings=Settings(n_chi=n, sky="medium"))
    return np.interp(CHI, res.chi, res.front[0]), np.interp(CHI, res.chi, res.back[0])


def edge_mask(array, sky, tol=0.01):
    g = Geometry(array)
    el, az, dni, *_ = rb.sky_inputs(sky, 0.0)
    out = {"front": np.zeros(N_SENS, bool), "back": np.zeros(N_SENS, bool)}
    if dni <= 0:
        return out
    a = profile_angle(np.radians(el), np.radians(az), 0.0)
    a1, _, a3, a4 = g.alpha
    if a < a1:
        out["front"] = np.abs(CHI + 0.5 - g.shaded_fraction_front(a)) < tol
    elif a3 <= a < a4:
        out["back"] = np.abs(CHI + 0.5 - g.shaded_fraction_back(a)) < tol
    return out


def main():
    env = _setup_env()
    rows = []
    cache = rb.WORK / "flat" / "cache.json"
    done = json.loads(cache.read_text()) if cache.exists() else {}
    stop_file = rb.WORK / "flat" / "STOP"
    for name, arr in GEOMETRIES:
        for sky in rb.SKIES:
            key = f"{name}|{sky}"
            if key not in done and stop_file.exists():
                stop_file.unlink()
                print(f"STOP file found: paused before {key} ({len(done)} cases cached; rerun to resume)")
                return
            if key not in done:
                t0 = time.perf_counter()
                out = {}
                for acc in ("low", "high"):
                    f, b = run_native(name, arr, sky, acc)
                    out[f"native_{acc}"] = [f.tolist(), b.tolist()]
                f, b = run_matched(name, arr, sky, env)
                out["matched"] = [f.tolist(), b.tolist()]
                out["seconds"] = time.perf_counter() - t0
                done[key] = out
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(done))
            out = done[key]
            mask = edge_mask(arr, sky)
            # slopevf for the geometry bifacial_radiance actually builds: the ground disc lies
            # 1 cm below z = 0, and the default 2 cm module sits on top of its axis (hub height)
            native_arr = Array(arr.beta_p, arr.beta_g, arr.rho_bar, arr.h_bar + 0.01 + 0.01, arr.gamma)
            matched_arr = Array(arr.beta_p, arr.beta_g, arr.rho_bar, arr.h_bar + 0.01, arr.gamma)
            vf = {"native": slopevf_profile(native_arr, sky, rb.ALBEDO),     # ground glow ~ ground albedo
                  "matched": slopevf_profile(matched_arr, sky, rb.ALBEDO_FG_FLAT)}
            row = dict(geometry=name, sky=sky, rho_bar=arr.rho_bar, beta_p=arr.beta_p, h_bar=arr.h_bar)
            for variant in ("native_low", "native_high", "matched"):
                ref = vf["matched" if variant == "matched" else "native"]
                for s, side in enumerate(("front", "back")):
                    keep = ~mask[side]
                    a, v = np.array(out[variant][s])[keep], ref[s][keep]
                    row[f"{variant}_{side}_diff_%"] = 100 * (v.mean() / a.mean() - 1)
                    row[f"{variant}_{side}_nrmse_%"] = 100 * np.sqrt(np.mean((v - a) ** 2)) / a.mean()
            rows.append(row)
            print(f"{name:12s} {sky:17s} " + "  ".join(
                f"{v}: F {row[f'{v}_front_diff_%']:+6.2f}% B {row[f'{v}_back_diff_%']:+6.2f}%"
                for v in ("native_low", "native_high", "matched")), flush=True)
    df = pd.DataFrame(rows)
    rb.RESULTS.mkdir(exist_ok=True)
    df.to_csv(rb.RESULTS / "flat_bifacial_radiance.csv", index=False)
    lines = ["## Flat ground: slopevf vs bifacial_radiance (panel-mean deviation slopevf/RT - 1)"]
    for v in ("native_low", "native_high", "matched"):
        for side in ("front", "back"):
            d = df[f"{v}_{side}_diff_%"]
            lines.append(f"{v:12s} {side:5s}: mean {d.mean():+.2f}%  |max| {d.abs().max():.2f}%  "
                         f"NRMSE mean {df[f'{v}_{side}_nrmse_%'].mean():.2f}%")
    text = "\n".join(lines)
    (rb.RESULTS / "flat_bifacial_radiance.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
