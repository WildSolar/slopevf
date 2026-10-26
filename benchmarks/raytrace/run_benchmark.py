"""Benchmark slopevf against Radiance ray tracing on sloped terrain.

Usage (Radiance on PATH or $RADIANCE_DIR set):

    python run_benchmark.py skycheck             # step 0: sky model only (no array)
    python run_benchmark.py pilot                # few cases, convergence checks
    python run_benchmark.py full --level ideal   # full matrix
    python run_benchmark.py report               # summary tables and figures from cached results

Ray-tracing results are cached per case in work/ (delete to recompute).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from slopevf import Array, FarGround, Settings, simulate
from slopevf.geometry import direction_vector, surface_normal
from slopevf.sky import make_sky_grid, sky_radiance

import radscene as rs

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
import pubstyle as ps  # noqa: E402
WORK = HERE / "work"
RESULTS = HERE / "results"

ALBEDO = 0.5
ALBEDO_FG = 0.3
ALBEDO_FG_FLAT = 0.5                         # flat ground: far ground = ground beyond the field
N_SENS = 20                                  # sensors along the panel height
CHI = -0.5 + (np.arange(N_SENS) + 0.5) / N_SENS
X_POS = list(np.linspace(-4.0, 4.0, 8))      # sensor positions along the row (averaged)
N_PROC = max(1, (os.cpu_count() or 2) - 1)

# --- test matrix --------------------------------------------------------------------------

def geometries():
    """(name, Array) pairs. Default: shading angle 15 deg, h_bar = 1, south-facing slope."""
    out = []
    for bg, bp in [(0, 30), (0, 60), (0, 90), (10, 30), (10, 60), (10, 90), (20, 30), (20, 60),
                   (20, 90), (30, 60), (30, 90)]:
        out.append((f"g{bg}_p{bp}", Array.from_shading_angle(bp, bg, 15, 1.0)))
    out += [
        ("g20_p60_h0.5", Array.from_shading_angle(60, 20, 15, 0.5)),
        ("g20_p60_h1.5", Array.from_shading_angle(60, 20, 15, 1.5)),
        ("g20_p60_sh5", Array.from_shading_angle(60, 20, 5, 1.0)),
        ("g20_p60_sh25", Array.from_shading_angle(60, 20, 25, 1.0)),
        ("g20_p60_az45", Array.from_shading_angle(60, 20, 15, 1.0, gamma=45)),
        ("g30_p90_az-45", Array.from_shading_angle(90, 30, 15, 1.0, gamma=-45)),
    ]
    return out


# sun azimuth relative to the slope azimuth (0 = sun in the downhill direction)
SKIES = {
    "clear_front_low": dict(el=15.0, az_rel=-30.0, dni=750.0, dhi=80.0),
    "clear_front_high": dict(el=55.0, az_rel=15.0, dni=850.0, dhi=110.0),
    "clear_along_rows": dict(el=35.0, az_rel=90.0, dni=750.0, dhi=100.0),
    "clear_behind": dict(el=20.0, az_rel=160.0, dni=650.0, dhi=80.0),
    "hazy": dict(el=20.0, az_rel=-6.1, dni=125.9, dhi=346.9),    # Fig. 15 of the paper
    "overcast": dict(el=30.0, az_rel=0.0, dni=0.0, dhi=220.0),
}


def sky_inputs(sky, gamma):
    s = SKIES[sky]
    az = gamma + s["az_rel"]
    az = (az + 180) % 360 - 180
    ghi = s["dhi"] + s["dni"] * np.sin(np.radians(s["el"]))
    return s["el"], az, s["dni"], s["dhi"], ghi


# --- step 0: sky only ------------------------------------------------------------------------

def skycheck():
    """Plane-of-array irradiance of unobstructed planes: gendaylit/rtrace vs the slopevf sky."""
    env = rs.radiance_env()
    wd = WORK / "skycheck"
    wd.mkdir(parents=True, exist_ok=True)
    planes = [(0, 0)] + [(t, a) for t in (30, 60, 90) for a in (0, 90, -90, 180)]
    grid = make_sky_grid("medium")
    rows = []
    for sky in SKIES:
        el, az, dni, dhi, ghi = sky_inputs(sky, 0.0)
        oct_ = rs.build_octree(wd, env, sky, {f"{sky}_sky.rad": rs.sky_rad(el, az, dni, dhi, ghi, 0.0)})
        rays = []
        for t, a in planes:
            n = surface_normal(np.radians(t), np.radians(a))
            rays.append([0, 0, 0, -n[0], -n[1], n[2]])   # Radiance: x = east, y = north
        e_rt = rs.irradiance(oct_, rays, wd, env, ["-ab", "1", "-aa", "0", "-ad", "65536", "-as", "0"])
        L = sky_radiance(grid, [dni], [ghi], np.radians([az]), np.radians([el]), sun_model="point").L[:, 0]
        sun = direction_vector(np.radians(el), np.radians(az))
        for (t, a), e in zip(planes, e_rt):
            n = surface_normal(np.radians(t), np.radians(a))
            e_vf = (grid.solid_angle * np.maximum(grid.vec @ n, 0)) @ L + dni * max(sun @ n, 0)
            rows.append(dict(sky=sky, tilt=t, azimuth=a, radiance=e, slopevf=e_vf))
    df = pd.DataFrame(rows)
    df["rel_diff_%"] = 100 * (df.slopevf / df.radiance - 1)
    RESULTS.mkdir(exist_ok=True)
    df.round(2).to_csv(RESULTS / "skycheck.csv", index=False)
    print(df.round(2).to_string(index=False))
    return df


# --- array cases -----------------------------------------------------------------------------

def rt_case(name, array, sky, level, size=rs.FieldSize(), params=rs.DEFAULT_RTRACE, tag=""):
    """Ray-traced front/back irradiance at the sensor positions (cached)."""
    key = f"{level}_{name}_{sky}{tag}"
    cache = WORK / "cases" / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    env = rs.radiance_env()
    wd = WORK / "scenes" / key
    wd.mkdir(parents=True, exist_ok=True)
    el, az, dni, dhi, ghi = sky_inputs(sky, array.gamma)
    t0 = time.perf_counter()
    oct_ = rs.build_octree(wd, env, "scene", {
        "sky.rad": rs.sky_rad(el, az, dni, dhi, ghi, ALBEDO_FG),
        "scene.rad": rs.scene_rad(array, level, ALBEDO, ghi, ALBEDO_FG, size)})
    front, back = rs.sensors(array, CHI, X_POS)
    e = rs.irradiance(oct_, np.concatenate([front.reshape(-1, 6), back.reshape(-1, 6)]), wd, env, params, N_PROC)
    e = e.reshape(2, len(X_POS), N_SENS)
    out = dict(front=e[0].mean(0).tolist(), back=e[1].mean(0).tolist(),
               front_std=e[0].std(0).tolist(), back_std=e[1].std(0).tolist(),
               seconds=time.perf_counter() - t0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def vf_case(array, sky, level):
    """slopevf irradiance at the sensor positions (element centres coincide with the sensors)."""
    el, az, dni, dhi, ghi = sky_inputs(sky, array.gamma)
    k = 15                                            # n_chi = 15 * N_SENS -> sensor = element centre
    res = simulate(array, dict(dni=[dni], ghi=[ghi], solar_azimuth=[az], solar_elevation=[el]),
                   albedo=ALBEDO, far_ground=FarGround(0, 0), albedo_fg=ALBEDO_FG,
                   settings=Settings(n_chi=k * N_SENS, sky="medium"))
    idx = k * np.arange(N_SENS) + k // 2
    assert np.allclose(res.chi[idx], CHI)
    return dict(front=res.front[0, idx], back=res.back[0, idx],
                front_farground=res.front_farground[0, idx])


EDGE_TOL = 0.01   # sensors closer than this (in chi) to a direct shadow edge are excluded


def shadow_edge_mask(array, sky):
    """Sensors that lie on the edge of the direct row-to-row shadow (front or back).

    There, a point sample (ray tracing, finite sun disc) and an element average
    (slopevf) are not comparable. Returns a dict side -> boolean mask of excluded sensors.
    """
    from slopevf.geometry import Geometry, profile_angle
    g = Geometry(array)
    el, az, dni, *_ = sky_inputs(sky, array.gamma)
    out = {"front": np.zeros(N_SENS, bool), "back": np.zeros(N_SENS, bool)}
    if dni <= 0:
        return out
    a = profile_angle(np.radians(el), np.radians(az), np.radians(array.gamma))
    a1, a2, a3, a4 = g.alpha
    if a < a1:
        out["front"] = np.abs(CHI + 0.5 - g.shaded_fraction_front(a)) < EDGE_TOL
    elif a3 <= a < a4:
        out["back"] = np.abs(CHI + 0.5 - g.shaded_fraction_back(a)) < EDGE_TOL
    return out


def compare(name, array, sky, level, **kw):
    rt = rt_case(name, array, sky, level, **kw)
    vf = vf_case(array, sky, level)
    row = dict(geometry=name, sky=sky, level=level, beta_g=array.beta_g, beta_p=array.beta_p,
               rho_bar=array.rho_bar, h_bar=array.h_bar, gamma=array.gamma, rt_seconds=rt["seconds"])
    excl = shadow_edge_mask(array, sky)
    for side in ("front", "back"):
        keep = ~excl[side]
        row[f"{side}_excluded_sensors"] = int((~keep).sum())
        a, b = np.array(rt[side])[keep], np.array(vf[side])[keep]
        row[f"{side}_rt_mean"] = a.mean()
        row[f"{side}_vf_mean"] = b.mean()
        row[f"{side}_mean_diff_%"] = 100 * (b.mean() / a.mean() - 1)
        row[f"{side}_nrmse_%"] = 100 * np.sqrt(np.mean((b - a) ** 2)) / a.mean()
        row[f"{side}_rt_noise_%"] = 100 * np.mean(np.array(rt[f"{side}_std"])[keep]) / a.mean() / np.sqrt(len(X_POS))
    profile = pd.DataFrame(dict(chi=CHI, rt_front=rt["front"], vf_front=vf["front"],
                                rt_back=rt["back"], vf_back=vf["back"], vf_front_farground=vf["front_farground"]))
    return row, profile


def run_matrix(cases, level, tag="", **kw):
    rows, profiles = [], {}
    for i, ((name, arr), sky) in enumerate(cases):
        t0 = time.perf_counter()
        row, prof = compare(name, arr, sky, level, tag=tag, **kw)
        rows.append(row)
        profiles[(name, sky)] = prof
        print(f"[{i + 1}/{len(cases)}] {level} {name:14s} {sky:17s} "
              f"front {row['front_mean_diff_%']:+6.2f}% (nrmse {row['front_nrmse_%']:5.2f}%)  "
              f"back {row['back_mean_diff_%']:+6.2f}% (nrmse {row['back_nrmse_%']:5.2f}%)  "
              f"[{time.perf_counter() - t0:5.1f} s]", flush=True)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / f"summary_{level}{tag}.csv", index=False)
    pd.concat({f"{n}|{s}": p for (n, s), p in profiles.items()}, names=["case"]).to_csv(
        RESULTS / f"profiles_{level}{tag}.csv")
    return df


def pilot():
    """Convergence of ray-tracing settings and field size on two demanding cases."""
    geo = dict(geometries())
    cases = [("g20_p60_h0.5", "hazy"), ("g30_p90", "clear_front_low")]
    variants = {
        "": {},
        "_ad8192": dict(params=["-ab", "2", "-aa", "0", "-ad", "8192", "-as", "0", "-lw", "1e-6", "-lr", "8"]),
        "_ab4": dict(params=["-ab", "4", "-aa", "0", "-ad", "2048", "-as", "0", "-lw", "1e-5", "-lr", "8"]),
        "_large": dict(size=rs.FieldSize(rows_extent=120, row_half_length=200, ground_margin=200)),
    }
    for tag, kw in variants.items():
        print(f"--- pilot variant '{tag or 'default'}'")
        run_matrix([((n, geo[n]), s) for n, s in cases], "ideal", tag=f"_pilot{tag}", **kw)


def full(level):
    cases = list(itertools.product(geometries(), SKIES))
    return run_matrix(cases, level)


def report():
    """Summary statistics and figures from results/summary_*.csv and profiles_*.csv."""
    import matplotlib.pyplot as plt

    lines = []
    levels = {lv: pd.read_csv(RESULTS / f"summary_{lv}.csv") for lv in ("ideal", "field")
              if (RESULTS / f"summary_{lv}.csv").exists()}
    for lv, df in levels.items():
        lines.append(f"## level '{lv}' ({len(df)} cases)")
        for side in ("front", "back"):
            d, n = df[f"{side}_mean_diff_%"], df[f"{side}_nrmse_%"]
            lines.append(f"{side:5s}: panel-mean diff  mean {d.mean():+.2f}%  |max| {d.abs().max():.2f}%  "
                         f"(P95 {d.abs().quantile(0.95):.2f}%);  profile NRMSE  mean {n.mean():.2f}%  "
                         f"max {n.max():.2f}%;  RT noise ~{df[f'{side}_rt_noise_%'].mean():.2f}%")
        lines.append(f"sensors excluded at direct shadow edges: front {df.front_excluded_sensors.sum()}, "
                     f"back {df.back_excluded_sensors.sum()} (of {len(df) * N_SENS} per side)")
        worst = df.reindex(df[["front_mean_diff_%", "back_mean_diff_%"]].abs().max(axis=1)
                           .sort_values(ascending=False).index).head(5)
        lines.append("largest deviations:\n" + worst[["geometry", "sky", "front_mean_diff_%", "back_mean_diff_%",
                                                       "front_nrmse_%", "back_nrmse_%"]].round(2).to_string(index=False))
        by_sky = df.groupby("sky")[["front_mean_diff_%", "back_mean_diff_%"]].agg(lambda x: x.abs().max())
        lines.append("max |panel-mean diff| by sky [%]:\n" + by_sky.round(2).to_string())
        lines.append("")
    if len(levels) == 2:
        a, b = levels["ideal"].set_index(["geometry", "sky"]), levels["field"].set_index(["geometry", "sky"])
        eff = 100 * (b["front_rt_mean"] / a["front_rt_mean"] - 1)
        lines.append("## far-ground idealisation: front-side RT mean, actual field vs uniform far ground")
        lines.append(f"mean {eff.mean():+.2f}%, range {eff.min():+.2f}% .. {eff.max():+.2f}%")
        lines.append((eff.unstack("sky").round(2)).to_string())
    text = "\n".join(lines)
    (RESULTS / "report.txt").write_text(text)
    print(text)

    if "ideal" in levels:
        figure_deviation(levels["ideal"])
        figure_profiles(pd.read_csv(RESULTS / "profiles_ideal.csv"))


# --- figures (publication style, see ../pubstyle.py) ------------------------------------------

def figure_deviation(df):
    """Relative deviation of the panel-mean irradiance, model vs. ray tracing (single format)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ps.apply_style()
    fig, axes = plt.subplots(2, 1, figsize=ps.SINGLE, sharex=True)
    for ax, side, title in zip(axes, ("front", "back"), ("(a) Front side", "(b) Back side")):
        ax.axhline(0.0, color=ps.MUTED, lw=0.6, zorder=1)
        for (bp, bg), g in df.groupby(["beta_p", "beta_g"]):
            ax.scatter(g[f"{side}_rt_mean"], g[f"{side}_mean_diff_%"], marker=ps.MARKER_TILT[bp], s=14,
                       color=ps.slope_color(bg), edgecolors="none", zorder=3)
        lim = 1.15 * max(0.2, df[f"{side}_mean_diff_%"].abs().max())
        ax.set(ylim=(-lim, lim), title=title, ylabel="Deviation [%]")
    axes[1].set(xlim=(0, 1050), xlabel="Panel-mean irradiance, ray tracing [W/m²]")
    slope_handles = [Line2D([], [], ls="none", marker="o", ms=4, color=ps.slope_color(b), label=f"{b}°")
                     for b in ps.SLOPES]
    tilt_handles = [Line2D([], [], ls="none", marker=ps.MARKER_TILT[t], ms=4, mfc="none", mec=ps.INK,
                           label=f"{t}°") for t in sorted(df.beta_p.unique())]
    axes[0].legend(handles=slope_handles, title="Ground slope", ncol=4, loc="lower right",
                   columnspacing=0.8, handletextpad=0.2, borderaxespad=0.3)
    axes[1].legend(handles=tilt_handles, title="Panel tilt", ncol=3, loc="lower right",
                   columnspacing=0.8, handletextpad=0.2, borderaxespad=0.3)
    ps.footnote(fig, f"{len(df)} scenes (17 geometries x 6 skies), scene variant 'ideal'.\n"
                     "Deviation = (slopevf - Radiance) / Radiance of the panel mean.", y=-0.01)
    ps.save(fig, RESULTS / "deviation_ideal")


PROFILE_CASES = [  # (geometry, sky, panel title, geometry annotation, its x position [axes fraction])
    ("g20_p60_h0.5", "hazy", "(a) Hazy", "$\\beta_p$ = 60°, $\\beta_g$ = 20°\n$\\bar{h}$ = 0.5", 0.48),
    ("g30_p90", "clear_front_low", "(b) Clear, low sun in front", "$\\beta_p$ = 90°, $\\beta_g$ = 30°", 0.45),
    ("g20_p60_az45", "clear_behind", "(c) Clear, sun behind",
     "$\\beta_p$ = 60°, $\\beta_g$ = 20°\n$\\gamma$ = 45°", 0.38),
    ("g0_p30", "overcast", "(d) Overcast", "$\\beta_p$ = 30°, $\\beta_g$ = 0°", 0.5),
]


def figure_profiles(prof):
    """Irradiance along the panel height, model vs. ray tracing (double-wide format)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ps.apply_style()
    fig, axes = plt.subplots(1, len(PROFILE_CASES), figsize=ps.DOUBLE, sharey=True)
    for ax, (geo, sky, title, note, x_note) in zip(axes, PROFILE_CASES):
        p = prof[prof.case == f"{geo}|{sky}"]
        for side, sign in (("front", 1.0), ("back", -1.0)):
            ax.plot(sign * p[f"vf_{side}"], p.chi, ls=ps.LINESTYLE["model"], color=ps.SIDE[side], zorder=3)
            ax.plot(sign * p[f"rt_{side}"], p.chi, ls="none", marker="o", ms=3, mfc="none", mew=0.6,
                    mec=ps.SIDE[side], zorder=4)
        ax.axvline(0.0, color=ps.MUTED, lw=0.6, zorder=1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{abs(x):g}"))
        ax.set_title(title)
        # geometry in an empty stretch near the bottom (position chosen per panel)
        ax.text(x_note, 0.06, note, transform=ax.transAxes, fontsize=ps.FS_NOTE, ha="center",
                va="bottom", zorder=5, bbox={"boxstyle": "square,pad=0.15", "fc": ps.SURFACE, "ec": "none"})
    axes[0].set(ylim=(-0.5, 0.5), ylabel=r"Position on the panel $\chi$ [-]")
    fig.supxlabel("Irradiance [W/m²]", fontsize=ps.FS_LABEL)
    side_handles = [Line2D([], [], color=ps.SIDE["back"], label="Back side (left)"),
                    Line2D([], [], color=ps.SIDE["front"], label="Front side (right)")]
    src_handles = [Line2D([], [], color=ps.INK, ls=ps.LINESTYLE["model"], label="Model"),
                   Line2D([], [], color=ps.INK, ls="none", marker="o", ms=3, mfc="none", mew=0.6,
                          label="Ray tracing")]
    axes[0].legend(handles=side_handles, loc="upper center", bbox_to_anchor=(0.5, 0.8))
    axes[1].legend(handles=src_handles, loc="upper center", bbox_to_anchor=(0.5, 0.8))
    ps.footnote(fig, "Back-side irradiance plotted to the left of zero. Scene variant 'ideal', "
                     "20 evaluation points per side; chi = -0.5 lower, +0.5 upper panel edge.", y=-0.02)
    ps.save(fig, RESULTS / "profiles_ideal")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["skycheck", "pilot", "full", "report"])
    ap.add_argument("--level", default="ideal", choices=["ideal", "field"])
    a = ap.parse_args()
    pd.set_option("display.width", 200)
    if a.what == "skycheck":
        skycheck()
    elif a.what == "pilot":
        pilot()
    elif a.what == "report":
        report()
    else:
        full(a.level)
