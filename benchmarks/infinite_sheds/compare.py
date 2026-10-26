"""Comparison of slopevf with pvlib's infinite_sheds model on horizontal ground.

Both models are two-dimensional view-factor models of infinitely long rows. They are
compared with an isotropic sky, which both represent exactly (pvlib's 'haydavies' option
only moves the circumsolar part into the beam and reduces to the same case). The
components are compared separately:

  direct + sky diffuse   should agree closely (same geometry, row-to-row shading and masking)
  ground-reflected       infinite_sheds uses the ground irradiation averaged over one row
                         period (uniformly bright ground), slopevf resolves the positions of
                         the irradiated and shaded strips

Usage:  python compare.py        (results in results/)
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from pvlib.bifacial import infinite_sheds

from slopevf import Array, FarGround, Settings
from slopevf.model import ArrayModel
from slopevf.sky import make_sky_grid

HERE = Path(__file__).parent
RESULTS = HERE / "results"
ALBEDO = 0.5

TILTS = [20, 30, 45, 60, 90]
SHADING_ANGLES = [10, 20]
HEIGHTS = [0.6, 1.0, 2.0]          # 0.6: lowest height that keeps a vertical panel off the ground
# (elevation, azimuth relative to the panel azimuth (0 = sun in front), DNI, DHI)
SKIES = {
    "clear_front_low": (15.0, -30.0, 700.0, 80.0),
    "clear_front_high": (40.0, 0.0, 850.0, 100.0),
    "clear_along_rows": (35.0, 90.0, 750.0, 100.0),
    "clear_behind": (20.0, 160.0, 600.0, 80.0),
    "hazy": (20.0, -6.1, 125.9, 346.9),
    "overcast": (30.0, 0.0, 0.0, 220.0),
}


def slopevf_isotropic(array, grid, el, az, dni, dhi):
    """Panel-averaged components of slopevf for an isotropic sky plus a point-sun beam."""
    model = ArrayModel(array, FarGround(), grid, Settings(n_chi=200))
    L = np.full((grid.size, 1), dhi / grid.horizontal_weight.sum())   # exact DHI normalisation
    sun = (np.array([dni]), np.radians([el]), np.radians([az]))
    out = model.apply(L, np.array([ALBEDO]), np.array([0.0]), sun)
    return {k: float(v.mean()) for k, v in out.items()}


def pvlib_isotropic(array, el, az, dni, dhi):
    ghi = dhi + dni * np.sin(np.radians(el))
    r = infinite_sheds.get_irradiance(
        surface_tilt=array.beta_p, surface_azimuth=180.0, solar_zenith=90.0 - el,
        solar_azimuth=az + 180.0, gcr=1.0 / array.rho_bar, height=array.h_bar, pitch=array.rho_bar,
        ghi=ghi, dhi=dhi, dni=dni, albedo=ALBEDO, model="isotropic", bifaciality=1.0,
        shade_factor=0.0, transmission_factor=0.0, npoints=1000)
    f = lambda k: float(np.atleast_1d(r[k])[0])  # noqa: E731
    return dict(front_sky=f("poa_front_direct") + f("poa_front_sky_diffuse"),
                front_ground=f("poa_front_ground_diffuse"),
                back_sky=f("poa_back_direct") + f("poa_back_sky_diffuse"),
                back_ground=f("poa_back_ground_diffuse"))


def main():
    grid = make_sky_grid("medium")
    rows = []
    for bp, sh, h, (sky, (el, az, dni, dhi)) in itertools.product(TILTS, SHADING_ANGLES, HEIGHTS, SKIES.items()):
        arr = Array.from_shading_angle(bp, 0, sh, h)
        s = slopevf_isotropic(arr, grid, el, az, dni, dhi)
        p = pvlib_isotropic(arr, el, az, dni, dhi)
        row = dict(beta_p=bp, alpha_sh=sh, h_bar=h, rho_bar=arr.rho_bar, sky=sky,
                   front_sky_vf=s["front_direct"], front_sky_pvlib=p["front_sky"],
                   front_ground_vf=s["front_nearground"], front_ground_pvlib=p["front_ground"],
                   back_sky_vf=s["back_direct"], back_sky_pvlib=p["back_sky"],
                   back_ground_vf=s["back_nearground"], back_ground_pvlib=p["back_ground"])
        for side in ("front", "back"):
            tot_vf = row[f"{side}_sky_vf"] + row[f"{side}_ground_vf"]
            tot_pv = row[f"{side}_sky_pvlib"] + row[f"{side}_ground_pvlib"]
            row[f"{side}_total_diff_%"] = 100 * (tot_vf / tot_pv - 1)
            row[f"{side}_sky_diff_W"] = row[f"{side}_sky_vf"] - row[f"{side}_sky_pvlib"]
            row[f"{side}_sky_diff_%"] = 100 * (row[f"{side}_sky_vf"] / row[f"{side}_sky_pvlib"] - 1)
            row[f"{side}_ground_diff_%"] = 100 * (row[f"{side}_ground_vf"] / row[f"{side}_ground_pvlib"] - 1)
        rows.append(row)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "infinite_sheds.csv", index=False)

    lines = [f"## slopevf vs pvlib infinite_sheds, isotropic sky, horizontal ground ({len(df)} cases)",
             "deviation slopevf / pvlib - 1, panel average"]
    for side in ("front", "back"):
        for comp in ("sky", "ground", "total"):
            d = df[f"{side}_{comp}_diff_%"]
            lines.append(f"{side:5s} {comp:6s}: mean {d.mean():+.2f}%  |max| {d.abs().max():.2f}%  "
                         f"(P95 {d.abs().quantile(0.95):.2f}%)")
        lines.append(f"{side:5s} sky    |max| absolute {df[f'{side}_sky_diff_W'].abs().max():.3f} W/m2")
    g = df.groupby("h_bar")[["front_ground_diff_%", "back_ground_diff_%"]].agg(["mean", lambda x: x.abs().max()])
    lines.append("ground-reflected deviation by height [%] (mean, |max|):\n" + g.round(2).to_string())
    g = df.groupby("sky")[["back_ground_diff_%"]].agg(["mean", lambda x: x.abs().max()])
    lines.append("backside ground-reflected deviation by sky [%] (mean, |max|):\n" + g.round(2).to_string())
    text = "\n".join(lines)
    (RESULTS / "infinite_sheds.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
