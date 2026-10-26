"""Effect of the diffuse reflection of the module surfaces on full-size arrays (Davos TMY).

Five arrays (h = 1, tilt 30 - 90 deg, slope 0 - 30 deg, shading angle 15 and 25 deg) with uncoated
glass (Fresnel reflection and transmission, n = 1.52). The diffuse component is linear in the
reflectance R_d, so it is computed once for R_d = 0.1 and scaled. Reported: relative change of the
yearly (albedo 0.5) and winter half-year (Oct - Mar, albedo 0.79) irradiation per side.

Usage:  python diffuse_effect.py      (results in output/diffuse_effect.csv)
"""

import warnings
from pathlib import Path

import pandas as pd

from slopevf import Array, FarGround, PanelGlass, Settings, simulate

warnings.simplefilter("ignore")
OUT = Path(__file__).parent / "output"
TMY = OUT / "pvgis_tmy_davos_v5_3_free.csv"
ARRAYS = {  # (tilt, slope, shading angle)
    "p90_g0_s15": (90, 0, 15), "p90_g20_s15": (90, 20, 15), "p90_g30_s25": (90, 30, 25),
    "p60_g20_s15": (60, 20, 15), "p30_g0_s25": (30, 0, 25),
}
R_D = (0.03, 0.05, 0.07)


def main():
    tmy = pd.read_csv(TMY, index_col=0, parse_dates=True)
    w = tmy[["dni", "ghi"]].copy()
    winter = w.index.month.isin([10, 11, 12, 1, 2, 3])
    arrays = [Array.from_shading_angle(bp, bg, a, 1.0) for bp, bg, a in ARRAYS.values()]
    rows = []
    for season, albedo, sel in (("year", 0.5, slice(None)), ("winter", 0.79, winter)):
        kw = dict(albedo=albedo, far_ground=FarGround(20, 160), albedo_fg=0.5, latitude=46.838, longitude=9.813,
                  altitude=1560, settings=Settings(sky="low", n_chi=50))
        base = simulate(arrays, w[sel], panel_glass=PanelGlass(transmission=True), **kw)
        diff = simulate(arrays, w[sel], panel_glass=PanelGlass(transmission=True, diffuse=0.1), **kw)
        for name, b, d in zip(ARRAYS, base, diff):
            for side in ("front", "back"):
                e0 = getattr(b, side).mean(axis=1).sum()
                d1 = getattr(d, f"{side}_diffuse").mean(axis=1).sum() / 0.1
                rows.append(dict(array=name, season=season, side=side,
                                 **{f"R_d={r}": 100 * r * d1 / e0 for r in R_D}))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "diffuse_effect.csv", index=False)
    print("Relative change of the irradiation due to diffuse panel reflection [%]")
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
