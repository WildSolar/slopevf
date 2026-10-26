"""Rerun the simulation cases of the publication in legacy (MATLAB) and corrected mode.

Cases (Wild et al., section 4):
  Fig. 15  front/back distribution, beta_g=20, beta_p=60, alpha_sh=15 deg, h=0.4
  Fig. 16  back side for h = 0.4 ... 1.4
  Fig. 17  front side of four shading-angle-equivalent arrays (slopes 0-30 deg, h=1)
  Table 2  yearly / seasonal irradiation (PVGIS TMY Davos), front + 0.9 * back

Three model variants are compared:
  legacy        Settings.matlab() with the paper's numerical parameters
  corrected     Settings(sun_model="buie")   -> effect of the bug fixes alone
  default       Settings()                   -> additionally the point-sun model
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from slopevf import Array, FarGround, Settings, simulate

warnings.simplefilter("ignore")
OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)

VARIANTS = {
    # paper section 3.2: 20'672 sky cells ('medium'), 180 profile angles, 100 d/zeta intervals
    "legacy": Settings.matlab(sky="medium", n_d=100, n_zeta=100),
    "corrected": Settings(sun_model="buie"),
    "default": Settings(),
}
FAR = FarGround(20, 160)

# weather of Figs. 15-17 (annotation of Fig. 15)
FIG_WEATHER = dict(dni=[125.9], ghi=[390.0], solar_azimuth=[-6.1], solar_elevation=[20.0])
FIG_KW = dict(albedo=0.79, far_ground=FAR, albedo_fg=0.5)


def run(arrays, weather, **kw):
    return {name: simulate(arrays, weather, settings=s, **kw) for name, s in VARIANTS.items()}


def rel(a, b):
    return f"{100 * (a / b - 1):+.1f}%"


def fig11():
    arr = Array.from_shading_angle(60, 20, 15, 0.4)
    res = run(arr, FIG_WEATHER, **FIG_KW)
    rows = []
    for name, r in res.items():
        back = r.back[0]
        rows.append(dict(variant=name, front_mean=r.front.mean(), back_mean=back.mean(),
                         back_min=back.min(), back_max=back.max(),
                         back_max_over_min=f"{100 * (back.max() / back.min() - 1):.0f}%",
                         front_farground=r.front_farground.mean(), front_nearground=r.front_nearground.mean(),
                         back_nearground=r.back_nearground.mean()))
    return pd.DataFrame(rows).set_index("variant").round(1), res


def fig12():
    rows = []
    for h in (0.4, 0.6, 0.8, 1.0, 1.2, 1.4):
        arr = Array.from_shading_angle(60, 20, 15, h)
        for name, r in run(arr, FIG_WEATHER, **FIG_KW).items():
            b = r.back[0]
            rows.append(dict(h_bar=h, variant=name, back_mean=b.mean(), back_min=b.min(), back_max=b.max()))
    return pd.DataFrame(rows).pivot(index="h_bar", columns="variant").round(1)


def fig13():
    arrs = [Array.from_shading_angle(60, s, 15, 1.0) for s in (0, 10, 20, 30)]
    rows = []
    for name, rs in run(arrs, FIG_WEATHER, **FIG_KW).items():
        for s, r in zip((0, 10, 20, 30), rs):
            f = r.front[0]
            rows.append(dict(slope=s, variant=name, front_mean=f.mean(), front_min=f.min(), front_max=f.max()))
    return pd.DataFrame(rows).pivot(index="slope", columns="variant").round(1)


def table1(tmy, offset_minutes=0):
    """Table 2: cumulative irradiation [MWh/m^2], front + 0.9 * back, average and minimum over the panel."""
    w = tmy[["dni", "ghi"]].copy()
    w.index = w.index + pd.Timedelta(minutes=offset_minutes)
    winter = w.index.month.isin([10, 11, 12, 1, 2, 3])
    albedo = np.where(winter, 0.7, 0.3)
    albedo_fg = np.where(winter, 0.3, 0.2)
    arrs = [Array.from_shading_angle(60, s, 15, 1.0) for s in (0, 10, 20, 30)]
    rows = []
    for name, s in VARIANTS.items():
        rs = simulate(arrs, w, albedo=albedo, far_ground=FAR, albedo_fg=albedo_fg, settings=s,
                      latitude=46.838, longitude=9.813, altitude=1560)
        for slope, r in zip((0, 10, 20, 30), rs):
            comb = r.front + 0.9 * r.back
            avg, mn = comb.mean(axis=1) / 1e6, comb.min(axis=1) / 1e6
            rows.append(dict(slope=slope, variant=name,
                             year_avg=avg.sum(), summer_avg=avg[~winter].sum(), winter_avg=avg[winter].sum(),
                             year_min=mn.sum(), summer_min=mn[~winter].sum(), winter_min=mn[winter].sum(),
                             farground=(r.front_farground.mean(axis=1)).sum() / 1e6))
    return pd.DataFrame(rows)


def tmy_sensitivity():
    """Effect of the corrections (legacy -> corrected, Buie sun) on Table 2 for several PVGIS TMYs.

    The TMY of the first paper could not be retrieved; the TMYs currently provided by PVGIS
    differ by up to 15 % in GHI. The relative effect of the corrections is evaluated for each.
    """
    import pvlib
    cols = ["year_avg", "summer_avg", "winter_avg", "year_min", "summer_min", "winter_min"]
    variants = {k: VARIANTS[k] for k in ("legacy", "corrected")}
    out = {}
    for version, horizon in (("v5_3", True), ("v5_3", False), ("v5_2", True)):
        f = OUT / f"pvgis_tmy_davos_{version}_{'hor' if horizon else 'free'}.csv"
        if not f.exists():
            pvlib.iotools.get_pvgis_tmy(46.838, 9.813, usehorizon=horizon, map_variables=True,
                                        url=f"https://re.jrc.ec.europa.eu/api/{version}/")[0].to_csv(f)
        tmy = pd.read_csv(f, index_col=0, parse_dates=True)
        saved = dict(VARIANTS)
        VARIANTS.clear(), VARIANTS.update(variants)
        try:
            t = table1(tmy)
        finally:
            VARIANTS.clear(), VARIANTS.update(saved)
        leg = t[t.variant == "legacy"].set_index("slope")[cols]
        cor = t[t.variant == "corrected"].set_index("slope")[cols]
        out[f"{version}, horizon={horizon}, GHI {tmy.ghi.sum() / 1e3:.0f} kWh/m2"] = (100 * (cor / leg - 1)).round(2)
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    t11, res11 = fig11()
    print("=== Fig. 15 (h=0.4) ===\n", t11, "\n")
    print("=== Fig. 16 (back side vs h) ===\n", fig12(), "\n")
    print("=== Fig. 17 (front side, equal shading angle) ===\n", fig13(), "\n")

    import pvlib
    tmy = pvlib.iotools.get_pvgis_tmy(46.838, 9.813, map_variables=True)[0]
    tmy.to_csv(OUT / "pvgis_tmy_davos.csv")
    for offset in (0, 30):
        t = table1(tmy, offset)
        print(f"=== Table 2 (TMY, sun position at HH:{offset:02d}) ===")
        print(t.set_index(["variant", "slope"]).round(3).unstack("variant").T.to_string(), "\n")
    for name, r in tmy_sensitivity().items():
        print(f"=== Table 2, change legacy -> corrected [%], TMY {name} ===\n{r.to_string()}\n")
