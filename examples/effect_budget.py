"""Effect of the model extensions on the yearly irradiation of full-size arrays (Davos TMY).

Five arrays (h = 1, tilt 30 - 90 deg, ground slope 0 - 30 deg, shading angle 15 and 25 deg) are
simulated with the base model (infinite rows, infinitely thin and non-reflective panels) and with
each extension switched on individually:

  three rows, centre row                 finite_rows = (1, 1)
  three rows, outermost row              frontside of the uphill row (2, 0), backside of the downhill row (0, 2)
  panel thickness 0.016 / 0.03           framed module of 2.2 m row height with 35 mm frame / with mounting rails
  specular reflection                    Fresnel glass, n = 1.52 (no transmission)
  transmission                           incidence angle modifier of uncoated glass (no reflection between rows)
  diffuse reflection, R_d = 0.05         single reflection between facing panels

Albedo 0.5, far ground = continuing slope (albedo 0.5), PVGIS TMY 5.3 free horizon. Reported: change of the
yearly panel-averaged irradiation per side relative to the base model (range over the five arrays), and for
the thickness the largest reduction of single hours (base backside irradiation above 20 W/m^2).

Usage:  python effect_budget.py      (results/effect_budget.csv, printed LaTeX table rows)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from slopevf import Array, FarGround, PanelGlass, Settings, simulate

warnings.simplefilter("ignore")
HERE = Path(__file__).parent
RESULTS = HERE / "results"
TMY = HERE / "output" / "pvgis_tmy_davos_v5_3_free.csv"
LAT, LON, ALT = 46.838, 9.813, 1560.0
ALBEDO = 0.5
ARRAYS = {  # (tilt, slope, shading angle)
    "p90_g0_s15": (90, 0, 15), "p90_g20_s15": (90, 20, 15), "p90_g30_s25": (90, 30, 25),
    "p60_g20_s15": (60, 20, 15), "p30_g0_s25": (30, 0, 25),
}
NO_REFL = lambda c: np.zeros_like(c)  # noqa: E731
BASE = Settings(sky="low", n_chi=50)


def run(arr, w, settings=BASE, glass=None):
    return simulate(arr, w, albedo=ALBEDO, far_ground=FarGround(arr.beta_g, 0.0), albedo_fg=ALBEDO,
                    panel_glass=glass, settings=settings, latitude=LAT, longitude=LON, altitude=ALT)


def main():
    w = pd.read_csv(TMY, index_col=0, parse_dates=True)[["dni", "ghi"]]
    rows = []
    for name, (bp, bg, sh) in ARRAYS.items():
        arr = Array.from_shading_angle(bp, bg, sh, 1.0)
        thick = lambda t: Array.from_shading_angle(bp, bg, sh, 1.0).__class__(  # noqa: E731
            arr.beta_p, arr.beta_g, arr.rho_bar, arr.h_bar, arr.gamma, thickness=t)
        base = run(arr, w)
        e0 = {s: getattr(base, f"{s}_mean").sum() for s in ("front", "back")}
        cases = {
            "three rows, centre row": run(arr, w, Settings(sky="low", n_chi=50, finite_rows=(1, 1))),
            "panel thickness 0.016": run(thick(0.016), w),
            "panel thickness 0.03": run(thick(0.03), w),
            "specular reflection": run(arr, w, glass=PanelGlass(n=1.52)),
            "transmission": run(arr, w, glass=PanelGlass(n=1.52, front=False, back=False, transmission=True)),
            "diffuse reflection": run(arr, w, glass=PanelGlass(reflectance=NO_REFL, diffuse=0.05)),
        }
        outer = {"front": run(arr, w, Settings(sky="low", n_chi=50, finite_rows=(2, 0))),
                 "back": run(arr, w, Settings(sky="low", n_chi=50, finite_rows=(0, 2)))}
        for case, r in cases.items():
            for s in ("front", "back"):
                rows.append(dict(array=name, effect=case, side=s,
                                 change=100 * (getattr(r, f"{s}_mean").sum() / e0[s] - 1)))
        for s in ("front", "back"):
            rows.append(dict(array=name, effect="three rows, outermost row", side=s,
                             change=100 * (getattr(outer[s], f"{s}_mean").sum() / e0[s] - 1)))
        # single hours: largest reduction by the thickness (base backside > 20 W/m2)
        sel = base.back_mean.to_numpy() > 20
        for t, key in ((0.016, "panel thickness 0.016"), (0.03, "panel thickness 0.03")):
            r = cases[key]
            red = 100 * (1 - r.back_mean.to_numpy()[sel] / base.back_mean.to_numpy()[sel])
            rows.append(dict(array=name, effect=key, side="back_hour_max_reduction", change=red.max()))
        print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "effect_budget.csv", index=False)
    order = ["three rows, centre row", "three rows, outermost row", "panel thickness 0.016",
             "panel thickness 0.03", "specular reflection", "transmission", "diffuse reflection"]
    print("\nChange of the yearly irradiation [%], range over the five arrays")
    for e in order:
        cells = []
        for s in ("front", "back"):
            v = df[(df.effect == e) & (df.side == s)].change
            cells.append(f"{v.min():+.1f} -- {v.max():+.1f}")
        print(f"{e:28s} front {cells[0]:>16s}   back {cells[1]:>16s}")
    for key in ("panel thickness 0.016", "panel thickness 0.03"):
        v = df[(df.effect == key) & (df.side == "back_hour_max_reduction")].change
        print(f"{key}: largest single-hour backside reduction {v.min():.0f} -- {v.max():.0f} %")


if __name__ == "__main__":
    main()
