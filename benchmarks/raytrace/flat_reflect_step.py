"""The 'reflect' step of the bifacial_radiance decomposition, reproduced with diffuse panel reflection.

flat_decomposition.py switches bifacial_radiance's module material from perfectly black to its
default 'black' (1 % diffuse reflectance). This compares the resulting change of the ray-traced
irradiance with slopevf's diffuse panel reflection, PanelGlass(reflectance=0, diffuse=0.01), for
the same geometry (ground disc 1 cm below nominal). Requires results/flat_decomposition.csv.

Usage:  python flat_reflect_step.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import flat_bifacial_radiance as fb
import run_benchmark as rb
from slopevf import Array, FarGround, PanelGlass, Settings, simulate


def main():
    d = pd.read_csv(rb.RESULTS / "flat_decomposition.csv").set_index(["case", "variant"])
    geo = dict(fb.GEOMETRIES)
    for case in d.index.get_level_values("case").unique():
        name, sky = case.split("|")
        a0 = geo[name]
        a = Array(a0.beta_p, a0.beta_g, a0.rho_bar, a0.h_bar + 0.01)
        el, az, dni, dhi, ghi = rb.sky_inputs(sky, 0.0)
        r = simulate(a, dict(dni=[dni], ghi=[ghi], solar_azimuth=[az], solar_elevation=[el]),
                     albedo=rb.ALBEDO, far_ground=FarGround(0, 0), albedo_fg=rb.ALBEDO,
                     panel_glass=PanelGlass(reflectance=lambda c: np.zeros_like(c), diffuse=0.01),
                     settings=Settings(n_chi=200, sky="medium"))
        for side in ("front", "back"):
            # model/RT - 1 drops by the reflected share in RT: RT gain = (1 + matched)/(1 + reflect) - 1
            m, f = d.loc[(case, "matched"), f"{side}_diff_%"], d.loc[(case, "reflect"), f"{side}_diff_%"]
            rt_gain = 100 * ((1 + m / 100) / (1 + f / 100) - 1)
            model_gain = 100 * getattr(r, f"{side}_diffuse").mean() / (getattr(r, side) - getattr(r, f"{side}_diffuse")).mean()
            print(f"{case:30s} {side:5s}: RT +{rt_gain:5.2f} %, model +{model_gain:5.2f} %")


if __name__ == "__main__":
    main()
