"""Effect of the local horizon at the Totalp site (Davos) over a TMY year.

Compares, for three arrays, the yearly irradiation without horizon to the horizon with
black terrain (masking only), horizontal terrain and terrain slopes facing the array
(30 deg = default, and 60 deg). The horizon file follows the paper convention (0 = south, +west).
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from slopevf import Array, FarGround, Horizon, Settings, prepare_weather, simulate

HERE = Path(__file__).parent
SITE = dict(latitude=46.838, longitude=9.813, altitude=1560)

cache = HERE / "output" / "pvgis_tmy_davos_free_horizon.csv"
if cache.exists():
    tmy = pd.read_csv(cache, index_col=0, parse_dates=True)
else:
    import pvlib
    # usehorizon=False: free-horizon data, the horizon is applied by slopevf
    tmy = pvlib.iotools.get_pvgis_tmy(SITE["latitude"], SITE["longitude"], usehorizon=False,
                                      map_variables=True)[0]
    cache.parent.mkdir(exist_ok=True)
    tmy.to_csv(cache)
w = prepare_weather(tmy[["dni", "ghi"]], **SITE)
winter = w.index.month.isin([10, 11, 12, 1, 2, 3])
w["albedo"] = np.where(winter, 0.7, 0.3)
w["albedo_fg"] = np.where(winter, 0.3, 0.2)
w["albedo_terrain"] = np.where(winter, 0.5, 0.2)      # snow-covered mountains in winter

arrays = {"Totalp, beta_p = 60": Array(60, 30, 1.04, 2.04),
          "Totalp, beta_p = 90": Array(90, 30, 1.04, 2.04),
          "beta_p = 60, beta_g = 20": Array.from_shading_angle(60, 20, 15, 1.0)}
hor = HERE / "data" / "Totalp_horizon.hor"
variants = {
    "no horizon": dict(),
    "masked, black terrain": dict(horizon=Horizon.from_file(hor), albedo_terrain=0.0),
    "horizontal terrain": dict(horizon=Horizon.from_file(hor, terrain="horizontal")),
    "facing terrain 30": dict(horizon=Horizon.from_file(hor, terrain="facing", terrain_tilt=30)),
    "facing terrain 60": dict(horizon=Horizon.from_file(hor, terrain="facing", terrain_tilt=60)),
}

rows = []
for name, kw in variants.items():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = simulate(list(arrays.values()), w, far_ground=FarGround(20, 160), settings=Settings(sky="low"), **kw)
    for a, r in zip(arrays, res):
        rows.append(dict(array=a, variant=name, front=r.front_mean.sum() / 1e3, back=r.back_mean.sum() / 1e3))
df = pd.DataFrame(rows)
for side in ("front", "back"):
    ref = df[df.variant == "no horizon"].set_index("array")[side]
    df[f"{side} [%]"] = [100 * (v / ref[a] - 1) for a, v in zip(df["array"], df[side])]
h = Horizon.from_file(hor)
print(f"horizon: {np.degrees(h._el).mean():.1f} deg on average, max {np.degrees(h._el).max():.1f} deg")
print(df.round(2).to_string(index=False))
df.to_csv(HERE / "output" / "horizon_effect.csv", index=False)
