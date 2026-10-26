"""Electrical mismatch due to the irradiance distribution on the panel (application showcase).

Four arrays at Davos (PVGIS TMY, free horizon; panel tilt 60 / 90 deg on horizontal ground and a
30 deg slope; shading angle 15 deg, h = 1; seasonal albedo as in table 2 of the first paper) are
simulated with slopevf (uncoated glass: Fresnel reflection and transmission). The effective
irradiance of each cell, front + bifaciality * back averaged over the cell's height band, is fed
into pvmismatch for one module per row height:

  portrait     72 cells, 12 rows along the panel height, 3 substrings of 2 columns (each
               substring spans the full height)
  half-cut     144 half-cells in portrait: per substring an upper and a lower branch in parallel
  landscape    72 cells in landscape: the 6 columns lie along the height, so each substring
               covers one third of the height

All modules of a row receive the same irradiance (2D model), so a string along the row has the
mismatch of its modules. Reference: the same module irradiated uniformly with the panel average
(what a model resolving only the panel average assumes). Also reported: the worst-case
assumption of the first paper, all cells at the panel minimum. Cell temperature 25 deg C.

Usage:  python mismatch.py      (results in results/mismatch.csv, output/mismatch_profiles.npz)
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pvmismatch import pvcell, pvconstants, pvmodule

from slopevf import Array, FarGround, PanelGlass, Settings, simulate

warnings.simplefilter("ignore")
OUT = Path(__file__).parent / "output"          # downloads and large intermediate files (not tracked)
RESULTS = Path(__file__).parent / "results"
TMY = OUT / "pvgis_tmy_davos_v5_3_free.csv"
LAT, LON, ALT = 46.838, 9.813, 1560.0
BIFACIALITY = 0.9
N_CHI = 120                                   # multiple of 6, 12 and 24 cell bands
ARRAYS = {"p60_g0": (60, 0), "p60_g30": (60, 30), "p90_g0": (90, 0), "p90_g30": (90, 30)}
GLASS = PanelGlass(n=1.52, transmission=True)


def _full_cell_module():
    return pvmodule.PVmodule(cell_pos=pvmodule.STD72, pvconst=pvconstants.PVconstants())


def _half_cut_module():
    """144 half-cells (6 columns x 24 rows), 3 substrings of 2 columns; each substring has an upper
    and a lower branch of 24 half-cells in parallel (partial cross-tie: joined at the ends)."""
    pos, idx = [], 0
    for _ in range(3):
        sub = []
        for _half in range(2):
            sub.append([{"crosstie": j == 0, "idx": idx + j} for j in range(24)])
            idx += 24
        pos.append(sub)
    const = pvconstants.PVconstants()
    half = pvcell.PVcell(Rs=2 * pvcell.RS, Rsh=2 * pvcell.RSH, Isat1_T0=pvcell.ISAT1_T0 / 2,
                         Isat2_T0=pvcell.ISAT2_T0 / 2, Isc0_T0=pvcell.ISC0_T0 / 2, aRBD=pvcell.ARBD / 2,
                         pvconst=const)
    return pvmodule.PVmodule(cell_pos=pos, pvcells=half, pvconst=const)


def _bands():
    """Height band (0 = bottom) of every cell index, and the number of bands, per layout."""
    out = {}
    # portrait, 72 cells: STD72 = 3 substrings x 2 columns x 12 rows; list position = physical row
    band = np.zeros(72, int)
    for sub in pvmodule.STD72:
        for col in sub:
            for row, cell in enumerate(col):
                band[cell["idx"]] = row
    out["portrait"] = (band, 12)
    # landscape: the 6 columns lie along the height
    band = np.zeros(72, int)
    c = 0
    for sub in pvmodule.STD72:
        for col in sub:
            for cell in col:
                band[cell["idx"]] = c
            c += 1
    out["landscape"] = (band, 6)
    # half-cut portrait: branch h (0 upper, 1 lower) of each substring covers 12 of 24 rows,
    # serpentine over its 2 columns
    band = np.zeros(144, int)
    for k in range(3):
        for h in range(2):
            for j in range(24):
                band[k * 48 + h * 24 + j] = (1 - h) * 12 + (j % 12)
    out["half-cut"] = (band, 24)
    return out


def module_power(module, ee):
    # pvmismatch returns NaN for cells at exactly zero irradiance, which occurs in a few low-sun
    # hours of the TMY with DHI = 0 (shaded part of the panel); floor at 0.001 suns
    module.setSuns(np.maximum(ee, 1e-3))
    return float(module.Pmod.max())


def main():
    tmy = pd.read_csv(TMY, index_col=0, parse_dates=True)
    w = tmy[["dni", "ghi"]]
    winter = w.index.month.isin([10, 11, 12, 1, 2, 3])
    albedo = np.where(winter, 0.7, 0.3)
    albedo_fg = np.where(winter, 0.3, 0.2)
    arrays = [Array.from_shading_angle(bp, bg, 15, 1.0) for bp, bg in ARRAYS.values()]
    res = simulate(arrays, w, albedo=albedo, far_ground=FarGround(20, 160), albedo_fg=albedo_fg,
                   panel_glass=GLASS, settings=Settings(sky="low", n_chi=N_CHI),
                   latitude=LAT, longitude=LON, altitude=ALT)
    modules = {"portrait": _full_cell_module(), "landscape": _full_cell_module(), "half-cut": _half_cut_module()}
    bands = _bands()
    rows, profiles = [], {}
    for name, r in zip(ARRAYS, res):
        ee = (r.front + BIFACIALITY * r.back) / 1000.0              # effective irradiance [suns], (T, N_CHI)
        hours = np.flatnonzero(ee.mean(axis=1) > 0.01)
        profiles[name] = ee
        p = {k: np.zeros(len(ee)) for k in ("uniform", "minimum", *modules)}
        for t in hours:
            mean, low = ee[t].mean(), ee[t].min()
            p["uniform"][t] = module_power(modules["portrait"], mean)
            p["minimum"][t] = module_power(modules["portrait"], low)
            for lay, mod in modules.items():
                band, n = bands[lay]
                cell_ee = ee[t].reshape(n, -1).mean(axis=1)[band]
                p[lay][t] = module_power(mod, cell_ee)
            p["uniform_half-cut"] = p.get("uniform_half-cut", np.zeros(len(ee)))
            p["uniform_half-cut"][t] = module_power(modules["half-cut"], mean)
        for season, sel in (("year", np.ones(len(ee), bool)), ("winter", winter)):
            row = dict(array=name, season=season,
                       back_inhomogeneity=float(np.mean((r.back.max(axis=1) - r.back.min(axis=1))[sel & (r.back.mean(axis=1) > 10)]
                                                       / r.back.mean(axis=1)[sel & (r.back.mean(axis=1) > 10)])))
            row["minimum"] = 100 * (1 - p["minimum"][sel].sum() / p["uniform"][sel].sum())
            for lay in modules:
                ref = p["uniform_half-cut"] if lay == "half-cut" else p["uniform"]
                row[lay] = 100 * (1 - p[lay][sel].sum() / ref[sel].sum())
            rows.append(row)
        print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / "mismatch.csv", index=False)
    np.savez_compressed(OUT / "mismatch_profiles.npz", chi=res[0].chi, index=w.index.astype("int64"),
                        **{k: v.astype(np.float32) for k, v in profiles.items()})
    print("Mismatch loss relative to a uniformly irradiated module [%]")
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
