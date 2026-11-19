"""Panel mean at a sensor on the edge of the direct shadow: convergence with the number of RT sensors.

In the 102-scene benchmark (run_benchmark.py) the ray tracing is evaluated at 20 points per panel side, and
points on the edge of the row-to-row shadow are excluded from the comparison (a point sample of a step is
not comparable with the element average of the model). One case, g10_p90 / clear_behind (backside), has such
a point: with all 20 sensors the panel mean of the model deviates by +4.4 % from the ray tracing, without the
point by +0.001 %. This script evaluates that scene with 20, 100, 400 and 1000 sensors per panel side: the
sample mean converges to the exact (element-averaged) panel mean of the model, i.e. the deviation of the 20-point
mean is the quadrature error of the point sampling (about the lit fraction / number of sensors).

Usage:  python edge_convergence.py      (results in results/edge_convergence.txt)
"""

from __future__ import annotations

import numpy as np

import radscene as rs
import run_benchmark as rb
from slopevf import FarGround, Settings, simulate

CASE, SKY = "g10_p90", "clear_behind"
N_SENSORS = (20, 100, 400, 1000)


def main():
    env = rs.radiance_env()
    array = dict(rb.geometries())[CASE]
    el, az, dni, dhi, ghi = rb.sky_inputs(SKY, array.gamma)
    wd = rb.WORK / "edge_convergence"
    wd.mkdir(parents=True, exist_ok=True)
    oct_ = rs.build_octree(wd, env, "scene", {
        "sky.rad": rs.sky_rad(el, az, dni, dhi, ghi, rb.ALBEDO_FG),
        "scene.rad": rs.scene_rad(array, "ideal", rb.ALBEDO, ghi, rb.ALBEDO_FG)})
    # exact (element-averaged) panel mean of the model, fine panel discretisation
    res = simulate(array, dict(dni=[dni], ghi=[ghi], solar_azimuth=[az], solar_elevation=[el]), albedo=rb.ALBEDO,
                   far_ground=FarGround(0, 0), albedo_fg=rb.ALBEDO_FG, settings=Settings(n_chi=2000, sky="medium"))
    vf_back = float(res.back_mean.iloc[0])
    lines = [f"## {CASE} / {SKY}: backside panel mean, ray tracing with n sensors per panel side vs. model",
             f"model (2000 elements): {vf_back:.3f} W/m2", "n sensors   RT mean [W/m2]   model / RT - 1 [%]"]
    for n in N_SENSORS:
        chi = -0.5 + (np.arange(n) + 0.5) / n
        front, back = rs.sensors(array, chi, rb.X_POS)
        e = rs.irradiance(oct_, back.reshape(-1, 6), wd, env, rs.DEFAULT_RTRACE, rb.N_PROC)
        rt = float(e.reshape(len(rb.X_POS), n).mean())
        lines.append(f"{n:9d}   {rt:14.3f}   {100 * (vf_back / rt - 1):+17.3f}")
        print(lines[-1], flush=True)
    text = "\n".join(lines)
    (rb.RESULTS / "edge_convergence.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
