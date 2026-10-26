"""Timing of a full-year hourly simulation (clear-sky weather, Davos), several configurations."""

import time

import pandas as pd
import pvlib

from slopevf import Array, FarGround, Settings, simulate

loc = pvlib.location.Location(46.838, 9.813, altitude=1560)
times = pd.date_range("2025-01-01", "2025-12-31 23:00", freq="h", tz="UTC")
cs = loc.get_clearsky(times)
weather = pd.DataFrame({"dni": cs["dni"], "ghi": cs["ghi"]}, index=times)
albedo = [0.7 if m in (10, 11, 12, 1, 2, 3) else 0.3 for m in times.month]

arrays = [Array.from_shading_angle(60, slope, 15, 1.0) for slope in (0, 10, 20, 30)]
print(f"{(cs['ghi'] > 0).sum()} daylight hours")
for settings in (Settings(sun_model="buie"), Settings(), Settings(sky="low")):
    t0 = time.perf_counter()
    res = simulate(arrays, weather, albedo=albedo, far_ground=FarGround(20, 160), albedo_fg=0.2,
                   settings=settings, latitude=loc.latitude, longitude=loc.longitude, altitude=loc.altitude)
    dt = time.perf_counter() - t0
    yearly = [r.front_mean.sum() / 1e6 + 0.9 * r.back_mean.sum() / 1e6 for r in res]
    print(f"sun_model={settings.sun_model:5s} sky={settings.sky:6s}: {dt:5.1f} s for {len(arrays)} arrays; "
          "yearly front+0.9*back [MWh/m2]: " + ", ".join(f"{y:.3f}" for y in yearly))
