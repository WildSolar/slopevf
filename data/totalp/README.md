# Totalp measurement data

Measurements of the adjustable miniature bifacial PV array at the Totalp site near Davos,
Switzerland, winter 2024/25 (see the paper cited in the main README for the setup). Used by
`benchmarks/validation/totalp_validation.py`. License: CC BY 4.0.

## Setup

- Three rows of seven miniature modules each; each module has three cells side by side. A row is
  about 15.4 panel heights long (including the covered gaps between modules).
- Panel height (row height) H = 96 mm.
- Centre height of the panels: 1.77 H, measured perpendicular to the sloped ground.
- The modules measure the average irradiation over the full row height.
- Frontside channel (3): module on the last (uphill) row, with two rows in front of it.
- Backside channel (22): module on the first (downhill) row, with two rows behind it.
- Plain glass, no anti-reflective coating or texture.
- Tilt, ground slope and row spacing are adjusted automatically, several configurations per day.
  Each record carries its own configuration.

## `Totalp_Measurement_Data.csv`

Semicolon-separated, decimal comma.

| Column | Description |
|---|---|
| `Timestamp` | Time of the measurement |
| `channel` | 22: backside, 3: frontside |
| `RA` | Row spacing, absolute, in the plane of the slope [mm] |
| `MN_relHor` | Panel tilt versus horizontal [deg] |
| `HN` | Ground slope [deg] |
| `Pnom` | Power output normalised to STC (dimensionless) |
| `elevation`, `azimuth` | Sun position [deg]; azimuth 0 = south, positive towards west |
| `DNI`, `GHI` | Direct normal and global horizontal irradiance [W/m²], measured locally (i.e. including the local horizon) |

The non-dimensional horizontal row spacing used by slopevf is ρ̄ = RA · cos(HN) / 96.

The timestamps are the database clock with daylight saving time: UTC + 124 min until 27 Oct 2024,
UTC + 64–65 min afterwards (Swiss civil time plus a clock drift of about 4–5 min, determined by
aligning `GHI` with the station's 1-min GHI). The sun position columns were computed from these
timestamps; the azimuth lags the true sun by about 1° (about 4 min). Neither matters for the
validation, which uses the stored sun position and the irradiance of the same record.

## `snow_cover.csv`

Daily snow cover of the surroundings (0: none, 1: closed cover), 1 Oct 2024 – 31 Mar 2025. The
heated base plate itself was kept free of snow.

- October 2024: estimated from the webcam time lapse (spotty cover until mid-October).
- 1–20 November 2024: from the daily ratio of the downward- and upward-facing SMP10 pyranometers of
  the full-size installation (bare ground 0.12, closed cover ≥ 0.7), linearly mapped to 0–1.
- From 21 November 2024: closed snow cover (ratio ≥ 0.71 in Dec–Feb, ≥ 0.58 in March).

Source of the station data: export of the Mini_PVA_Analysis project
(`scripts/export_v1_period_weather.py`).

## `Totalp_Horizon.hor`

Local horizon of the site: azimuth [deg] (0 = south, positive towards west) and horizon
elevation [deg]. Read with `slopevf.Horizon.from_file(path)`.
