# slopevf — notes for Claude sessions

View-factor model for the location-resolved irradiation on the front and back side of PV panels
in multi-row arrays on arbitrarily sloped, reflective ground (Python port of the user's MATLAB
model, Solar Energy 2026, plus the extensions of the follow-up paper: finite arrays, panel
thickness, specular and diffuse reflection between rows, glass transmission, local horizon).
Closed-form view factors; a full year for one array takes about a second. `README.md` is the
user documentation, `docs/legacy.md` the corrections relative to the MATLAB code.

Released: PyPI `slopevf` 1.0.0, https://github.com/WildSolar/slopevf, Zenodo concept DOI
10.5281/zenodo.23666920 (version 1.0.0: 10.5281/zenodo.23666921). Cite the version DOI and the
paper in publications.

## Using it from another project

Declare it in that project's `pyproject.toml` (editable, so it follows this repository), then
`uv sync`:

```toml
[project]
dependencies = ["slopevf[solar,plot]"]

[tool.uv.sources]
slopevf = { path = "C:/Users/wimm/Documents/packages/slopevf", editable = true }
```

To freeze a finished project, or for code that others install, depend on a released version
from PyPI instead (`"slopevf>=1.0"`, without the uv source). Don't copy the code.

```python
from slopevf import Array, FarGround, Horizon, PanelGlass, Settings, simulate

# lengths relative to the panel (row) height; angles in degrees
arr = Array(beta_p=60, beta_g=20, rho_bar=1.6, h_bar=1.2, gamma=0, thickness=0.0)
arr = Array.from_shading_angle(60, 20, alpha_sh=15, h_bar=1.0)      # spacing from shading angle

# weather: DataFrame/dict with dni, ghi (or dhi) [W/m2] and solar_azimuth / solar_elevation [deg],
# azimuth from SOUTH, positive towards WEST; or a DatetimeIndex + latitude/longitude (needs pvlib)
res = simulate(arr, weather, albedo=0.5,                              # scalar or per time step
               far_ground=FarGround(beta=20, azimuth=160), albedo_fg=0.3,
               horizon=Horizon.from_file("site.hor"), albedo_terrain=0.5,   # optional
               panel_glass=PanelGlass(n=1.52, transmission=True, diffuse=0.0),  # optional
               settings=Settings(sky="low", n_chi=100, finite_rows=None))
res.front, res.back            # (n_times, n_chi) irradiance along the panel, chi = -0.5 bottom .. +0.5 top
res.front_mean, res.back_min   # pandas Series per time step
res.summary()                  # averages and components (direct, nearground, farground, reflected, diffuse)
```

Pitfalls:

- **Azimuths:** south = 0, positive towards west (the paper's convention). perezsky uses north = 0,
  clockwise; slopevf converts internally.
- **Geometry:** `h_bar` is the *vertical* height of the panel centre above the sloped ground and
  `rho_bar` the *horizontal* row pitch, both divided by the panel height (a height measured
  perpendicular to the slope must be divided by cos(beta_g)). Panels face downhill; `gamma` is
  the downhill azimuth. `from_shading_angle` avoids mistakes in design studies.
- **Settings:** the default point sun (`sun_model="point"`) makes `sky="low"` sufficient and fast;
  `sun_model="buie"` (circumsolar cone, as in the first paper) needs a finer sky.
- **Finite arrays:** `finite_rows=(n_front, n_back)` = rows in front of (downhill) / behind
  (uphill) the evaluated row; `(0, 0)` is a single row. Default `None` = infinite array.
- **Glass:** without `panel_glass` the panels are black and the result is the irradiance on the
  module plane. `PanelGlass(transmission=True)` gives the effective irradiance behind the glass
  (IAM-weighted, normalised to normal incidence); `diffuse=R_d` adds diffuse reflection off the
  adjacent row (single reflection; underestimates by 10-25 % of that term).
- **Horizon:** weather must be free-horizon (PVGIS masks the DNI by default; use usehorizon=False).
- **Memory:** results hold several (n_times x n_chi) arrays per array configuration; for many
  configurations over a year, reduce `n_chi` or process in batches.
- **Legacy:** `Settings.matlab()` reproduces the MATLAB code including its errors; only for
  reproducing the first paper.

## Working on this repository

- Environment: `.venv` (Python 3.10, editable slopevf and perezsky, extras `all`). Tests:
  `.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider` (about 5 min).
- Radiance (benchmarks, Radiance tests): `$env:RADIANCE_DIR = "<repo>\.venv\Lib\site-packages\pyradiance"`.
  Ray-tracing caches live in `benchmarks/*/work/` (ignored, several hundred MB; rerunning
  everything takes hours).
- The follow-up manuscript lives in the private repository
  `C:\Users\wimm\Documents\GitHub\General_Irradiation_Model` (`Publication_Followup/`); its
  figures are read from `benchmarks/*/results/` and `examples/results/` here.
- Sky model: the separate package perezsky (`C:\Users\wimm\Documents\packages\PerezSky`). Change
  it there, not here.
- Releases: bump the version in `pyproject.toml`, `src/slopevf/__init__.py` and `CITATION.cff`;
  the user publishes a GitHub release `vX.Y.Z`, which Zenodo archives and
  `.github/workflows/publish.yml` publishes to PyPI (trusted publishing). Then add the new
  version DOI to `CITATION.cff` and the README citation.
- git is not on PATH: use GitHub Desktop's bundled
  `%LOCALAPPDATA%\GitHubDesktop\app-*\resources\app\git\cmd\git.exe`. Commit on `main` with a
  message file (`commit -F`, UTF-8 without BOM); never push — the user pushes from GitHub Desktop.
