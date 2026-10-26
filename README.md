# slopevf

[![PyPI](https://img.shields.io/pypi/v/slopevf)](https://pypi.org/project/slopevf/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23666920.svg)](https://doi.org/10.5281/zenodo.23666920)

Location-resolved irradiance on the front and back side of PV panels in an infinite
field of regularly spaced rows on arbitrarily sloped, reflective ground.

This is the Python implementation of the view-factor model described in

> M. Wild, S. Strebel, D. Anderegg, J. Rohrer: *A general view-factor based irradiation
> distribution model for multi-row bifacial PV arrays on sloped and reflective ground.*
> Solar Energy (2026).

The original MATLAB implementation is available at
[github.com/WildSolar/General_PV_Irradiation_Model](https://github.com/WildSolar/General_PV_Irradiation_Model).

## Installation

```bash
pip install slopevf               # with sun position and plots: pip install "slopevf[solar,plot]"
```

The core dependencies are numpy, scipy, pandas and [perezsky](https://github.com/WildSolar/PerezSky)
(the sky model). pvlib is optional (sun position from timestamps) and so is matplotlib (plots).
The extras `examples` and `benchmarks` install what the scripts in `examples/` and `benchmarks/`
of the repository need (pvmismatch; Radiance via pyradiance, bifacial_radiance).

## Usage

```python
from slopevf import Array, FarGround, Settings, simulate

array = Array(beta_p=60, beta_g=20, rho_bar=1.125, h_bar=1.84, gamma=0)
# or: Array.from_shading_angle(beta_p=60, beta_g=20, alpha_sh=15, h_bar=1.0)

res = simulate(array, weather, albedo=0.79,
               far_ground=FarGround(beta=20, azimuth=160), albedo_fg=0.3)

res.front, res.back            # (n_times, n_chi) irradiance along the panel [W/m²]
res.front_mean, res.back_min   # panel average / minimum per time step (pandas Series)
res.summary()                  # all averages and components as a DataFrame
res.plot(0)                    # distribution of one time step
```

### Inputs

- **Geometry:** all lengths are relative to the panel height. Angles are in degrees.
  Azimuths are measured from south, positive towards west (the paper's convention).
- **Weather:** a DataFrame or dict with `dni`, `ghi` (any two of `dni`/`ghi`/`dhi` work),
  `solar_azimuth` and `solar_elevation`.
  - If the sun angles are missing, pass a DatetimeIndex plus `latitude`/`longitude`,
    and pvlib computes them.
  - The albedos can be scalars, one value per time step, or `albedo`/`albedo_fg`
    columns.
- **Several arrays:** pass a list of `Array`s. The sky radiance is computed once and
  shared between them.

### Local horizon

```python
from slopevf import Horizon

horizon = Horizon.from_file("Totalp_horizon.hor")   # azimuth [deg] (0 = south, +west), elevation [deg]
res = simulate(array, weather, albedo=0.79, horizon=horizon, albedo_terrain=0.5)
```

- **Masking:** sky elements below the horizon receive the radiance of the terrain instead of the
  sky, and the beam is blocked while the sun is behind the horizon. This affects the panels and
  the near ground consistently.
- **Distant terrain:** both the terrain above the horizon and the far ground below the horizontal
  are irradiated under free-horizon conditions, because their own horizon is unknown.
- **Terrain radiance:** the horizon gives elevation per azimuth only, so the terrain is modelled as
  a Lambertian surface:

  | Model | Irradiation on the terrain |
  |---|---|
  | `terrain="facing"` (default, `terrain_tilt=30`) | a slope facing the array, lit by the beam (unless the sun is behind the horizon), the diffuse sky and its surroundings |
  | `terrain="horizontal"` | GHI, i.e. L = albedo · GHI / π |

- **Inputs:**
  - `albedo_terrain` can be a scalar, a series per time step or a weather column. It defaults to
    the far-ground albedo.
  - Files with azimuths from north (pvlib/GIS convention) are read with
    `azimuth_convention="north"`.
- **Negative horizon elevations** (curvature of the earth at high altitude) are treated as 0.
- **The weather data must be free-horizon.** PVGIS data obtained through
  `pvlib.iotools.get_pvgis_tmy` include PVGIS's own horizon by default; request them with
  `usehorizon=False`.

`examples/horizon_example.py` quantifies the effect at the Totalp site in Davos. Front-side
irradiation per year:

| Treatment | Change |
|---|---|
| Default (terrain facing the array, 30°) | −3.8 to −5.1% |
| Terrain facing the array, 60° | −4.1 to −5.5% |
| Horizontal terrain | −3.4 to −4.6% |
| Masking only (black terrain) | −5.0 to −6.7% |

`benchmarks/raytrace/horizon_check.py` verifies the treatment against Radiance with terrain built
around the array. The horizon effect agrees within 0.1%. Compared with physically lit terrain slopes,
the `"facing"` model deviates by at most 2%, the `"horizontal"` model by up to 14% in single
situations: backsides facing sunlit terrain, or the sun hidden behind the horizon.

### Finite arrays

```python
res = simulate(array, weather, albedo=0.79, settings=Settings(finite_rows=(0, 2)))
```

`finite_rows=(n_front, n_back)` evaluates a row with `n_front` rows in front of it (downhill) and
`n_back` rows behind it (uphill); `(0, 0)` is a single row. Only these rows shade the ground.
A side without an adjacent row faces away from the array: it is not shaded by another row and sees
the ground out to infinity and, facing downhill, the terrain below the horizontal as far ground.
Verified against Radiance (`benchmarks/raytrace/finite_array_check.py`, `outward_check.py`).

### Panel reflection

```python
from slopevf import PanelGlass

res = simulate(array, weather, albedo=0.79, panel_glass=PanelGlass(n=1.52))
res.back_reflected, res.front_reflected      # contributions of the reflection [W/m²]
```

The module glass reflects specularly with the Fresnel reflectance of uncoated glass. Two paths are
modelled:

- off the front of the uphill row onto our back side, and off the back of the downhill row onto
  our front side;
- the reflected light that misses the adjacent row lands on the ground and is reflected onto both
  sides.

**Why the geometry stays 2D:** a reflection off a module plane leaves the component along the rows
unchanged. The geometry therefore stays two-dimensional, while the reflectance uses the true 3D
incidence angle of each sky element.

**Options:**

- `PanelGlass(front=..., back=...)` selects the reflecting sides.
- `reflectance=` accepts a custom function R(cos θ), e.g. for anti-reflective coatings.
- `diffuse=` adds the diffuse (Lambertian) reflection of the module surface: cells, inactive area,
  frames. Each face then reflects R_d times the irradiance it receives onto the facing side of the
  adjacent row (`res.front_diffuse`, `res.back_diffuse`). This matters for the backside at dense
  spacing, where the sunlit front of the next row is the brightest surface in view. Only the direct
  exchange between facing panels is modelled (single reflection). Compared with Radiance
  (`benchmarks/raytrace/diffuse_check.py`), this captures 75–90 % of the diffuse contribution in
  dense arrays; the remainder is mostly panel → ground → panel.

**Not included:** the glass mirroring the bright ground (ground → glass → panel).

**Effect.** Uncoated glass with steep panels and a low sun can contribute more than half of the
backside irradiance. In winter at Davos, with vertical panels on a 15–30° slope, the backside gains
9–38%.

**Verification.** `benchmarks/raytrace/reflection_check.py` compares against Radiance, using mirrors
of constant reflectance. The total irradiance agrees within 0.7% on the front and 2.0% on the back;
the remainder is the path that is not included.

### Settings

| Option | Default | Meaning |
|---|---|---|
| `n_chi` | 100 | Number of panel elements |
| `n_alpha` | 720 | Profile-angle resolution of the multiplication table (0.25°) |
| `sky` | `"medium"` | Sky grid: `"low"`, `"medium"` (~20k cells, as in the paper), `"high"`, `"ultra"`, or `(band_deg, n_azimuth)` |
| `sun_model` | `"point"` | `"point"`: the beam is a point source with exact shading at the sun's profile angle; `sky="low"` is then sufficient. `"buie"`: the beam is spread over the circumsolar cone following the Buie sunshape, as in the paper. |
| `solar_constant` | 1361 | Extraterrestrial irradiance [W/m²] for the Perez sky brightness. It is corrected for the Earth–Sun distance if the weather has a DatetimeIndex or a `day_of_year` column. |
| `legacy` | `False` | `Settings.matlab()` reproduces the MATLAB code exactly |

## Speed

A full year of hourly data (about 4,400 daylight hours) for 4 array configurations,
measured on a laptop:

| Settings | Time |
|---|---|
| `Settings(sun_model="buie")` | ~6 s |
| `Settings(sky="low")` (point sun) | ~4 s |

This speed comes from two things:

1. **Closed-form view factors.** The near-ground and far-ground view factors are exact
   closed-form expressions, so building the tables takes well under a second.
2. **Sparse matrix products.** All geometry-only quantities are precomputed once per
   array as sparse matrices. The irradiance for each block of time steps is then a pair
   of matrix products.

## Legacy mode and corrections

`Settings.matlab()` reproduces the published MATLAB implementation; the tests compare it
against MATLAB reference output. The default mode fixes several issues found during the
port, such as the air mass, the Perez coefficients, the far-ground view factor and
negative far-ground irradiance. See
[docs/legacy.md](https://github.com/WildSolar/slopevf/blob/main/docs/legacy.md) for the full list.

## Tests

```bash
python -m pytest
```

The suite checks:

- geometry against brute-force 2D ray tracing;
- the closed-form view factors against numerical integration;
- energy closure of the sky model;
- legacy mode against MATLAB.

The MATLAB reference data can be regenerated with `tests/matlab/generate_reference.m`.

## Repository contents

Besides the package (`src/slopevf`) and its tests, the repository contains the verification and
validation studies of the follow-up paper (`benchmarks/`: Radiance ray tracing, bifacial_radiance,
pvlib's infinite_sheds, convergence, timing, Totalp validation), application examples
(`examples/`) and the measurement data of the Totalp miniature array (`data/totalp/`, CC BY 4.0).

## Citation

If you use slopevf in published work, please cite the paper above and the version you used,
archived on Zenodo:

> M. Wild: *slopevf: View-factor based irradiation distribution on multi-row bifacial PV arrays
> on sloped and reflective ground*, version 1.0.0, Zenodo, 2026. https://doi.org/10.5281/zenodo.23666921

The DOI [10.5281/zenodo.23666920](https://doi.org/10.5281/zenodo.23666920) always resolves to the latest version. Citation
metadata: [CITATION.cff](https://github.com/WildSolar/slopevf/blob/main/CITATION.cff).

## License

The code is licensed under the MIT license, see
[LICENSE](https://github.com/WildSolar/slopevf/blob/main/LICENSE). The measurement data in
`data/` are licensed under CC BY 4.0.
