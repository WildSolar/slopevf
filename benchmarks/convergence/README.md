# Numerical convergence

`convergence.py` varies one discretisation parameter at a time from the defaults (`Settings()`)
and compares the result with a fine reference for that parameter.

- **Weather:** a full TMY year (PVGIS, Davos), with seasonal albedo of 0.7 / 0.3 and far-ground
  albedo of 0.3 / 0.2.
- **Arrays:** four configurations, `p60_g20`, `p90_g30`, `p30_g0` and `p60_g20_low` with h̄ = 0.4.
- **Reporting:** the worst case over the four arrays and both panel sides.

## Results

Each table compares against its fine reference.

**Sky grid, point sun (default)** — reference: "ultra" grid

| Setting | Yearly mean | Yearly min | Hourly mean | Hourly element | Time |
|---|---|---|---|---|---|
| "low" | 0.008% | 0.035% | 0.03% | 0.20 W/m² | 3.8 s |
| **"medium"** | 0.002% | 0.014% | 0.00% | 0.09 W/m² | 5.8 s |
| "high" | 0.000% | 0.008% | 0.00% | 0.05 W/m² | 10.0 s |

**Sky grid, Buie sun** — reference: "ultra" grid

| Setting | Yearly mean | Yearly min | Hourly mean | Hourly element | Time |
|---|---|---|---|---|---|
| "low" | 0.10% | 0.36% | 28% | 401 W/m² | 4.3 s |
| "medium" | 0.07% | 0.15% | 16% | 295 W/m² | 6.9 s |
| "high" | 0.02% | 0.32% | 6.3% | 251 W/m² | 12.3 s |

**Profile-angle intervals** — reference: 5760

| Setting | Yearly mean | Yearly min | Hourly mean | Hourly element | Time |
|---|---|---|---|---|---|
| 180 | 0.018% | 0.38% | 0.18% | 6.8 W/m² | 5.2 s |
| 360 | 0.004% | 0.17% | 0.10% | 1.7 W/m² | 5.5 s |
| **720** | 0.001% | 0.030% | 0.02% | 0.38 W/m² | 6.5 s |
| 1440 | 0.000% | 0.006% | 0.01% | 0.10 W/m² | 8.2 s |

**Panel elements** — reference: 800

| Setting | Yearly mean | Yearly min | Hourly mean | Hourly element | Time |
|---|---|---|---|---|---|
| 25 | 0.15% | 0.96% | 0.51% | – | 5.3 s |
| 50 | 0.04% | 0.54% | 0.12% | – | 5.8 s |
| **100** | 0.010% | 0.26% | 0.03% | – | 6.3 s |
| 200 | 0.002% | 0.12% | 0.01% | – | 8.2 s |

**Ground extent** — reference: 1000 panel heights

| Setting | Yearly mean | Yearly min | Hourly mean | Hourly element | Time |
|---|---|---|---|---|---|
| 10 | 0.008% | 0.011% | 0.08% | 0.97 W/m² | 5.3 s |
| 25 | 0.000% | 0.001% | 0.00% | 0.07 W/m² | 5.7 s |
| **50** | 0.000% | 0.000% | 0.00% | 0.01 W/m² | 6.4 s |
| 100 | 0.000% | 0.000% | 0.00% | 0.00 W/m² | 7.7 s |

Defaults in bold. The times are for a full year and four arrays together, on an Intel Core Ultra 5 238V.

Column definitions:

| Column | Meaning |
|---|---|
| Yearly mean | Relative error of the yearly panel-average irradiation |
| Yearly min | Relative error of the yearly sum of the hourly panel minimum, which is relevant for mismatch |
| Hourly mean | Largest relative error of an hourly panel average, for hours with more than 50 W/m² |
| Hourly element | Largest absolute error of an hourly element value |

## Interpretation

- **Point sun:** with the point sun, even the coarsest sky grid is converged. The beam is evaluated
  exactly, and the remaining diffuse sky is smooth.
- **Buie sun:** with the Buie sunshape, yearly totals converge well, but single hours do not. The beam
  sits in individual sky cells, and near the edge of the row-to-row shadow this shifts the shadow on
  the panel by up to one cell. This is the main numerical reason for the point sun as default.
- **Panel elements:** the panel average converges quickly. The panel minimum converges only slowly,
  because finer elements resolve a deeper minimum. This is a property of the quantity, not an error.
  For mismatch studies, the element size should match the physical cell rows of the module.
- **Table resolution and ground extent:** both are converged at their defaults.
