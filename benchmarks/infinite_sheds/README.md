# Comparison with pvlib's infinite_sheds model

`compare.py` compares slopevf with `pvlib.bifacial.infinite_sheds` (pvlib 0.15) on horizontal
ground. Both are two-dimensional view-factor models of infinitely long rows. The comparison uses
an isotropic sky, which both models represent exactly. pvlib's `haydavies` option only moves the
circumsolar part into the beam, so it reduces to the same case.

The matrix covers 180 cases:

| Parameter | Values |
|---|---|
| Panel tilt | 20, 30, 45, 60, 90° |
| Shading angle | 10, 20° |
| h̄ | 0.6, 1.0, 2.0 |
| Skies | 6 |

Albedo is 0.5 throughout. Panel averages are compared component by component.

## Results

| Component | Deviation slopevf / pvlib − 1 |
|---|---|
| Direct + sky diffuse, front | max 0.003 W/m² (0.00%) |
| Direct + sky diffuse, back | max 0.005 W/m² (0.03%) |
| Ground-reflected, front | mean +1.4%, P95 24%, max 46% |
| Ground-reflected, back | mean −5.3%, P95 34%, max 49% |

**Direct + sky diffuse.** The two models agree exactly. The row geometry, the row-to-row shading and
the view factors to the sky are identical.

**Ground-reflected.** The differences stem from the treatment of the ground.

- `infinite_sheds` averages the ground irradiation over one row period, so the ground is uniformly
  bright.
- slopevf resolves where the irradiated and shaded strips lie relative to the panel.

The effect grows with decreasing clearance. At h̄ = 2 the backside mean is −1.1% (max 21%); at
h̄ = 0.6 it is −11.8% (max 49%).

`raytrace_check.py` ray-traces the five cases with the largest deviations in Radiance, using an
isotropic sky and the scenes of `../raytrace`. The ground-reflected component comes from runs with
albedo 0.5 and 0.

| Model | Ground-reflected deviation from ray tracing |
|---|---|
| slopevf | within ±0.2% on both sides |
| infinite_sheds, front | −32 to +12% |
| infinite_sheds, back | −30 to +97% |

The averaged ground is a deliberate simplification of `infinite_sheds` that keeps the model fast
and simple. For panel averages at typical clearances its effect on the total backside irradiation is
moderate: the backside total deviates by −4.4% on average over all cases. It matters for low
clearances, dense arrays, and anything that relies on the backside distribution.
