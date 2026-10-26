# Ray-tracing benchmark on sloped terrain

Compares slopevf with Radiance ray tracing (`rtrace`) for multi-row arrays on sloped ground.
bifacial_radiance places all rows at one height and adds a horizontal ground disc, so the sloped
scenes are built directly from Radiance primitives (`radscene.py`). The Radiance programs used are
the same ones bifacial_radiance calls internally.

## Setup

- Radiance ≥ 6.0 (e.g. the official Windows zip): set `RADIANCE_DIR` to its folder, or put
  `Radiance/bin` on `PATH`.
- A Python environment with slopevf installed (`pip install -e python`).

```bash
python run_benchmark.py skycheck            # sky model only
python run_benchmark.py pilot               # convergence of rtrace settings and field size
python run_benchmark.py full --level ideal  # implementation check
python run_benchmark.py full --level field  # effect of the far-ground idealisation
```

Ray-tracing results are cached in `work/`, which is git-ignored. Summaries and profiles are written
to `results/`.

## Design

**Sky.** `gendaylit` differs from slopevf's sky in two ways:

- Its Perez brightness parameter Δ uses a date-dependent extraterrestrial irradiance.
- `perezlum.cal` blends the sky with the ground brightness near the horizon. At the horizon this
  is about 45% ground, which changes the irradiance on vertical planes by up to 5%.

To keep sky-model differences out of the geometry check, the Radiance sky uses slopevf's
coefficients through its own `.cal` file. It has no horizon blending and is normalised to DHI by
fine numerical integration. The sun is a 0.533° disc carrying DNI, and the ground below the horizon
is a uniform glow of GHI·albedo_fg/π. The step 0 check (`skycheck`) confirms that the two skies
agree within ±0.2% on horizontal and tilted planes of all orientations.

**Level `ideal`** matches slopevf's idealisations:

- thin black panels without gaps;
- a Lambertian ground;
- 2 × 60 panel heights of rows, each row 200 panel heights long, so the field is effectively
  infinite;
- sensors on the centre row, averaged over 4 positions along the row.

slopevf treats everything seen over the top edge of the downhill row as horizontal far ground. In
the ray-tracing scene this region is replaced by a glowing plane of the same radiance. The plane
runs through that top edge, parallel to the slope. A `mixfunc` makes it visible only to rays
travelling downwards and transparent to everything else, including sun rays to the ground below
it. Any remaining difference is therefore the error of the slopevf implementation and its
discretisation.

**Level `field`** omits that plane, so the view over the downhill row shows the actual field.
Comparing the two levels quantifies the far-ground idealisation of the model.

The two levels agree within 0.03%, and this follows from the geometry. Because β_p ≥ β_g, the line
of sight from any panel point over the top edge of the downhill row is never steeper than the slope
itself. Everything slopevf calls "far ground" therefore lies beyond the infinite slope. In an
infinite field the far-ground treatment is exact; the only assumptions are the orientation and
albedo of the distant terrain.

## Flat ground: bifacial_radiance's own scene builder

`flat_bifacial_radiance.py` compares slopevf with bifacial_radiance (0.5.4) on horizontal ground,
using its scene builder, ground disc and sensor positions. It covers 5 geometries × 6 skies.
`flat_decomposition.py` attributes the deviations of the native variant to the individual defaults.

| Variant | Setup | Max. panel-mean deviation |
|---|---|---|
| matched | bifacial_radiance scene, but non-reflective modules of negligible thickness without gaps, slopevf sky, accurate `rtrace` | 0.08% front, 0.15% back |
| native | all bifacial_radiance defaults | 2.5% front, 7.1% back |

The native deviations are fully explained by the inputs:

- the `gendaylit` sky: up to 2% on the front side under overcast skies;
- the 2 cm default module: 3–4% on the back side when the sun is just above the shading angle;
- the 1% reflectance of the `black` material: −1.3 to −1.8% on the back side;
- ambient caching in the default settings: about ±1%.

Notes:

- slopevf is evaluated for the geometry bifacial_radiance actually builds. Its ground disc lies 1 cm
  below z = 0, and the default module sits on top of its axis.
- bifacial_radiance evaluates the irradiance on the module surface itself (`rtrace -i`), so its 5 mm
  sensor offset does not cause the bias seen with point sensors.
- On Windows, `xform -a` re-runs the module's `!genbox | xform` command for every module in the
  field, one shell launch each, which takes hours for large fields. The scripts evaluate the module
  file once beforehand, which gives identical geometry.
- To pause a long run, create `work/flat/STOP`; finished cases are cached in `work/flat/cache.json`.

## Local horizon

`horizon_check.py` builds the Totalp horizon (`Totalp_Horizon.hor`) as terrain 10,000 panel heights
from the array, so every point of the array sees the same horizon. It runs 3 geometries × 4 skies,
including the sun hidden behind the horizon.

| Level | Terrain in Radiance | Max. deviation slopevf / Radiance − 1 |
|---|---|---|
| A (implementation) | A wall along the horizon line, glowing with exactly the radiance of the slopevf terrain model | Horizon effect within 0.03% (front) and 0.08% (back) |
| B (terrain model) | Lambertian slopes, tilted 30° and facing the array, physically lit (sun with cast shadows, Perez sky, interreflection) | `"facing"` model: 1.0% (front), 2.1% (back). `"horizontal"` model: 7.9% (front), 14.2% (back) |

The large deviations of the `"horizontal"` model occur in two situations:

- **Sun hidden behind the horizon:** the terrain faces seen from the array then lie in their own
  shadow, while the horizontal model still gives them the full GHI.
- **Backside facing sunlit terrain:** sunlit slopes facing the array are much brighter than
  horizontal terrain.

The synthetic terrain is built as slopes facing the array, so Level B partly favours the `"facing"`
model by construction. It tests the simplified irradiation of the terrain (isotropic sky, sun
visibility taken from the array, no interreflection), but not the assumed orientation.

The check also exposed an inconsistency. While the sun was hidden, the far ground lost its beam
irradiation, whereas the terrain above the horizon kept the full GHI. Distant terrain is now
consistently irradiated under free-horizon conditions.

## Panel reflection

`reflection_check.py` gives the panels of the `ideal` scenes a perfectly specular surface of
constant reflectance R = 0.05 on both sides (Radiance `mirror`). For planar mirrors Radiance
builds exact virtual sources, i.e. mirror images of the sun. With angle-dependent glass
(`BRTDfunc`), the reflected sun would only be found by chance and the result would be very noisy.
slopevf uses the same constant reflectance; the Fresnel formula itself is covered by the unit tests.
The case matrix is 5 geometries × 5 skies.

Results: the total irradiance (including reflection) agrees within 0.72% on the front side and
1.96% on the back side.

- **Dominant case (Totalp, β_p = 90°, sun low in front).** Reflection supplies 62% of the backside
  irradiance; slopevf gives 23.27 W/m², Radiance 23.26 W/m².
- **Missing path.** The remaining deviation comes from a path slopevf leaves out: the glass also
  mirrors the brightly lit ground between the rows (ground → glass → panel). It shows up mainly with
  a high sun on steep vertical arrays (back side −2.0%, front side −0.6%); the mean is −0.3%.
- **The ground path matters.** Without the mirror → ground → panel path, the deviations reached
  4.7%. For flat ground and a low sun, most of the reflected bundle passes below the adjacent row
  and lands on the ground.

## Finite row length and module thickness (Totalp miniature array)

`finite_array_check.py` quantifies the two geometric idealisations left in the Totalp
validation:

- **Row length:** slopevf's rows are infinitely long, but the miniature rows are only
  15.4 panel heights long.
- **Thickness:** slopevf's panels are infinitely thin, but each row is a stack of about 15 mm
  (0.16 H): two monofacial modules back to back with a central rod of 25 × 7 mm, lying flat,
  so the 7 mm side adds to the stack.

The scenes reproduce the actual layout:

- the frontside module on the last row (rows −2, −1, 0), the backside module on the first
  (rows 0, +1, +2);
- black panels on Lambertian snow (albedo 0.79), slopevf's Perez sky, and 8 measured time steps
  for each of 12 configurations;
- sensors on the measuring module only (module 4 of 7, the central seventh of the row), over
  its full height, on the outer face of the stack.

Each scene is traced in three variants: 600 H thin rows (slopevf's idealisation), 15.4 H thin
rows, and 15.4 H rows with the 15 mm stack centred on the row plane.

| Side | Length: E(15.4 H) / E(600 H) | Thickness: E(15 mm) / E(thin) | Total |
|---|---|---|---|
| Back, 60° | 1.013–1.030 | 0.960–0.980 | 0.972–1.009 |
| Back, 90° | 1.017–1.039 | 0.871–0.970 | 0.886–1.008 |
| Front | 1.000–1.003 | 0.975–0.991 | 0.977–0.991 |

- **Row length.** Light entering past the row ends mainly reaches the back side, through the
  ground lit beside the rows. On the central module the effect is small. Averaged over the
  whole row it would be 8–14%, so the position of the measuring module matters.
- **Thickness.** The stack is the larger effect, as with the panel thickness in the benchmark
  above. It grows with denser spacing and steeper slopes (back side −13% at 90°/30°/ρ̄ = 1.04).
  Likely cause (not decomposed): the stack widens the shadows on the ground, which narrows the
  sunlit strips between the rows that the back side sees.

**slopevf against these scenes.** slopevf models the stack with `Array(..., thickness=...)` and
the three rows with `Settings(finite_rows=...)`. It is compared with the 600 H rows, thin and
thick; the glow below the horizon is the continuing slope (albedo × irradiance on the slope / π),
like slopevf's far ground.

| Side | slopevf / Radiance, thin | slopevf / Radiance, 15 mm | Thickness effect, Radiance / slopevf |
|---|---|---|---|
| Back (9 configurations) | 0.999–1.001 | 0.999–1.001 | equal within 0.001 |
| Front (3 configurations) | 0.994–1.015 | 0.996–1.003 | 0.974–0.991 / 0.958–0.991 |

On the front side, slopevf treats everything seen over row −1 as uniform far ground; in the
three-row scene this includes row −2 and its shadow.

The comparison exposed a missing source. Sloped ground also sees the downhill terrain below the
horizon (the wedge between the horizontal and the slope, up to (1 − cos β_g)/2 of its view).
In infinite arrays the rows block most of it, which is why the benchmark above did not show it.
The unshaded ground around a finite array receives all of it. Without this source, slopevf
underpredicted the back side on 20–30° slopes by 2–5%. It is now included: the ground is
irradiated from profile angles in (−β_g, 0) with the far-ground radiance.

## Pitfalls found while setting up the comparison

- **Sensor offset.** Rear-side sensors 1 cm behind the panel surface read 1–3% low where the panel
  is close to the ground. The sensors are placed 1 mm off the surface.
- **Panel thickness.** Close to the shading angle the sunlit ground strips are very narrow. A panel
  thickness of only 0.2% of the panel height reduced their width, and with it the ground-reflected
  beam on the rear side, by about 12% in one case. The panels in the `ideal` and `field` scenes are
  therefore practically infinitely thin (1e-5 H). Real frames, at about 0.02 H, would have a
  noticeable effect on rear irradiance near the shading angle.
- **Shadow edges.** On the edge of a direct row-to-row shadow, a point sample (ray tracing) and an
  element average (slopevf) can't be compared. Sensors within 0.01 of the edge are excluded from
  the statistics, and the report gives their number.

**Matrix.** 17 geometries × 6 skies:

| Parameter | Values |
|---|---|
| Ground slope / panel tilt | 0–30° / 30–90° |
| Panel centre height h̄ | 0.5–1.5 |
| Shading angle | 5–25° |
| Slope azimuth | ±45° |
| Skies | clear (sun low in front, high in front, along the rows, behind); hazy (paper Fig. 15); overcast |

20 sensors per side along the panel height; slopevf is evaluated at the same positions.
