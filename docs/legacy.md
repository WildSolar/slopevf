# Legacy mode and differences to the MATLAB implementation

`Settings.matlab()` reproduces the original MATLAB implementation
(https://github.com/WildSolar/General_PV_Irradiation_Model), the code used for the publication. The test suite checks it against reference output generated with
MATLAB (`tests/matlab/generate_reference.m`): the multiplication tables agree to
machine precision and the irradiances to about 1e-11 W/m².

The default mode (`Settings()`) corrects the following points.

## Model and implementation corrections

| # | MATLAB implementation | Default (corrected) mode |
|---|---|---|
| 1 | `skyLuminanceBatch.m` passes the solar elevation in radians to `kasten.m`, which expects sin(elevation). The air mass is too small at mid elevations, and above 1 rad (57.3°) the intermediate value becomes complex and is reset to 1 by `airmass(airmass<1) = 1`. | Kasten & Young (1989) air mass with the correct argument. |
| 2 | Perez bin 1: `c = exp(Δ·(c1 + c2·Z)^c3) − 1`. | `c = exp((Δ·(c1 + c2·Z))^c3) − 1`, as in Perez et al. (1993). |
| 3 | Perez coefficient table: b4 of bin 1 is `-1.233`, c3 of bin 6 is `-7.1407`, d4 of bin 7 is `0.2652`. | `-0.1233`, `-7.1406`, `0.2656` (Radiance `gendaylit.c`). |
| 4 | Cosine factors of the sky cells are not clipped. For the far ground, which has its own orientation, sky cells behind its plane (e.g. a low winter sun with β_fg = 20°, α_fg = 160°) contribute *negative* irradiance. | `max(0, ŝ·n)` for all surfaces. |
| 5 | `VF_farground.m`: prefactor 1/(2π) instead of 1/π, `sec(φ)` instead of `sec²(φ)`. Eqn. 41 of the paper additionally lacks the cos σ Jacobian of the solid angle. | Exact closed form, see below. |
| 6 | Near-ground view factor by midpoint integration of eqn. 26 (`n_d`, `n_zeta`), with `n_d` reduced for distant strips and the number of rows limited to ±`maxRowsConsidered`. | Exact closed form, see below. |
| 7 | Panel sampled at `n_chi + 1` nodes including both edges (χ = −0.5 has an infinite view area); panel average is the node mean. Shaded fraction approximated by a ramp of one node spacing. | `n_chi` elements, exact element-averaged lit fraction, view factors at element centres; panel average is exact for the midpoint rule. |
| 8 | Sky cell solid angle approximated as cos(el_mid)·Δel·Δaz; number of azimuth cells per band from cos of the band's upper edge. | Exact solid angles Δaz·(sin el_hi − sin el_lo). |
| 9 | Diffuse and beam normalised separately and then rescaled to GHI as a whole. Negative Perez radiance clipped only if all values exceed −100. | Diffuse normalised to DHI, beam to DNI·cos Z (sum = GHI). Negative Perez radiance always clipped. |
| 10 | DHI < 0 (non-physical input) replaced by 0.05 W/m². | Clipped to 0 with a warning. |
| 11 | Perez sky brightness Δ computed with a constant extraterrestrial irradiance of 1361 W/m². | Date-dependent extraterrestrial irradiance (Spencer 1971) when the weather has a DatetimeIndex or a `day_of_year` column, otherwise the solar constant (`Settings.solar_constant`). Effect on yearly totals < 0.05%, on single hours up to about 1.3%. |
| 12 | Perez model without range limits. Very bright overcast skies (Δ > 0.6) give a gradation parameter b > 0, and the radiance near the horizon diverges (vertical planes received up to 30 times the GHI in the Totalp data). | Δ clamped to [0.01, 0.6] and ε to [1, 12.01) as in Radiance's gendaylit; b capped at 0. |
| 13 | No correction for dark intermediate skies: for 1.065 < ε < 2.8 and Δ < 0.2 the zenith becomes almost black and the horizon up to 3.5 times brighter than the mean sky radiance. | Δ ≥ 0.2 when computing a..e, as in gendaylit. Davos TMY: 14 of ~4'400 daylight hours change (sun at 3–6°), annual totals by < 0.01%. |

Since 2026-10-22 the default sky is the `perezsky` package
(`C:\Users\wimm\Documents\packages\PerezSky`), which implements items 1–3 and 8–13 and is
validated against gendaylit and rtrace. The legacy sky remains in `slopevf.sky`.

## Closed-form view factors

Every ground strip and every boundary in the model is parallel to the rows
(z'), so the problem is two-dimensional. Integrating eqn. 26 over z' analytically,

    ∫ dz / (r² + z²)² = π / (2 r³),

reduces the view factor from a point P(χ) to an infinitely long strip to

    F = ½ |sin θ₂ − sin θ₁|,

where θ₁ and θ₂ are the angles, measured in the cross-section, between the panel
normal and the lines of sight to the two strip edges. The far ground is the
wedge between the horizontal and the line of sight to the upper edge of the row
in front; both boundaries contain the z' direction, so the same formula applies.
For a vertical panel looking at the whole half space in front of it, this gives
the expected F = ½. The tests verify the closed form against direct numerical
integration of eqn. 26 and of the far-ground solid angle, and the aggregated
lit-ground view factor against brute-force ray tracing.

Lit strips are summed exactly within `ground_extent` (default 50 panel heights)
of the point where the panel plane meets the ground. Beyond that, the remaining
part of the visible window is weighted by the lit fraction of the ground. The
error of this approximation is below 2·10⁻⁴ in the view factor.

## Paper typos noticed during the port

* Eqn. for d̄_b,max (section 2.4): stray "Y" before sin(β_p).
* Eqns. for cases (i) and (iv): the conditions χ ≥ η should read χ + ½ ≥ η.
* Section 3.1 calls Ω_fg the far-ground *apparent* albedo, while section 2.6
  states it is specifically not an apparent albedo.
* Caption of the shading-angle-equivalence figure says "midday", the text "mid-morning".
* Eqn. 41 lacks the cos σ Jacobian (see item 5).
