"""Local horizon: masking of the sky and radiance of the terrain above the horizontal.

The horizon is given as elevation per azimuth only, as provided by typical horizon tools.
Sky elements below the horizon receive the radiance of the terrain instead of the sky, and
the solar beam is blocked when the sun is behind the horizon. Since the whole model is
linear in the radiance of the sky elements, the panels, the near ground and the far ground
are all affected consistently.

Without information on the terrain itself, its radiance is modelled as Lambertian with
albedo ``albedo_terrain`` and one of two irradiation models:

* ``"facing"`` (default): in each direction, a terrain slope of tilt ``terrain_tilt`` (default
  30 deg) facing the array, irradiated by the beam (if the sun is not hidden behind the
  horizon for the array), the isotropic diffuse sky and the reflection of its own
  surroundings. Terrain visible from the array necessarily faces the array to some degree;
  compared with physically irradiated terrain slopes in Radiance, this model deviates by at
  most about 2 % (benchmarks/raytrace/horizon_check.py).
* ``"horizontal"``: the terrain receives the global horizontal irradiation,
  L = albedo * GHI / pi, analogous to the far-ground treatment. It ignores that terrain faces
  are in their own shadow when the sun is behind the horizon, and that slopes facing the
  array can be much brighter than horizontal terrain (deviations up to about 14 %).

Only the array and its near ground see the horizon. Distant terrain (the terrain above the
horizon as well as the far ground below the horizontal) is irradiated under free-horizon
conditions, since its own horizon is unknown.

Negative horizon elevations (terrain below the horizontal, e.g. due to the curvature of the
earth at high altitude) are treated as zero, i.e. the sky is not extended below the horizontal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import direction_vector, surface_normal


@dataclass(frozen=True)
class Horizon:
    """Horizon profile and terrain description.

    Parameters
    ----------
    azimuth : array_like
        Azimuths [deg]; 0 = south, positive towards west (paper convention).
    elevation : array_like
        Horizon elevation at these azimuths [deg].
    terrain : {"horizontal", "facing"}
        Model of the irradiation on the terrain seen below the horizon (see module docstring).
    terrain_tilt : float
        Tilt of the terrain slopes for ``terrain="facing"`` [deg].
    """

    azimuth: np.ndarray
    elevation: np.ndarray
    terrain: str = "facing"
    terrain_tilt: float = 30.0
    _az: np.ndarray = field(init=False, repr=False)
    _el: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        az = np.asarray(self.azimuth, dtype=float).ravel()
        el = np.asarray(self.elevation, dtype=float).ravel()
        if az.shape != el.shape or az.size < 2:
            raise ValueError("azimuth and elevation must be 1-D arrays of equal length (>= 2)")
        if self.terrain not in ("horizontal", "facing"):
            raise ValueError(f"unknown terrain model {self.terrain!r}")
        az = (az + 180.0) % 360.0 - 180.0
        order = np.argsort(az)
        object.__setattr__(self, "_az", np.radians(az[order]))
        object.__setattr__(self, "_el", np.radians(np.maximum(el[order], 0.0)))

    @classmethod
    def from_file(cls, path, azimuth_convention: str = "south", **kwargs) -> "Horizon":
        """Read a two-column text file (azimuth, elevation in degrees), e.g. a ``.hor`` file.

        ``azimuth_convention``: ``"south"`` (0 = south, positive towards west) or ``"north"``
        (0 = north, positive towards east, as used by pvlib and most GIS tools).
        """
        data = np.loadtxt(Path(path), comments=("#", "%"))
        az, el = data[:, 0], data[:, 1]
        if azimuth_convention == "north":
            az = az - 180.0
        elif azimuth_convention != "south":
            raise ValueError("azimuth_convention must be 'south' or 'north'")
        return cls(az, el, **kwargs)

    def elevation_at(self, azimuth):
        """Horizon elevation [rad] at azimuths [rad] (periodic linear interpolation, >= 0)."""
        a = (np.asarray(azimuth, dtype=float) + np.pi) % (2 * np.pi) - np.pi
        return np.interp(a, self._az, self._el, period=2 * np.pi)

    def masked(self, elevation, azimuth):
        """True where a direction lies below the horizon."""
        return np.asarray(elevation) < self.elevation_at(azimuth)

    def terrain_radiance(self, elevation, azimuth, albedo, dni, ghi, sun_elevation, sun_azimuth):
        """Radiance [W/m^2/sr] of the terrain in the given directions, shape (n_dirs, n_t).

        ``albedo``, ``dni``, ``ghi`` and the sun position are per time step (n_t,).
        """
        albedo, dni, ghi = (np.atleast_1d(np.asarray(v, dtype=float)) for v in (albedo, dni, ghi))
        n = np.size(elevation)
        if self.terrain == "horizontal":
            return np.broadcast_to(albedo * ghi / np.pi, (n, ghi.size)).copy()
        bt = np.radians(self.terrain_tilt)
        a_t = np.asarray(azimuth, dtype=float) + np.pi                   # slopes face the array
        n_t = surface_normal(np.full(a_t.shape, bt), a_t)                 # (3, n)
        s = direction_vector(np.atleast_1d(sun_elevation), np.atleast_1d(sun_azimuth))   # (n_t, 3)
        visible = ~self.masked(sun_elevation, sun_azimuth)
        dhi = np.maximum(ghi - dni * np.sin(sun_elevation), 0.0)
        beam = (dni * visible)[None, :] * np.maximum(n_t.T @ s.T, 0.0)
        diffuse = dhi * (1 + np.cos(bt)) / 2 + albedo * ghi * (1 - np.cos(bt)) / 2
        return albedo[None, :] * (beam + diffuse[None, :]) / np.pi
