"""Discretised sky hemisphere and sky radiance distribution (Perez + Buie).

The sky radiance ``L`` [W/m^2/sr] is computed on a fixed grid of sky cells for
many time steps at once (shape ``(n_cells, n_times)``).

Two implementations are provided:

* the default one is the ``perezsky`` package (Perez 1993 as published and as in
  Radiance's gendaylit, Kasten-Young air mass, range limits, exact solid angles, energy
  closure of the diffuse (DHI) and beam (DNI cos Z) parts separately). This module only
  converts between perezsky's conventions (degrees, azimuth from north, clockwise) and
  slopevf's (radians, azimuth from south, positive towards west);
* ``legacy``, which reproduces ``hemispherePartition.m``, ``skyLuminanceBatch.m``
  and ``totalSkyLuminance.m`` of the original MATLAB implementation, including
  their known deviations (see ``docs/legacy.md``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import perezsky

from .geometry import direction_vector

# Number of azimuthal cells at the horizon for the named sky resolutions
# (identical to hemispherePartition.m); all presets use 0.5 deg elevation bands.
SKY_PRESETS = dict(perezsky.grid.RESOLUTIONS)

# Perez et al. (1993) coefficients [bin, (a, b, c, d, e), (x1, x2, x3, x4)] as in gendaylit.c
PEREZ_COEFFICIENTS = perezsky.LUMINANCE_1993

# Coefficients exactly as in skyLuminanceBatch.m (three transcription differences).
PEREZ_COEFFICIENTS_LEGACY = PEREZ_COEFFICIENTS.copy()
PEREZ_COEFFICIENTS_LEGACY[0, 1, 3] = -1.233
PEREZ_COEFFICIENTS_LEGACY[5, 2, 2] = -7.1407
PEREZ_COEFFICIENTS_LEGACY[6, 3, 3] = 0.2652

PEREZ_EPSILON_BINS = np.array([1.0, 1.065, 1.230, 1.5, 1.95, 2.8, 4.5, 6.2])
SOLAR_CONSTANT = 1361.0  # W/m^2


@dataclass(frozen=True)
class SkyGrid:
    """A discretised sky hemisphere.

    Attributes
    ----------
    elevation, azimuth : ndarray
        Cell-centre directions [rad]; azimuth from south, positive towards west.
    solid_angle : ndarray
        Solid angle of each cell [sr].
    vec : ndarray
        (n, 3) unit vectors towards the cell centres (x = west, y = south, z = up).
    legacy : bool
        True for the MATLAB-identical grid.
    perez : perezsky.SkyGrid
        The same grid in perezsky's conventions (None for the legacy grid).
    """

    elevation: np.ndarray
    azimuth: np.ndarray
    solid_angle: np.ndarray
    vec: np.ndarray
    legacy: bool = False
    perez: object = None

    @property
    def size(self) -> int:
        return self.elevation.size

    @property
    def horizontal_weight(self) -> np.ndarray:
        """Solid angle times cos(zenith); sum(L * w) is the horizontal irradiance."""
        return self.solid_angle * np.sin(self.elevation)


def _azimuth_from_north(azimuth):
    """slopevf azimuth [rad, from south, positive towards west] -> perezsky [deg, from north, clockwise]."""
    return np.degrees(np.asarray(azimuth, dtype=float)) + 180.0


def make_sky_grid(resolution="medium") -> SkyGrid:
    """Sky grid of 0.5 deg elevation bands with near-constant cell size (perezsky's grid).

    ``resolution`` is a preset name (see ``SKY_PRESETS``) or a tuple
    ``(band_width_deg, n_azimuth_at_horizon)``. Solid angles are exact.
    """
    g = perezsky.make_sky_grid(resolution)
    el = np.radians(g.elevation)
    az = np.radians(g.azimuth - 180.0)          # from south, positive towards west, in (-pi, pi)
    return SkyGrid(el, az, g.solid_angle, direction_vector(el, az), legacy=False, perez=g)


# --- MATLAB-identical helpers ------------------------------------------------------

def _matlab_round(x):
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def _matlab_sind(x):
    x = np.asarray(x, dtype=float)
    n = _matlab_round(x / 90.0)
    r = np.pi / 180 * (x - n * 90.0)
    m = np.mod(n, 4)
    return np.select([m == 0, m == 1, m == 2], [np.sin(r), np.cos(r), -np.sin(r)], -np.cos(r))


def _matlab_cosd(x):
    x = np.asarray(x, dtype=float)
    n = _matlab_round(x / 90.0)
    r = np.pi / 180 * (x - n * 90.0)
    m = np.mod(n, 4)
    return np.select([m == 0, m == 1, m == 2], [np.cos(r), -np.sin(r), -np.cos(r)], np.sin(r))


@dataclass(frozen=True)
class LegacySkyGrid(SkyGrid):
    """Sky grid identical to hemispherePartition.m (with its auxiliary arrays)."""

    elevation_deg: np.ndarray = None
    azimuth_deg: np.ndarray = None
    area_sin_elev: np.ndarray = None   # sind(el) * area
    zenith: np.ndarray = None          # (90 - el)/180*pi
    pos: np.ndarray = None             # (n, 3) [-sind(az)cosd(el), -cosd(az)cosd(el), sind(el)]

    @property
    def horizontal_weight(self) -> np.ndarray:
        return self.solid_angle * np.sin(self.elevation)


def make_legacy_sky_grid(precision="high") -> LegacySkyGrid:
    """Reproduce hemispherePartition.m."""
    n_1 = SKY_PRESETS[precision]
    elev_disc = 0.5
    lower = np.arange(0.0, 90.0, elev_disc)
    el_l, az_l, az_lo, az_hi = [], [], [], []
    for lo in lower:
        n_loco = int(np.ceil(n_1 * _matlab_cosd(lo + 0.5))) + 1
        az_vec = np.linspace(0.0, 360.0, n_loco) + 0.25
        el_l.append(np.full(n_loco - 1, lo + elev_disc / 2))
        az_l.append((az_vec[:-1] + az_vec[1:]) / 2)
        az_lo.append(az_vec[:-1])
        az_hi.append(az_vec[1:])
    el_deg, az_deg, a_lo, a_hi = map(np.concatenate, (el_l, az_l, az_lo, az_hi))
    lower_el = el_deg - elev_disc / 2
    upper_el = lower_el + elev_disc
    area = (_matlab_cosd((lower_el + upper_el) / 2) * (upper_el - lower_el) / 180 * np.pi
            * (a_hi - a_lo) / 180 * np.pi)
    sin_el, cos_el = _matlab_sind(el_deg), _matlab_cosd(el_deg)
    pos = np.stack([-_matlab_sind(az_deg) * cos_el, -_matlab_cosd(az_deg) * cos_el, sin_el], axis=-1)
    elevation = el_deg / 180 * np.pi
    azimuth = (np.mod(az_deg + 180, 360) - 180) / 180 * np.pi
    return LegacySkyGrid(
        elevation=elevation, azimuth=azimuth, solid_angle=area,
        vec=direction_vector(elevation, azimuth), legacy=True,
        elevation_deg=el_deg, azimuth_deg=az_deg, area_sin_elev=sin_el * area,
        zenith=(90 - el_deg) / 180 * np.pi, pos=pos)


# --- model components ---------------------------------------------------------------

# perezsky's implementations (re-exported for convenience)
extraterrestrial_irradiance = perezsky.extraterrestrial_irradiance
buie_sunshape = perezsky.buie_sunshape   # identical to BuieSolarDistribution.m


def _kasten_legacy(mu0):
    """kasten.m called with the solar elevation in radians (as in the MATLAB code).

    The function expects sin(elevation); for arguments > 1 the MATLAB code
    produces complex air masses. Reproduced with complex arithmetic (the real
    part of all final results is independent of the branch chosen).
    """
    mu0 = np.asarray(mu0, dtype=complex)
    gam = np.arcsin(mu0) * (180 / np.pi)
    am = 1.0 / (mu0 + 0.50572 * (gam + 6.07995) ** -1.6364)
    am = np.where(am.real < 1, 1.0 + 0j, am)
    return am


def _perez_parameters_legacy(dhi, dni, zenith, airmass):
    """Perez parameters a..e as in skyLuminanceBatch.m: mistyped coefficients, exponent of the
    category-1 formula for c misplaced, no range limits. Returns (5, n_times), complex if
    ``airmass`` is complex."""
    coefficients = PEREZ_COEFFICIENTS_LEGACY
    dhi, dni, zenith = (np.asarray(v, dtype=float) for v in (dhi, dni, zenith))
    kz3 = 1.041 * zenith ** 3
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = ((dhi + dni) / dhi + kz3) / (1 + kz3)
    eps = np.where(dhi > 0, eps, np.inf)
    bins = np.clip(np.sum(eps[:, None] > PEREZ_EPSILON_BINS, axis=1), 1, 8) - 1
    delta = airmass * dhi / SOLAR_CONSTANT
    x = coefficients[bins]  # (n, 5, 4)
    z = zenith[:, None]
    par = x[:, :, 0] + x[:, :, 1] * z + delta[:, None] * (x[:, :, 2] + x[:, :, 3] * z)
    par = par.T.copy()
    b1 = bins == 0
    if np.any(b1):
        c, d = coefficients[0, 2], coefficients[0, 3]
        zz, dd = zenith[b1], delta[b1]
        par[2, b1] = np.exp(dd * (c[0] + c[1] * zz) ** c[2]) - 1
        par[3, b1] = -np.exp(dd * (d[0] + d[1] * zz)) + d[2] + dd * d[3]
    return par


def perez_luminance(par, cos_zenith_cells, gamma, cos_gamma=None):
    """Relative sky luminance lv for cells (rows) and time steps (columns) (legacy path).

    The gradation term depends only on the cell zenith angle and is evaluated
    once per distinct elevation (band) instead of once per cell.
    """
    a, b, c, d, e = par[:, None, :]
    cz, inverse = np.unique(cos_zenith_cells, return_inverse=True)
    gradation = (1 + a * np.exp(b / cz[:, None]))[inverse.ravel()]
    if cos_gamma is None:
        cos_gamma = np.cos(gamma)
    return gradation * (1 + c * np.exp(d * gamma) + e * cos_gamma ** 2)


@dataclass
class SkyRadiance:
    """Radiance on the sky grid for a set of time steps."""

    L: np.ndarray            # (n_cells, n_times) [W/m^2/sr]
    dhi: np.ndarray          # diffuse horizontal irradiance used [W/m^2]


def sky_radiance(grid: SkyGrid, dni, ghi, sun_azimuth, sun_elevation, *, sun_model="buie",
                 csr=0.3, circumsolar_half_angle=6.0, day_of_year=None,
                 solar_constant=SOLAR_CONSTANT) -> SkyRadiance:
    """Sky radiance for time steps with the sun above the horizon.

    Parameters
    ----------
    dni, ghi : array_like
        Direct normal and global horizontal irradiance [W/m^2].
    sun_azimuth, sun_elevation : array_like
        Sun position [rad]; azimuth from south, positive towards west.
    sun_model : {"buie", "point"}
        ``"buie"``: the beam is distributed over the circumsolar cone (half angle
        ``circumsolar_half_angle`` deg) following the Buie sunshape, replacing the
        Perez radiance there (as in the paper). ``"point"``: the returned
        radiance contains only the diffuse sky; the beam is treated separately as
        a point source by the caller.
    day_of_year, solar_constant
        Extraterrestrial irradiance in the Perez sky brightness: ``solar_constant``
        [W/m^2] corrected for the Earth-Sun distance if ``day_of_year`` is given
        (see :func:`extraterrestrial_irradiance`). Ignored for the legacy grid, which
        reproduces the MATLAB code (constant 1361).

    The default implementation is :func:`perezsky.perez_sky` with its defaults (range limits
    and gendaylit's brightness correction).
    """
    if grid.legacy:
        return _sky_radiance_legacy(grid, dni, ghi, sun_azimuth, sun_elevation)
    dni, ghi, az, el = (np.atleast_1d(np.asarray(v, dtype=float)) for v in
                        (dni, ghi, sun_azimuth, sun_elevation))
    zen = 90.0 - np.degrees(el)
    dhi = ghi - dni * np.sin(el)
    if np.any(dhi < -1.0):
        warnings.warn(f"{np.sum(dhi < -1.0)} time step(s) with DNI*cos(Z) > GHI; "
                      "the diffuse irradiance is set to 0 for these.", stacklevel=3)
    dhi = np.maximum(dhi, 0.0)
    grid_p = grid.perez if grid.perez is not None else perezsky.SkyGrid(
        np.degrees(grid.elevation), np.degrees(grid.azimuth) + 180.0, grid.solid_angle,
        perezsky.direction_vector(np.degrees(grid.elevation), np.degrees(grid.azimuth) + 180.0))
    sky = perezsky.perez_sky(dni, zen, _azimuth_from_north(az), dhi=dhi, grid=grid_p,
                             day_of_year=day_of_year, solar_constant=solar_constant,
                             sun_model=sun_model, csr=csr,
                             circumsolar_half_angle=circumsolar_half_angle)
    return SkyRadiance(sky.L, dhi)


def _sky_radiance_legacy(grid: LegacySkyGrid, dni, ghi, sun_azimuth, sun_elevation) -> SkyRadiance:
    """Vectorised reproduction of totalSkyLuminance.m / skyLuminanceBatch.m."""
    dni, ghi, az, el = (np.atleast_1d(np.asarray(v, dtype=float)) for v in
                        (dni, ghi, sun_azimuth, sun_elevation))
    zenith = np.pi / 2 - el
    dhi = ghi - dni * np.cos(zenith)
    if np.any(dhi < 0):
        warnings.warn("Non-Physical Irradiation Condition detected! (DHI < 0 set to 0.05, "
                      "as in the MATLAB code)", stacklevel=3)
    dhi = np.where(dhi < 0, 0.05, dhi)

    elevation = np.pi / 2 - zenith
    par = _perez_parameters_legacy(dhi, dni, zenith, _kasten_legacy(elevation))
    sunpos = np.stack([-np.sin(az) * np.cos(elevation), -np.cos(az) * np.cos(elevation),
                       np.sin(elevation)])                                    # (3, n_t)
    p = grid.pos
    dot = p[:, 0:1] * sunpos[0] + p[:, 1:2] * sunpos[1] + p[:, 2:3] * sunpos[2]
    gamma = np.arccos(np.clip(dot, -1.0, 1.0))
    lv = perez_luminance(par, np.cos(grid.zenith), gamma)

    phi, dist = buie_sunshape(0.3, 100, 100)
    inside = ~(gamma > 6 / 180 * np.pi)
    rad_direct = np.where(inside, np.interp(gamma, phi * 1e-3, dist), 0.0)
    total_direct = np.sum(rad_direct * grid.solid_angle[:, None] * np.cos(gamma), axis=0)
    lv = np.where(inside, 0.0, lv)
    total_diffuse = np.sum(lv * grid.area_sin_elev[:, None], axis=0)
    lv_scale = dhi / total_diffuse * lv
    # MATLAB: if min(LV_scale) > -1e2, LV_scale = max(LV_scale, 0). For complex
    # columns MATLAB compares magnitudes, which leaves the values unchanged.
    is_complex = np.any(np.imag(lv_scale) != 0, axis=0)
    clip = (~is_complex) & (np.min(np.real(lv_scale), axis=0) > -1e2)
    lv_scale = np.where(clip, np.maximum(np.real(lv_scale), 0.0), lv_scale)
    rad_direct = dni / total_direct * rad_direct
    rmap = lv_scale + rad_direct
    total_global = np.sum(rmap * grid.area_sin_elev[:, None], axis=0)
    rmap = ghi / total_global * rmap
    return SkyRadiance(rmap, dhi)
