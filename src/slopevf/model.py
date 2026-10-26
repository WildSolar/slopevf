"""Top-level simulation: geometry tables x sky radiance -> irradiance distribution on the panel.

For each array configuration, everything that depends only on geometry is
computed once:

* the multiplication tables T (profile angle x panel position), and
* sparse matrices W mapping sky cells onto the profile-angle grid (linear
  interpolation weights times solid angle times the cosine factor of the
  respective surface).

The irradiance for many time steps is then ``T.T @ (W @ L)`` with the sky
radiance ``L`` (cells x time steps), computed in chunks of time steps. The sky
radiance is computed once per chunk and shared by all arrays.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse

from .geometry import Array, FarGround, Geometry, direction_vector, profile_angle, surface_normal
from .horizon import Horizon
from .reflection import PanelGlass
from .sky import SkyGrid, make_legacy_sky_grid, make_sky_grid, sky_radiance
from .tables import Tables, build_tables, build_tables_legacy, direct_fractions, reflected_fractions
from .viewfactors import panel_view_factors

COMPONENTS = ("front_direct", "front_nearground", "front_farground", "front_reflected", "front_diffuse",
              "back_direct", "back_nearground", "back_reflected", "back_diffuse")


@dataclass(frozen=True)
class Settings:
    """Numerical and modelling options.

    Parameters
    ----------
    n_chi : int
        Number of panel elements (legacy: number of intervals; results are given
        at the ``n_chi + 1`` nodes as in the MATLAB code).
    n_alpha : int
        Number of profile-angle intervals of the multiplication table over [0, pi].
    sky : str or tuple
        Sky grid resolution: ``"low"``, ``"medium"``, ``"high"``, ``"ultra"`` or
        ``(band_width_deg, n_azimuth_at_horizon)``.
    sun_model : {"point", "buie"}
        Beam treatment. ``"point"`` (default): the beam is a point source with
        exact direct shading at the sun's profile angle; the circumsolar part of
        the Perez model remains in the diffuse sky. This allows much coarser sky
        grids (``"low"`` is sufficient). ``"buie"``: the beam is spread over the
        circumsolar cone following the Buie sunshape, as in the paper (see
        :func:`slopevf.sky.sky_radiance`).
    circumsolar_ratio, circumsolar_half_angle : float
        Buie sunshape CSR and half angle of the circumsolar cone [deg].
    solar_constant : float
        Extraterrestrial normal irradiance at mean Earth-Sun distance [W/m^2], used in
        the Perez sky brightness. It is corrected for the day of the year if the weather
        has a DatetimeIndex or a ``day_of_year`` column (otherwise used as is). Not used
        in legacy mode, which keeps the constant 1361 W/m^2 of the MATLAB code.
    ground_extent : float
        Range (in panel heights) over which lit ground strips are summed exactly.
    finite_rows : (int, int) or None
        Finite array ``(n_front, n_back)``: the evaluated row has n_front rows in front of it
        (downhill) and n_back rows behind it (uphill); e.g. (2, 0) for the last row of a
        three-row array, (0, 2) for the first. None (default): infinite rows. Only these rows
        cast shadows on the ground and reflect light onto it; beyond the array the ground is
        unshaded. A side without an adjacent row (front if n_front = 0, back if n_back = 0)
        faces away from the array: it is not shaded by another row, sees the sky down to the
        horizontal (front) or the slope (back), the ground out to infinity and, facing
        downhill, the terrain below the horizontal as far ground; it receives no reflection
        off panels.
    chunk_size : int
        Number of time steps processed at once (memory ~ 8 * n_cells * chunk_size bytes).
    legacy : bool
        Reproduce the original MATLAB implementation, including its
        known deviations. Use :meth:`Settings.matlab` to get matching defaults.
    n_d, n_zeta, n_phi, n_sig, max_rows : int
        Legacy-only numerical integration parameters (see example_script.m).
    """

    n_chi: int = 100
    n_alpha: int = 720
    sky: object = "medium"
    sun_model: str = "point"
    circumsolar_ratio: float = 0.3
    circumsolar_half_angle: float = 6.0
    solar_constant: float = 1361.0
    ground_extent: float = 50.0
    finite_rows: tuple[int, int] | None = None
    chunk_size: int = 256
    legacy: bool = False
    n_d: int = 50
    n_zeta: int = 50
    n_phi: int = 100
    n_sig: int = 100
    max_rows: int = 20

    @classmethod
    def matlab(cls, **kwargs) -> "Settings":
        """Legacy mode with the defaults of example_script.m (n_sigma = n_alpha = 180)."""
        base = dict(legacy=True, sun_model="buie", n_chi=100, n_alpha=180, sky="high", n_d=50, n_zeta=50,
                    n_phi=100, n_sig=100, max_rows=20)
        base.update(kwargs)
        return cls(**base)


@dataclass
class Result:
    """Irradiance on the panel front and back side [W/m^2] for each time step and position.

    The component arrays have shape (n_times, n_chi). ``chi`` runs from the
    lower (-0.5) to the upper (+0.5) panel edge.
    """

    chi: np.ndarray
    index: pd.Index
    front_direct: np.ndarray
    front_nearground: np.ndarray
    front_farground: np.ndarray
    back_direct: np.ndarray
    back_nearground: np.ndarray
    ghi_check: np.ndarray
    front_reflected: np.ndarray = None   # specular reflection off the adjacent row's glass
    back_reflected: np.ndarray = None
    front_diffuse: np.ndarray = None     # diffuse reflection off the adjacent row's face
    back_diffuse: np.ndarray = None
    array: Array = None
    settings: Settings = field(default=None, repr=False)

    def __post_init__(self):
        for c in ("front_reflected", "back_reflected", "front_diffuse", "back_diffuse"):
            if getattr(self, c) is None:
                setattr(self, c, np.zeros_like(self.front_direct))

    @property
    def front(self) -> np.ndarray:
        return (self.front_direct + self.front_nearground + self.front_farground + self.front_reflected
                + self.front_diffuse)

    @property
    def back(self) -> np.ndarray:
        return self.back_direct + self.back_nearground + self.back_reflected + self.back_diffuse

    def _series(self, values, name):
        return pd.Series(values, index=self.index, name=name)

    @property
    def front_mean(self) -> pd.Series:
        return self._series(self.front.mean(axis=1), "front_mean")

    @property
    def back_mean(self) -> pd.Series:
        return self._series(self.back.mean(axis=1), "back_mean")

    @property
    def front_min(self) -> pd.Series:
        return self._series(self.front.min(axis=1), "front_min")

    @property
    def back_min(self) -> pd.Series:
        return self._series(self.back.min(axis=1), "back_min")

    def summary(self) -> pd.DataFrame:
        """Panel-average and minimum irradiance per time step, plus the component averages."""
        df = pd.DataFrame({c: getattr(self, c).mean(axis=1) for c in COMPONENTS}, index=self.index)
        df.insert(0, "back_min", self.back.min(axis=1))
        df.insert(0, "front_min", self.front.min(axis=1))
        df.insert(0, "back_mean", self.back.mean(axis=1))
        df.insert(0, "front_mean", self.front.mean(axis=1))
        df["ghi_check"] = self.ghi_check
        return df

    def plot(self, time=0, ax=None, **kwargs):
        """Plot the irradiance distribution of one time step (index position or label)."""
        from .plotting import plot_distribution
        return plot_distribution(self, time, ax=ax, **kwargs)


class ArrayModel:
    """Geometry-only precomputation for one array (tables and sky-cell weights)."""

    def __init__(self, array: Array, far_ground: FarGround, grid: SkyGrid, settings: Settings,
                 glass: PanelGlass | None = None):
        self.array = array
        self.glass = glass
        self.far_ground = far_ground
        self.settings = settings
        self.geom = Geometry(array)
        s = settings
        if s.legacy and s.finite_rows is not None:
            raise ValueError("finite_rows is not available in legacy mode")
        if s.legacy and glass is not None:
            raise ValueError("panel glass is not supported in legacy mode")
        if s.legacy and array.thickness > 0:
            raise ValueError("panel thickness is not supported in legacy mode")
        # relative transmittance of the receiving glass (1 without glass transmission)
        self.tau = glass.transmittance if glass is not None else np.ones_like
        if s.legacy:
            self.tables: Tables = build_tables_legacy(self.geom, s.n_alpha, s.n_chi, s.n_d, s.n_zeta,
                                                      s.max_rows, s.n_phi, s.n_sig)
        else:
            # the reflected-ground view factors are only needed for reflecting glass
            self.tables = build_tables(self.geom, s.n_alpha, s.n_chi, s.ground_extent, s.finite_rows,
                                       kernel=None if glass is None else glass.view_kernel(),
                                       reflection=glass is not None and glass.reflects)

        gamma = np.radians(array.gamma)
        self.n_pf = surface_normal(self.geom.bp, gamma)
        self.n_g = surface_normal(self.geom.bg, gamma)
        self.n_fg = surface_normal(np.radians(far_ground.beta), np.radians(far_ground.azimuth))

        if s.legacy:
            el, az = grid.elevation, grid.azimuth
            with np.errstate(divide="ignore"):
                alpha = np.mod(np.arctan(np.tan(el) / np.cos(az - gamma)) + np.pi, np.pi)
            clip = lambda x: x  # noqa: E731  (MATLAB: no clipping of cosine factors)
        else:
            alpha = profile_angle(grid.elevation, grid.azimuth, gamma)
            clip = lambda x: np.maximum(x, 0.0)  # noqa: E731
        omega = grid.solid_angle
        v = grid.vec
        c_pf, c_pb = clip(v @ self.n_pf), clip(-(v @ self.n_pf))
        if glass is not None:
            c_pf, c_pb = c_pf * self.tau(c_pf), c_pb * self.tau(c_pb)
        self.W_pf = self._interp_matrix(alpha, omega * c_pf)
        self.W_pb = self._interp_matrix(alpha, omega * c_pb)
        self.W_g = self._interp_matrix(alpha, omega * clip(v @ self.n_g))
        self.w_fg = omega * clip(v @ self.n_fg)
        self.W_rb = self.W_rf = None
        if glass is not None and glass.reflects:
            # mirror and receiver are parallel: the incidence angle on the receiver equals
            # the one on the mirror (cos_m); weight = solid angle * R(cos_m) * cos_m * tau(cos_m)
            cos_front, cos_back = np.maximum(v @ self.n_pf, 0.0), np.maximum(-(v @ self.n_pf), 0.0)
            if glass.front:     # front of the uphill row reflects onto our back side
                self.W_rb = self._interp_matrix(alpha, omega * glass(cos_front) * cos_front
                                                * self.tau(cos_front))
            if glass.back:      # back of the downhill row reflects onto our front side
                self.W_rf = self._interp_matrix(alpha, omega * glass(cos_back) * cos_back
                                                * self.tau(cos_back))
            # reflected radiation reaching the ground: weight = solid angle * R * cos(r, n_g)
            self.W_rg = {}
            for mirror, use, n_m, cos_m in (("front", glass.front, self.n_pf, cos_front),
                                            ("back", glass.back, -self.n_pf, cos_back)):
                if use:
                    r3 = -v + 2 * (v @ n_m)[:, None] * n_m[None, :]          # reflected propagation
                    cos_g = np.maximum(-(r3 @ self.n_g), 0.0)
                    self.W_rg[mirror] = self._interp_matrix(alpha, omega * glass(cos_m) * cos_g * (cos_m > 0))
        # diffuse reflection: view factors from each side to the facing face of the adjacent row
        self.V = {}
        if glass is not None and glass.diffuse > 0:
            edges = np.linspace(-0.5, 0.5, self.tables.chi.size + 1)
            for side, use in (("back", glass.front), ("front", glass.back)):
                if use:
                    self.V[side] = panel_view_factors(self.geom, self.tables.chi, edges, side,
                                                      glass.view_kernel())

    def _interp_matrix(self, alpha, mult):
        n = self.tables.alpha.size - 1
        if self.settings.legacy:
            x1 = n * alpha / np.pi + 1          # applyMultiplicationTable.m (1-based)
            lo1 = np.maximum(1, np.floor(x1))
            frac, lo = x1 - lo1, (lo1 - 1).astype(int)
        else:
            x = alpha * n / np.pi
            lo = np.clip(np.floor(x), 0, n - 1).astype(int)
            frac = x - lo
        cols = np.arange(alpha.size)
        return sparse.csr_matrix(
            (np.concatenate([(1 - frac) * mult, frac * mult]),
             (np.concatenate([lo, lo + 1]), np.concatenate([cols, cols]))),
            shape=(n + 1, alpha.size))

    def far_ground_irradiance(self, L, sun=None):
        """Irradiation on the far ground (n_t,) for sky radiance ``L`` and an optional point sun."""
        e_fg = self.w_fg @ L
        if sun is not None:
            dni, el, az = sun
            e_fg = e_fg + dni * np.maximum(direction_vector(el, az) @ self.n_fg, 0.0)
        return e_fg

    def apply(self, L, albedo, albedo_fg, sun=None, e_fg=None):
        """Irradiance components for a chunk of time steps.

        ``L`` is (n_cells, n_t); ``albedo``/``albedo_fg`` are (n_t,). ``sun`` is
        ``(dni, elevation, azimuth)`` for the point-sun model, else None. ``e_fg``
        optionally overrides the irradiation on the far ground (n_t,), which by
        default is computed from ``L`` and ``sun``.
        """
        T = self.tables
        b_pf, b_pb, b_g = self.W_pf @ L, self.W_pb @ L, self.W_g @ L
        out = {
            "front_direct": b_pf.T @ T.front_direct,
            "back_direct": b_pb.T @ T.back_direct,
            "front_nearground": albedo[:, None] * (b_g.T @ T.front_ground),
            "back_nearground": albedo[:, None] * (b_g.T @ T.back_ground),
        }
        if e_fg is None:
            e_fg = self.far_ground_irradiance(L, sun)
        if sun is not None:
            # point sun: exact direct fractions at the sun's profile angle; the ground
            # view factors are smooth in alpha' and interpolated from the table
            dni, el, az = sun
            s = direction_vector(el, az)
            alpha = profile_angle(el, az, np.radians(self.array.gamma))
            fd, bd = direct_fractions(self.geom, alpha, T.chi, 1.0 / T.chi.size, self.settings.finite_rows)
            W = self._interp_matrix(alpha, np.ones_like(alpha))
            fg, bg = W.T @ T.front_ground, W.T @ T.back_ground
            c_pf, c_pb = np.maximum(s @ self.n_pf, 0.0), np.maximum(-(s @ self.n_pf), 0.0)
            i_pf = dni * c_pf * self.tau(c_pf)
            i_pb = dni * c_pb * self.tau(c_pb)
            i_g = dni * np.maximum(s @ self.n_g, 0.0)
            out["front_direct"] = out["front_direct"] + i_pf[:, None] * fd
            out["back_direct"] = out["back_direct"] + i_pb[:, None] * bd
            out["front_nearground"] = out["front_nearground"] + (albedo * i_g)[:, None] * fg
            out["back_nearground"] = out["back_nearground"] + (albedo * i_g)[:, None] * bg
        out["front_farground"] = (albedo_fg * e_fg)[:, None] * T.far_ground[None, :]
        if T.front_terrain is not None:
            # near ground irradiated by the terrain below the horizon (far-ground radiance)
            l_terrain = albedo * albedo_fg * e_fg / np.pi
            out["front_nearground"] = out["front_nearground"] + l_terrain[:, None] * T.front_terrain[None, :]
            out["back_nearground"] = out["back_nearground"] + l_terrain[:, None] * T.back_terrain[None, :]

        # specular reflection off the glass of the adjacent rows
        n_t, n_c = L.shape[1], T.chi.size
        for key, W, table, sign in (("back_reflected", self.W_rb, T.back_reflected, 1.0),
                                    ("front_reflected", self.W_rf, T.front_reflected, -1.0)):
            if W is None:
                out[key] = np.zeros((n_t, n_c))
                continue
            out[key] = (W @ L).T @ table
            if sun is not None:
                cos_m = np.maximum(sign * (s @ self.n_pf), 0.0)       # incidence on the mirror
                rb, rf = reflected_fractions(self.geom, alpha, T.chi, 1.0 / n_c, self.settings.finite_rows)
                frac = rb if key == "back_reflected" else rf
                out[key] = out[key] + (dni * self.glass(cos_m) * cos_m * self.tau(cos_m))[:, None] * frac

        # reflected radiation reaching the ground, reflected again by the ground (albedo)
        if self.W_rf is not None or self.W_rb is not None:
            Wi = self._interp_matrix(alpha, np.ones_like(alpha)) if sun is not None else None
            for mirror, W in self.W_rg.items():
                b = W @ L                                                   # (n_alpha + 1, n_t)
                if sun is not None:
                    n_m = self.n_pf if mirror == "front" else -self.n_pf
                    cos_m = np.maximum(s @ n_m, 0.0)
                    r3 = -s + 2 * (s @ n_m)[:, None] * n_m[None, :]
                    cos_g = np.maximum(-(r3 @ self.n_g), 0.0)
                    b = b + Wi.multiply((dni * self.glass(cos_m) * cos_m * cos_g)[None, :]).toarray()
                for receiver in ("front", "back"):
                    vf = T.reflected_ground[(mirror, receiver)]
                    out[f"{receiver}_reflected"] = out[f"{receiver}_reflected"] + albedo[:, None] * (b.T @ vf)
        # diffuse reflection needs the irradiance of the adjacent row: added in simulate()
        out["front_diffuse"] = np.zeros((n_t, n_c))
        out["back_diffuse"] = np.zeros((n_t, n_c))
        return out

    def neighbours(self, far_ground: FarGround, grid: SkyGrid):
        """Models of the adjacent rows whose facing face reflects diffusely onto this row:
        ``{receiving side: model or None}``, None meaning this row itself (infinite rows: all
        rows are irradiated alike). In a finite array, the adjacent row has one row more on one
        side and one less on the other."""
        out = {}
        rows = self.settings.finite_rows
        for side in self.V:
            if rows is None:
                out[side] = None
            elif (rows[1] if side == "back" else rows[0]) > 0:
                n_front, n_back = rows
                adj = (n_front + 1, n_back - 1) if side == "back" else (n_front - 1, n_back + 1)
                out[side] = ArrayModel(self.array, far_ground, grid, replace(self.settings, finite_rows=adj),
                                       self.glass)
        return out


def _column(df, names):
    for n in names:
        if n in df.columns:
            return df[n].to_numpy(dtype=float)
    return None


def prepare_weather(weather, latitude=None, longitude=None, altitude=0.0) -> pd.DataFrame:
    """Normalise weather input to a DataFrame with dni, ghi, solar_azimuth, solar_elevation.

    ``weather`` may be a DataFrame or a dict of array-likes. Column names are
    case-insensitive. Any two of ``dni``, ``ghi``, ``dhi`` are sufficient.
    Sun angles are in degrees with azimuth from south, positive towards west
    (paper convention). If they are missing, they are computed with pvlib
    from a DatetimeIndex and ``latitude``/``longitude``.
    """
    df = weather.copy() if isinstance(weather, pd.DataFrame) else pd.DataFrame(weather)
    df.columns = [str(c).lower() for c in df.columns]
    has_sun = "solar_azimuth" in df.columns and "solar_elevation" in df.columns
    if not has_sun:
        if latitude is None or longitude is None or not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("weather needs 'solar_azimuth' and 'solar_elevation' columns, or a "
                             "DatetimeIndex together with latitude and longitude")
        from .solar import solar_position
        sp = solar_position(df.index, latitude, longitude, altitude)
        df["solar_azimuth"] = sp["solar_azimuth"].to_numpy()
        df["solar_elevation"] = sp["solar_elevation"].to_numpy()
    dni, ghi, dhi = (_column(df, [n]) for n in ("dni", "ghi", "dhi"))
    if ghi is None and dni is not None and dhi is not None:
        df["ghi"] = dhi + dni * np.sin(np.radians(df["solar_elevation"].to_numpy(dtype=float)))
    elif dni is None and ghi is not None and dhi is not None:
        sin_el = np.sin(np.radians(df["solar_elevation"].to_numpy(dtype=float)))
        with np.errstate(divide="ignore", invalid="ignore"):
            df["dni"] = np.where(sin_el > 0, (ghi - dhi) / sin_el, 0.0)
    elif dni is None or ghi is None:
        raise ValueError("weather needs at least two of the columns 'dni', 'ghi', 'dhi'")
    return df


def _day_of_year(df):
    """Day of the year from a 'day_of_year' column or a DatetimeIndex, else None."""
    if "day_of_year" in df.columns:
        return df["day_of_year"].to_numpy(dtype=float)
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index.dayofyear.to_numpy(dtype=float)
    return None


def _per_time(value, df, column, n, default=None):
    if value is None:
        if column in df.columns:
            return df[column].to_numpy(dtype=float)
        if default is None:
            raise ValueError(f"provide '{column}' as argument or weather column")
        value = default
    value = np.asarray(value, dtype=float)
    return np.broadcast_to(value, (n,)).astype(float)


def simulate(arrays: Array | Sequence[Array], weather, *, albedo=None, far_ground: FarGround | None = None,
             albedo_fg=None, horizon: Horizon | None = None, albedo_terrain=None,
             panel_glass: PanelGlass | None = None,
             settings: Settings | None = None, latitude=None, longitude=None,
             altitude=0.0, progress: bool = False):
    """Simulate the irradiance distribution on the panels of one or several arrays.

    Parameters
    ----------
    arrays : Array or sequence of Array
        Array configuration(s). The sky radiance is computed once and shared.
    weather : DataFrame or dict
        Columns ``dni``, ``ghi`` (or ``dhi``) [W/m^2], ``solar_azimuth``,
        ``solar_elevation`` [deg] (see :func:`prepare_weather`). Optional columns
        ``albedo`` and ``albedo_fg`` are used if the arguments are not given. The
        day of the year (for the extraterrestrial irradiance) is taken from a
        ``day_of_year`` column or a DatetimeIndex.
    albedo : float or array_like
        Near-ground broadband albedo (scalar or per time step).
    far_ground : FarGround, optional
        Orientation of the far ground. Without it, no far-ground reflection is added.
    albedo_fg : float or array_like
        Far-ground albedo (required if ``far_ground`` is given and the weather
        has no ``albedo_fg`` column).
    horizon : Horizon, optional
        Local horizon (see :class:`slopevf.horizon.Horizon`). Sky elements below it receive
        the radiance of the terrain, and the beam is blocked when the sun is behind it.
        The weather data are assumed to be free-horizon (unobstructed) values.
    albedo_terrain : float or array_like
        Albedo of the terrain seen below the horizon (scalar or per time step). Defaults
        to an ``albedo_terrain`` weather column, else to the far-ground albedo.
    panel_glass : PanelGlass, optional
        Specularly reflecting module glass (see :mod:`slopevf.reflection`). Adds the
        components ``front_reflected`` and ``back_reflected``. Without it, the panels are
        non-reflective (as in the original model). With ``PanelGlass(transmission=True)``,
        all components are weighted with the relative transmittance of the receiving glass
        (effective irradiance behind the glass, normalised to normal incidence). With
        ``PanelGlass(diffuse=...)``, the diffuse reflection of the module surface off the
        adjacent row is added as ``front_diffuse`` and ``back_diffuse``.
    settings : Settings, optional
        Numerical options; ``Settings.matlab()`` reproduces the MATLAB code.

    Returns
    -------
    Result or list of Result
        Time steps with the sun below the horizon or GHI <= 0 are returned as zeros.
    """
    single = isinstance(arrays, Array)
    arrays = [arrays] if single else list(arrays)
    settings = settings or Settings()
    df = prepare_weather(weather, latitude, longitude, altitude)
    n = len(df)
    dni, ghi = df["dni"].to_numpy(dtype=float), df["ghi"].to_numpy(dtype=float)
    az = np.radians(df["solar_azimuth"].to_numpy(dtype=float))
    el = np.radians(df["solar_elevation"].to_numpy(dtype=float))
    alb = _per_time(albedo, df, "albedo", n)
    if far_ground is None:
        far_ground = FarGround()
        alb_fg = _per_time(albedo_fg, df, "albedo_fg", n, default=0.0)
    else:
        alb_fg = _per_time(albedo_fg, df, "albedo_fg", n)
    if horizon is not None:
        if settings.legacy:
            raise ValueError("the horizon is not supported in legacy mode")
        if albedo_terrain is None and "albedo_terrain" not in df.columns:
            if albedo_fg is None and "albedo_fg" not in df.columns:
                raise ValueError("provide 'albedo_terrain' (or a far-ground albedo) for the horizon")
            alb_t = alb_fg
        else:
            alb_t = _per_time(albedo_terrain, df, "albedo_terrain", n)
    doy = _day_of_year(df)

    if settings.legacy:
        if settings.sun_model != "buie":
            raise ValueError("legacy mode only supports sun_model='buie'")
        grid = make_legacy_sky_grid(settings.sky)
    else:
        grid = make_sky_grid(settings.sky)
    models = [ArrayModel(a, far_ground, grid, settings, panel_glass) for a in arrays]
    n_chi = models[0].tables.chi.size
    res = [{c: np.zeros((n, n_chi)) for c in COMPONENTS} for _ in models]
    # diffuse panel reflection: the facing face of the adjacent row is the source; in finite
    # arrays its irradiance differs from this row's and is computed with its own model
    neighbours = [m.neighbours(far_ground, grid) for m in models]
    source = [{side: np.zeros((n, n_chi)) for side, nm in nb.items() if nm is not None} for nb in neighbours]
    ghi_check = np.zeros(n)

    point = settings.sun_model == "point"
    valid = np.flatnonzero((el > 0) & (ghi > 0))
    if horizon is not None:
        hmask = horizon.masked(grid.elevation, grid.azimuth)
        sun_visible = ~horizon.masked(el, az)
    complex_seen = False
    chunks = range(0, valid.size, settings.chunk_size)
    if progress:
        try:
            from tqdm import tqdm
            chunks = tqdm(chunks, unit="chunk")
        except ImportError:
            pass
    for start in chunks:
        idx = valid[start:start + settings.chunk_size]
        sky = sky_radiance(grid, dni[idx], ghi[idx], az[idx], el[idx], sun_model=settings.sun_model,
                           csr=settings.circumsolar_ratio,
                           circumsolar_half_angle=settings.circumsolar_half_angle,
                           day_of_year=None if doy is None else doy[idx],
                           solar_constant=settings.solar_constant)
        L = sky.L
        if np.iscomplexobj(L):
            complex_seen |= bool(np.any(L.imag != 0))
        dni_beam = dni[idx]
        e_fg = None
        if horizon is not None:
            # distant terrain (far ground and terrain above the horizon) is irradiated under
            # free-horizon conditions; only the array and its near ground see the horizon
            free_sun = (dni[idx], el[idx], az[idx]) if point else None
            e_fg = [m.far_ground_irradiance(L, free_sun) for m in models]
            L[hmask] = horizon.terrain_radiance(grid.elevation[hmask], grid.azimuth[hmask], alb_t[idx],
                                                dni[idx], ghi[idx], el[idx], az[idx])
            dni_beam = dni_beam * sun_visible[idx]
        h = np.real(grid.horizontal_weight @ L)
        if point:
            h = h + dni_beam * np.sin(el[idx])
        ghi_check[idx] = h
        sun = (dni_beam, el[idx], az[idx]) if point else None
        for k, (model, r) in enumerate(zip(models, res)):
            e_fg_k = None if e_fg is None else e_fg[k]
            out = model.apply(L, alb[idx], alb_fg[idx], sun, e_fg_k)
            for c in COMPONENTS:
                r[c][idx] = np.real(out[c])
            for side, buf in source[k].items():
                face = "front" if side == "back" else "back"
                o = neighbours[k][side].apply(L, alb[idx], alb_fg[idx], sun, e_fg_k)
                buf[idx] = np.real(sum(o[c] for c in COMPONENTS if c.startswith(face)))
    if complex_seen:
        warnings.warn("legacy mode: the MATLAB air-mass formula produced complex values (solar "
                      "elevation > 1 rad); the real part of the results is returned.", stacklevel=2)
    for model, r, nb, src in zip(models, res, neighbours, source):
        for side in nb:
            face = "front" if side == "back" else "back"
            # single reflection: the source excludes its own diffuse component
            e = src[side] if side in src else sum(r[c] for c in COMPONENTS
                                                  if c.startswith(face) and not c.endswith("diffuse"))
            r[f"{side}_diffuse"] = panel_glass.diffuse * (e @ model.V[side].T)

    results = [Result(chi=m.tables.chi, index=df.index, ghi_check=ghi_check, array=m.array,
                      settings=settings, **r) for m, r in zip(models, res)]
    return results[0] if single else results
