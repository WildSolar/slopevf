"""Array geometry in the cross-section perpendicular to the rows.

All lengths are non-dimensionalised by the panel height H. The cross-section
coordinate system (x', y') has its origin on the ground directly below the
centre of panel row i = 0, x' pointing uphill and y' pointing up (paper,
Fig. 2). A point on the ground is addressed by its distance d along the slope,
a point on panel i by chi in [-1/2, 1/2] (bottom to top edge).

Angles are stored in radians internally; the public :class:`Array` takes
degrees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Array:
    """A regular, infinitely long multi-row PV array on a sloped plane.

    Parameters
    ----------
    beta_p : float
        Panel tilt from horizontal [deg]. Must satisfy ``beta_g < beta_p <= 90``.
    beta_g : float
        Ground slope from horizontal [deg].
    rho_bar : float
        Horizontal row spacing divided by the panel height [-].
    h_bar : float
        Vertical height of the panel centre above ground divided by the panel
        height [-].
    gamma : float, optional
        Azimuth of the slope (downhill direction) [deg]; 0 = south, positive
        towards west. Panels face downhill.
    thickness : float, optional
        Thickness of the panel (module stack) divided by the panel height [-]. Each row is
        then a rectangle centred on the row plane (``h_bar`` refers to its centre), and the
        front and back sides are its outer faces. Default 0: infinitely thin panels.
    """

    beta_p: float
    beta_g: float
    rho_bar: float
    h_bar: float
    gamma: float = 0.0
    thickness: float = 0.0

    def __post_init__(self):
        if self.thickness < 0.0:
            raise ValueError(f"thickness must be non-negative, got {self.thickness}")
        if not 0.0 <= self.beta_g < 90.0:
            raise ValueError(f"beta_g must be in [0, 90) deg, got {self.beta_g}")
        if not self.beta_g < self.beta_p <= 90.0:
            raise ValueError(
                f"panel tilt must satisfy beta_g < beta_p <= 90 deg, got beta_p={self.beta_p}, "
                f"beta_g={self.beta_g}")
        if self.rho_bar <= 0.0:
            raise ValueError(f"rho_bar must be positive, got {self.rho_bar}")
        g = Geometry(self)
        if g.clearance <= 0.0:
            raise ValueError(
                f"the lower panel edge is below ground (h_bar={self.h_bar} is too small "
                f"for the given tilts; minimum is {self.h_bar - g.clearance:.4f})")

    @classmethod
    def from_shading_angle(cls, beta_p: float, beta_g: float, alpha_sh: float, h_bar: float,
                           gamma: float = 0.0) -> "Array":
        """Create an array (thin panels) whose row spacing yields the shading angle ``alpha_sh`` [deg]."""
        bp, bg, a = np.radians([beta_p, beta_g, alpha_sh])
        rho_bar = (np.tan(a) * np.cos(bp) + np.sin(bp)) / (np.tan(bg) + np.tan(a))
        return cls(beta_p, beta_g, float(rho_bar), h_bar, gamma)

    @property
    def shading_angle(self) -> float:
        """Solar profile angle below which row-to-row shading occurs [deg] (eqn. 4)."""
        return float(np.degrees(Geometry(self).alpha[0]))


@dataclass(frozen=True)
class FarGround:
    """Characteristic orientation of the distant terrain seen above the row in front.

    Parameters
    ----------
    beta : float
        Slope of the far ground [deg].
    azimuth : float
        Azimuth of the far-ground slope [deg]; 0 = south, positive towards west.
    """

    beta: float = 0.0
    azimuth: float = 0.0


@dataclass(frozen=True)
class Geometry:
    """Derived cross-section quantities of an :class:`Array` (radians, units of H)."""

    array: Array
    bp: float = field(init=False)
    bg: float = field(init=False)
    rho: float = field(init=False)
    h: float = field(init=False)
    p: float = field(init=False)
    t: float = field(init=False)

    def __post_init__(self):
        a = self.array
        object.__setattr__(self, "bp", np.radians(a.beta_p))
        object.__setattr__(self, "bg", np.radians(a.beta_g))
        object.__setattr__(self, "rho", float(a.rho_bar))
        object.__setattr__(self, "h", float(a.h_bar))
        # row period along the slope
        object.__setattr__(self, "p", self.rho / np.cos(self.bg))
        object.__setattr__(self, "t", float(a.thickness))

    # --- basic vectors --------------------------------------------------------
    @property
    def tangent(self) -> np.ndarray:
        """Unit vector along the panel, pointing from the lower to the upper edge."""
        return np.array([np.cos(self.bp), np.sin(self.bp)])

    @property
    def normal(self) -> np.ndarray:
        """Unit normal of the front side (pointing downhill and up)."""
        return np.array([-np.sin(self.bp), np.cos(self.bp)])

    @property
    def ground_dir(self) -> np.ndarray:
        """Unit vector along the ground, pointing uphill."""
        return np.array([np.cos(self.bg), np.sin(self.bg)])

    def height_above_ground(self, x, y):
        """Vertical distance of a cross-section point above the ground plane."""
        return y - x * np.tan(self.bg)

    def _offset(self, side):
        """Offset of the given side's face from the row plane (thickness / 2 along the normal)."""
        if side is None:
            return 0.0
        if side not in ("front", "back"):
            raise ValueError(side)
        return 0.5 * self.t * (1.0 if side == "front" else -1.0)

    def panel_point(self, chi, row: int = 0, side: str | None = None):
        """Cross-section coordinates of position ``chi`` on panel ``row``: on the row plane, or on
        the outer face of ``side`` ("front"/"back") for a panel of finite thickness."""
        chi = np.asarray(chi, dtype=float)
        o = self._offset(side)
        x = row * self.rho + chi * np.cos(self.bp) - o * np.sin(self.bp)
        y = self.h + row * self.rho * np.tan(self.bg) + chi * np.sin(self.bp) + o * np.cos(self.bp)
        return x, y

    def corners(self, row: int = 0) -> np.ndarray:
        """Corners of the panel cross-section of ``row`` as (4, 2): bottom-front, bottom-back,
        top-front, top-back (coinciding pairwise for thin panels)."""
        pts = [self.panel_point(c, row, s) for c in (-0.5, 0.5) for s in ("front", "back")]
        return np.array(pts, dtype=float)

    @property
    def clearance(self) -> float:
        """Vertical clearance of the lowest panel corner above the ground."""
        c = self.corners()
        return float(np.min(self.height_above_ground(c[:, 0], c[:, 1])))

    # --- shading cases (section 2.2) ------------------------------------------
    def _sun(self, alpha):
        alpha = np.asarray(alpha, dtype=float)
        return -np.cos(alpha), np.sin(alpha)           # unit vector towards the sun (profile plane)

    def _face_projection(self, alpha, side):
        """Interval (lo, hi) in chi on the outer face of ``side`` of row 0 covered by the projection
        of the adjacent row (front: row -1, back: row +1) along the sun direction."""
        ux, uy = self._sun(alpha)
        tx, ty = self.tangent
        fx, fy = self.panel_point(0.0, 0, side)
        q = self.corners(-1 if side == "front" else +1)
        den = tx * uy - ty * ux                         # cross(t, u)
        with np.errstate(divide="ignore", invalid="ignore"):
            chi = [((qx - fx) * uy - (qy - fy) * ux) / den for qx, qy in q]
        return np.minimum.reduce(chi), np.maximum.reduce(chi)

    def _sun_angle_to(self, px, py, q):
        """Profile angles at which the sun direction from (px, py) passes through the points q."""
        return np.arctan2(q[:, 1] - py, -(q[:, 0] - px))

    @property
    def alpha(self) -> np.ndarray:
        """Transition profile angles alpha_1..alpha_4 (eqns. 7-10) [rad].

        alpha_1 / alpha_3: begin of row-to-row shading on the front / back side, i.e. the sun
        direction from the lower edge of the face through the upper silhouette corner of the
        adjacent row (for thin panels eqns. 7 and 9).
        """
        bp, bg = self.bp, self.bg
        top_down, top_up = self.corners(-1)[2:], self.corners(+1)[2:]
        a1 = np.max(self._sun_angle_to(*self.panel_point(-0.5, 0, "front"), top_down))
        a3 = np.min(np.mod(self._sun_angle_to(*self.panel_point(-0.5, 0, "back"), top_up), 2 * np.pi))
        return np.array([a1, np.pi - bp, a3, np.pi - bg])

    def shaded_fraction_front(self, alpha):
        """Shaded fraction eta_f of the front side in case (i) (eqn. 10).

        For thin panels written as ``1 - p sin(a + bg) / sin(a + bp)``, which equals eqn. 10
        but stays finite at a profile angle of 90 deg; in general from the projection of the
        downhill row onto the face (shading starts at the lower edge).
        """
        alpha = np.asarray(alpha, dtype=float)
        if self.t == 0.0:
            return 1.0 - self.p * np.sin(alpha + self.bg) / np.sin(alpha + self.bp)
        return self._face_projection(alpha, "front")[1] + 0.5

    def shaded_fraction_back(self, alpha):
        """Shaded fraction eta_b of the back side in case (iv) (eqn. 14)."""
        alpha = np.asarray(alpha, dtype=float)
        if self.t == 0.0:
            return 1.0 + self.p * np.sin(alpha + self.bg) / np.sin(alpha + self.bp)
        return self._face_projection(alpha, "back")[1] + 0.5

    def shadow_edges(self, alpha):
        """Ground positions d1, d2 bounding the shadow cast by row 0.

        Thin panels: shadows of the lower/upper edge, eqns. 13/14 at i = 0 multiplied through by
        cos(alpha) so that they hold for all profile angles in (0, pi - beta_g). Finite
        thickness: the extreme projections of the four corners (d1 <= d2).
        """
        alpha = np.asarray(alpha, dtype=float)
        if self.t == 0.0:
            den = 2.0 * np.sin(alpha + self.bg)
            num0 = 2.0 * self.h * np.cos(alpha)
            s = np.sin(alpha + self.bp)
            return (num0 - s) / den, (num0 + s) / den
        ux, uy = self._sun(alpha)
        f_u = uy - ux * np.tan(self.bg)                # rate of height gain along the sun direction
        d = []
        for qx, qy in self.corners():
            with np.errstate(divide="ignore", invalid="ignore"):
                s = self.height_above_ground(qx, qy) / f_u
            d.append((qx - s * ux) * np.cos(self.bg) + (qy - s * uy) * np.sin(self.bg))
        return np.minimum.reduce(d), np.maximum.reduce(d)

    # --- visible ground (section 2.4) -----------------------------------------
    @property
    def d_plane(self) -> float:
        """Ground position where the panel plane meets the ground (= d_f,max = d_b,min)."""
        return -self.h / (np.cos(self.bg) * (np.tan(self.bp) - np.tan(self.bg)))

    def d_face(self, side: str) -> float:
        """Ground position where the plane of the outer face of ``side`` meets the ground."""
        if self.t == 0.0:
            return self.d_plane
        x0, y0 = self.panel_point(0.0, 0, side)
        tx, ty = self.tangent
        k = -self.height_above_ground(x0, y0) / (ty - tx * np.tan(self.bg))
        return float((x0 + k * tx) * np.cos(self.bg) + (y0 + k * ty) * np.sin(self.bg))

    def _ray_ground(self, chi, qx, qy, side=None):
        """Ground position hit by the ray from P(chi) through the point (qx, qy).

        Returns +-inf if the ray does not descend towards the ground.
        """
        px, py = self.panel_point(chi, 0, side)
        fp = self.height_above_ground(px, py)
        fq = self.height_above_ground(qx, qy)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = fp / (fp - fq)
            x = px + t * (qx - px)
            y = py + t * (qy - py)
            d = x * np.cos(self.bg) + y * np.sin(self.bg)
        direction = np.sign((qx - px) * np.cos(self.bg) + (qy - py) * np.sin(self.bg))
        return np.where(fq < fp, d, direction * np.inf)

    def view_window(self, chi, side: str, outward: bool = False):
        """Visible near-ground interval [lo, hi] for points ``chi`` (eqns. 23-26).

        The window is bounded by the plane of the (outer) face and by the nearest ground point
        hidden by the adjacent row (downhill row for the front, uphill row for the back), i.e.
        the ray through its lower edge, or for finite thickness the extreme corner projection.
        ``outward``: the side has no adjacent row (outermost row of a finite array), and the
        window extends to infinity.
        """
        chi = np.asarray(chi, dtype=float)
        if outward:
            d0 = np.full(chi.shape, self.d_face(side))
            return (np.full_like(d0, -np.inf), d0) if side == "front" else (d0, np.full_like(d0, np.inf))
        if side == "front":
            q = self.corners(-1) if self.t > 0 else [self.panel_point(-0.5, row=-1)]
            lo = np.maximum.reduce([self._ray_ground(chi, qx, qy, side) for qx, qy in q])
            return lo, np.full_like(lo, self.d_face(side))
        if side == "back":
            q = self.corners(+1) if self.t > 0 else [self.panel_point(-0.5, row=+1)]
            hi = np.minimum.reduce([self._ray_ground(chi, qx, qy, side) for qx, qy in q])
            return np.full_like(hi, self.d_face(side)), hi
        raise ValueError(side)

    def sin_view_angle(self, chi, d, side: str | None = None):
        """Sine of the angle between the panel normal and the line of sight to ground point d.

        The sign convention is such that the value increases towards the upper
        panel edge direction. For d = +-inf the limiting direction is used. With finite
        thickness, ``side`` selects the face the point lies on.
        """
        px, py = self.panel_point(chi, 0, side)
        d = np.asarray(d, dtype=float)
        tx, ty = np.cos(self.bp), np.sin(self.bp)
        cg, sg = np.cos(self.bg), np.sin(self.bg)
        finite = np.isfinite(d)
        dd = np.where(finite, d, 0.0)
        ux = dd * cg - px
        uy = dd * sg - py
        s = (tx * ux + ty * uy) / np.hypot(ux, uy)
        s_inf = np.sign(d) * (tx * cg + ty * sg)
        return np.where(finite, s, s_inf)


def profile_angle(elevation, azimuth, gamma):
    """Profile angle alpha' of a sky direction in the cross-section plane (eqn. 2) [rad].

    All arguments in radians; azimuths measured from south, positive towards west.
    """
    return np.arctan2(np.sin(elevation), np.cos(elevation) * np.cos(azimuth - gamma))


def direction_vector(elevation, azimuth):
    """Unit vector (x = west, y = south, z = up) towards a sky direction (radians)."""
    ce = np.cos(elevation)
    return np.stack([np.sin(azimuth) * ce, np.cos(azimuth) * ce, np.sin(elevation)], axis=-1)


def surface_normal(tilt, azimuth):
    """Unit normal of a plane tilted by ``tilt`` towards ``azimuth`` (radians)."""
    return np.array([np.sin(azimuth) * np.sin(tilt), np.cos(azimuth) * np.sin(tilt), np.cos(tilt)])
