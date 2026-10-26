"""Specular reflection of the module glass between adjacent rows.

The front side of row i and the back side of row i-1 face each other (as do the back side of
row i and the front side of row i+1). Radiation incident on the glass of one of them is partly
reflected specularly towards the other.

Since every module plane contains the direction of the rows, a specular reflection preserves
the component of a ray along the rows and mirrors only its component in the cross-section.
The geometry of the reflected radiation (which part of the mirror is irradiated, where the
reflected rays arrive) therefore remains two-dimensional and depends on the profile angle
only, whereas the angle-dependent reflectance uses the true three-dimensional incidence angle
of each sky element. As mirror and receiving surface are parallel, the incidence angle on the
receiving surface equals the one on the mirror, and the mirror image is merely shifted along
the panel.

Two paths are modelled for every sky element and the sun:

* mirror -> adjacent row (front of row i -> back of row i-1, back of row i -> front of row i+1);
* mirror -> ground -> panels: the part of the reflected bundle that misses the adjacent row
  irradiates periodic strips of the ground, which reflect it onto both panel sides (view
  factors as for the directly irradiated ground).

Optionally (``PanelGlass(transmission=True)``), the transmission into the receiving module is
included as well: every contribution is weighted with tau(theta) / tau(0) = (1 - R(theta)) /
(1 - R(0)) of its true incidence angle. For the beam and sky elements this is a weight per
direction; for the (Lambertian, infinitely long) ground strips the weight is integrated along
the strips, which turns the 2D view factor 1/2 |s_2 - s_1| into 1/2 |G(s_2) - G(s_1)| (see
:func:`view_kernel`).

Not included: the specular reflection of ground-reflected radiance by the glass
(ground -> glass -> panel), reflections reaching non-adjacent rows, and multiple reflections
between panels. Compared with Radiance, the remaining deviation is below 2 % of the total
irradiation (benchmarks/raytrace/reflection_check.py).

Besides the specular reflection at the glass surface, modules reflect diffusely: the cells
behind the glass, the inactive area around and between them, frames and the edges of the
module. With ``PanelGlass(diffuse=R_d)``, every panel face is a Lambertian reflector of the
irradiance it receives (all components, with transmission: the irradiance behind the glass),
with radiance R_d E(chi) / pi. The facing side of the adjacent row receives it through the view
factors between the two parallel faces (single reflection; components ``front_diffuse`` and
``back_diffuse``). For the back side, whose brightest view is usually the sunlit front of the
next row, this is significant at dense row spacing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


def fresnel_reflectance(cos_incidence, n=1.52):
    """Reflectance of an uncoated air/glass interface for unpolarised radiation."""
    c = np.clip(np.asarray(cos_incidence, dtype=float), 0.0, 1.0)
    sin_t = np.sqrt(1.0 - c ** 2) / n
    cos_t = np.sqrt(1.0 - sin_t ** 2)
    rs = ((c - n * cos_t) / (c + n * cos_t)) ** 2
    rp = ((cos_t - n * c) / (cos_t + n * c)) ** 2
    return 0.5 * (rs + rp)


@dataclass(frozen=True)
class PanelGlass:
    """Module glass: specular reflection towards the adjacent rows and, optionally, the
    angle-dependent transmission into the receiving module.

    Parameters
    ----------
    n : float
        Refractive index of the (uncoated) glass, used for the Fresnel reflectance.
    front, back : bool
        Whether the front / back side of the modules reflects (glass-glass or back-to-back
        modules: both).
    reflectance : callable, optional
        Custom reflectance R(cos_incidence), e.g. for anti-reflective coatings; replaces
        the Fresnel reflectance.
    transmission : bool
        If True, every contribution is weighted with the relative transmittance of the
        receiving glass, tau(theta) = (1 - R(theta)) / (1 - R(0)) (incidence angle modifier,
        normalised to normal incidence as under STC). The results are then the effective
        irradiance behind the glass instead of the irradiance on the module plane.
    diffuse : float
        Diffuse (Lambertian) reflectance of the module surface, in addition to the specular
        reflection of the glass: cells, inactive area, frames (see the module docstring).
        Applies to the sides selected by ``front`` / ``back``. Default 0.
    """

    n: float = 1.52
    front: bool = True
    back: bool = True
    reflectance: Optional[Callable] = None
    transmission: bool = False
    diffuse: float = 0.0

    def __post_init__(self):
        if not 0.0 <= self.diffuse < 1.0:
            raise ValueError(f"diffuse must be in [0, 1), got {self.diffuse}")

    def __call__(self, cos_incidence):
        if self.reflectance is not None:
            return np.asarray(self.reflectance(np.asarray(cos_incidence, dtype=float)), dtype=float)
        return fresnel_reflectance(cos_incidence, self.n)

    @property
    def reflects(self) -> bool:
        return self.front or self.back

    def transmittance(self, cos_incidence):
        """Relative transmittance tau(theta) / tau(0) of the receiving glass (1 without
        ``transmission``)."""
        c = np.asarray(cos_incidence, dtype=float)
        if not self.transmission:
            return np.ones_like(c)
        return np.where(c > 0, (1.0 - self(c)) / (1.0 - self(np.ones(1))[0]), 0.0)

    def view_kernel(self):
        """Transmission-weighted counterpart G(s) of s = sin(theta) in the 2D view factor
        F = 1/2 |s_2 - s_1| (None without ``transmission``), see :func:`view_kernel`."""
        return view_kernel(self.transmittance) if self.transmission else None


def view_kernel(iam, n=2001):
    """2D view-factor kernel for an angle-dependent receiver weight ``iam(cos_theta)``.

    For an infinitely long strip, a line of sight at angle phi from the panel normal (in the
    cross-section) and elevation omega along the strip has cos(theta) = cos(omega) cos(phi).
    The irradiance from a Lambertian strip, integrated along the strip, then becomes

        F = 1/2 |G(sin phi_2) - G(sin phi_1)|,  G(s) = int_0^{arcsin s} K(phi) cos(phi) dphi,
        K(phi) = int iam(cos(omega) cos(phi)) cos^2(omega) domega / (pi / 2),

    with G(s) = s for iam = 1. Returns G as a vectorised, odd function on [-1, 1].
    """
    phi = np.linspace(0.0, np.pi / 2, n)
    m = 400
    omega = -np.pi / 2 + (np.arange(m) + 0.5) * np.pi / m
    w = np.cos(omega) ** 2 * (np.pi / m) / (np.pi / 2)
    K = np.array([np.sum(iam(np.cos(omega) * np.cos(f)) * w) for f in phi])
    integrand = K * np.cos(phi)
    g = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(phi))])
    s = np.sin(phi)

    def G(x):
        x = np.asarray(x, dtype=float)
        return np.sign(x) * np.interp(np.abs(x), s, g)

    return G
