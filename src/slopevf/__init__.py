"""View-factor based irradiance distribution on multi-row bifacial PV arrays on sloped ground.

Python implementation of

    M. Wild, S. Strebel, D. Anderegg, J. Rohrer: "A general view-factor based
    irradiation distribution model for multi-row bifacial PV arrays on sloped
    and reflective ground".
"""

from .geometry import Array, FarGround
from .horizon import Horizon
from .model import COMPONENTS, ArrayModel, Result, Settings, prepare_weather, simulate
from .reflection import PanelGlass

__all__ = ["Array", "FarGround", "Horizon", "PanelGlass", "Settings", "Result", "ArrayModel", "simulate",
           "prepare_weather", "COMPONENTS"]
__version__ = "1.0.0"
