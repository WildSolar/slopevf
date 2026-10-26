"""Solar position helper (requires pvlib)."""

from __future__ import annotations

import pandas as pd


def solar_position(times: pd.DatetimeIndex, latitude: float, longitude: float, altitude: float = 0.0):
    """Sun position in the paper's convention.

    Returns a DataFrame with ``solar_azimuth`` (0 = south, positive towards west)
    and ``solar_elevation`` (apparent, i.e. refraction corrected) in degrees.
    Naive timestamps are interpreted as UTC by pvlib.
    """
    try:
        import pvlib
    except ImportError as err:  # pragma: no cover
        raise ImportError("computing the sun position requires pvlib (pip install pvlib)") from err
    sp = pvlib.solarposition.get_solarposition(times, latitude, longitude, altitude=altitude)
    return pd.DataFrame({"solar_azimuth": sp["azimuth"].to_numpy() - 180.0,
                         "solar_elevation": sp["apparent_elevation"].to_numpy()}, index=times)
