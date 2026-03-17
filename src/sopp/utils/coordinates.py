"""Coordinate transformation utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from skyfield.api import load
from skyfield.toposlib import wgs84

from sopp.models.ground.facility import Facility


def altaz_to_radec(
    azimuth: npt.NDArray[np.float64],
    altitude: npt.NDArray[np.float64],
    times: npt.NDArray[np.object_],
    facility: Facility,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Convert horizontal (Az/El) coordinates to equatorial (RA/Dec).

    Args:
        azimuth: Azimuth angles in degrees.
        altitude: Elevation angles in degrees.
        times: Array of UTC datetime objects.
        facility: Observer facility (provides location).

    Returns:
        Tuple of (ra_degrees, dec_degrees) as numpy arrays.
    """
    if len(azimuth) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    ts = load.timescale()
    t = ts.from_datetimes(times)

    observer = wgs84.latlon(
        latitude_degrees=facility.coordinates.latitude,
        longitude_degrees=facility.coordinates.longitude,
        elevation_m=facility.elevation,
    )

    eph = load("de421.bsp")
    earth = eph["earth"]

    position = (earth + observer).at(t)
    sky = position.from_altaz(alt_degrees=altitude, az_degrees=azimuth)
    ra, dec, _ = sky.radec()

    return ra._degrees, dec.degrees
