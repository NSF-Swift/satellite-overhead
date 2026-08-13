"""Satellite data model."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from skyfield.api import load
from skyfield.sgp4lib import EarthSatellite

if TYPE_CHECKING:
    from sopp.models.core import FrequencyRange
    from sopp.models.satellite.tle import TleInformation
    from sopp.models.satellite.transmitter import Transmitter

_SKYFIELD_TIMESCALE = load.timescale()


@dataclass
class Satellite:
    """A satellite with a name and optional orbital/frequency data.

    Satellites loaded from TLE files will have tle_information populated.
    Satellites reconstructed from trajectory files may not.
    """

    name: str
    tle_information: TleInformation | None = None
    frequency: list[FrequencyRange] = field(default_factory=list)
    transmitter: Transmitter | None = None

    @property
    def satellite_number(self) -> int | None:
        """NORAD catalog number, if available."""
        if self.tle_information is None:
            return None
        return self.tle_information.satellite_number

    def to_skyfield(self) -> EarthSatellite:
        """Convert to a Skyfield EarthSatellite. Requires orbital elements.

        Builds the SGP4 propagator directly from the parsed elements, so
        satellite numbers beyond the TLE format's field width work fine.
        """
        if self.tle_information is None:
            raise ValueError(
                f"Satellite '{self.name}' has no TLE data. "
                "Cannot convert to Skyfield without orbital parameters."
            )
        earth_satellite = EarthSatellite.from_satrec(
            self.tle_information.to_satrec(), _SKYFIELD_TIMESCALE
        )
        earth_satellite.name = self.name
        return earth_satellite

    @property
    def orbits_per_day(self) -> float:
        """Calculate orbits per day from mean motion. Requires TLE data."""
        if self.tle_information is None:
            raise ValueError(
                f"Satellite '{self.name}' has no TLE data. "
                "Cannot calculate orbits per day without orbital parameters."
            )
        return self.tle_information.mean_motion.value * 1440 / (2 * math.pi)
