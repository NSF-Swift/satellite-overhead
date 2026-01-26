import math
from dataclasses import dataclass, field
from typing import Optional

from skyfield.sgp4lib import EarthSatellite

from sopp.models.core import FrequencyRange
from sopp.models.satellite.tle import TleInformation

NUMBER_OF_LINES_PER_TLE_OBJECT = 3


@dataclass
class Satellite:
    """A satellite with a name and optional orbital/frequency data.

    Satellites loaded from TLE files will have tle_information populated.
    Satellites reconstructed from trajectory files may not.
    """

    name: str
    tle_information: Optional[TleInformation] = None
    frequency: list[FrequencyRange] = field(default_factory=list)

    @property
    def satellite_number(self) -> int | None:
        """NORAD catalog number, if available."""
        if self.tle_information is None:
            return None
        return self.tle_information.satellite_number

    def to_skyfield(self) -> EarthSatellite:
        """Convert to a Skyfield EarthSatellite. Requires TLE data."""
        if self.tle_information is None:
            raise ValueError(
                f"Satellite '{self.name}' has no TLE data. "
                "Cannot convert to Skyfield without orbital parameters."
            )
        line1, line2 = self.tle_information.to_tle_lines()
        return EarthSatellite(line1=line1, line2=line2, name=self.name)

    @property
    def orbits_per_day(self) -> float:
        """Calculate orbits per day from mean motion. Requires TLE data."""
        if self.tle_information is None:
            raise ValueError(
                f"Satellite '{self.name}' has no TLE data. "
                "Cannot calculate orbits per day without orbital parameters."
            )
        return self.tle_information.mean_motion.value * 1440 / (2 * math.pi)
