import math
from dataclasses import dataclass, field
from pathlib import Path

from skyfield.sgp4lib import EarthSatellite

from sopp.models.core import FrequencyRange
from sopp.models.satellite.tle import TleInformation

"""
The Satellite data class stores all of the TLE information for each satellite, which is loaded from a TLE file using the class method from_tle_file()
and can be converted to a Skyfield API object EarthSatellite using the to_skyfield() method. It also stores all the frequency information
for each satellite.

  + name:               name of satellite. string.
  + tle_information:    stores TLE information. TleInformation is another custom object to store TLE data and can be found in
                        ROOT/sopp.models.satellite.tle.py
  + frequency:          list of type FrequencyRange. FrequencyRange is a custom dataclass that stores a center frequency and bandwidth.

  + to_skyfield():    class method to convert a Satellite object into a Skyfield EarthSatellite object for use with the Skyfield API
  + from_tle_file():    class method to load Satellite from provided TLE file. Returns a list of type Satellite.
"""

NUMBER_OF_LINES_PER_TLE_OBJECT = 3


@dataclass
class Satellite:
    name: str
    tle_information: TleInformation
    frequency: list[FrequencyRange] = field(default_factory=list)

    def to_skyfield(self) -> EarthSatellite:
        line1, line2 = self.tle_information.to_tle_lines()
        return EarthSatellite(line1=line1, line2=line2, name=self.name)

    @property
    def orbits_per_day(self) -> float:
        return self.tle_information.mean_motion.value * 1440 / (2 * math.pi)
