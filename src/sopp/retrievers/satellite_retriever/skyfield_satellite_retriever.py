from dataclasses import dataclass

from skyfield.api import EarthSatellite, load


@dataclass
class SkyfieldSatelliteList:
    satellites: list[EarthSatellite]

    @classmethod
    def load_tle(cls, tle_file) -> "SkyfieldSatelliteList":
        return cls(satellites=load.tle_file(tle_file))
