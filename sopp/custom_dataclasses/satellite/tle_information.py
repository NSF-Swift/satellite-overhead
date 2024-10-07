from dataclasses import dataclass
from typing import Optional

from sgp4.exporter import export_tle
from sgp4.model import Satrec
from sgp4.vallado_cpp import WGS72
from sgp4.io import verify_checksum

from sopp.custom_dataclasses.satellite.international_designator import InternationalDesignator
from sopp.custom_dataclasses.satellite.mean_motion import MeanMotion

from math import radians, sqrt


@dataclass
class TleInformation:
    argument_of_perigee: float
    drag_coefficient: float
    eccentricity: float
    epoch_days: float
    inclination: float
    mean_anomaly: float
    mean_motion: MeanMotion
    revolution_number: int
    right_ascension_of_ascending_node: float
    satellite_number: int
    classification: str = 'U'
    international_designator: Optional[InternationalDesignator] = None

    def to_tle_lines(self):
        satrec = Satrec()
        satrec.sgp4init(
            WGS72,
            'i',
            self.satellite_number,
            self.epoch_days,
            self.drag_coefficient,
            self.mean_motion.first_derivative,
            self.mean_motion.second_derivative,
            self.eccentricity,
            self.argument_of_perigee,
            self.inclination,
            self.mean_anomaly,
            self.mean_motion.value,
            self.right_ascension_of_ascending_node,
        )
        satrec.classification = self.classification
        satrec.intldesg = self.international_designator.to_tle_string() if self.international_designator is not None else ""
        satrec.revnum = self.revolution_number

        return export_tle(satrec=satrec)

    @classmethod
    def from_tle_lines(cls, line1: str, line2: str) -> 'TleInformation':
        verify_checksum(line1, line2)
        satrec = Satrec.twoline2rv(line1=line1, line2=line2)
        international_designator = InternationalDesignator.from_tle_string(satrec.intldesg) if satrec.intldesg else None

        return TleInformation(
            argument_of_perigee=satrec.argpo,
            drag_coefficient=satrec.bstar,
            eccentricity=satrec.ecco,
            epoch_days=satrec.jdsatepoch - 2433281.5 + satrec.jdsatepochF, #what is this number??
            inclination=satrec.inclo,
            international_designator=international_designator,
            mean_anomaly=satrec.mo,
            mean_motion=MeanMotion(
                first_derivative=satrec.ndot,
                second_derivative=satrec.nddot,
                value=satrec.no_kozai
            ),
            revolution_number=satrec.revnum,
            right_ascension_of_ascending_node=satrec.nodeo,
            satellite_number=satrec.satnum,
            classification=satrec.classification
        )
    
    def calculate_velocity_vector(self) -> tuple:
        # Step 1: Compute mean motion (n) in radians per minute
        mean_motion_rad_per_min = radians(self.mean_motion.value)

        # Step 2: Compute semi-major axis (a) in kilometers
        mean_motion_rad_per_sec = mean_motion_rad_per_min / 60.0
        a = (398600.4418 / mean_motion_rad_per_sec**2)**(1/3.0)

        # Step 3: Compute velocity (V) in km/s
        e = self.eccentricity
        V = sqrt(398600.4418 / a * (1 + e))

        # Step 4: Compute velocity components
        V_along_track = V * (1 - e)
        V_cross_track = V * sqrt(1 - e**2)

        # Return velocity vector as a tuple, in terms of geocentric coordinate system
        return V_along_track, V_cross_track
