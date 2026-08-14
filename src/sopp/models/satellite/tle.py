"""TLE (Two-Line Element) data model and parsing."""

from dataclasses import dataclass

from sgp4.api import WGS72, Satrec
from sgp4.exporter import export_tle
from sgp4.io import verify_checksum


@dataclass
class MeanMotion:
    """Mean motion of a satellite in radians per minute.

    Attributes:
        first_derivative: First derivative of mean motion (rad/min^2).
        second_derivative: Second derivative of mean motion (rad/min^3).
        value: Mean motion value (rad/min).
    """

    first_derivative: float
    second_derivative: float
    value: float


@dataclass
class InternationalDesignator:
    """International designator (COSPAR ID) identifying a launch and piece.

    Attributes:
        year: Four-digit launch year.
        launch_number: Sequential launch number within the year.
        launch_piece: Piece identifier (e.g. 'A', 'B').
    """

    year: int
    launch_number: int
    launch_piece: str

    def to_tle_string(self) -> str:
        return f"{str(self.year % 100).zfill(2)}{str(self.launch_number).zfill(3)}{self.launch_piece}"

    @classmethod
    def from_tle_string(cls, tle_string: str) -> "InternationalDesignator":
        # Two-digit years pivot at 57: nothing was launched before Sputnik (1957).
        two_digit_year = int(tle_string[0:2])
        century = 1900 if two_digit_year >= 57 else 2000
        return InternationalDesignator(
            year=century + two_digit_year,
            launch_number=int(tle_string[2:5]),
            launch_piece=tle_string[5:].strip(),
        )

    @classmethod
    def from_object_id(cls, object_id: str) -> "InternationalDesignator":
        """Parses an OMM OBJECT_ID such as '1998-067A'."""
        year, launch = object_id.split("-")
        return InternationalDesignator(
            year=int(year),
            launch_number=int(launch[:3]),
            launch_piece=launch[3:].strip(),
        )


@dataclass
class TleInformation:
    """Parsed orbital elements from a Two-Line Element set.

    All angular values are in radians (SGP4 convention).
    """

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
    classification: str = "U"
    international_designator: InternationalDesignator | None = None

    def to_satrec(self) -> Satrec:
        """Build an SGP4 Satrec propagator directly from the stored elements."""
        satrec = Satrec()
        satrec.sgp4init(
            WGS72,
            "i",
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
        satrec.intldesg = (
            self.international_designator.to_tle_string()
            if self.international_designator is not None
            else ""
        )
        satrec.revnum = self.revolution_number

        return satrec

    def to_tle_lines(self):
        """Export as TLE lines.

        The TLE satellite number field is 5 characters, so plain digits
        stop at 99999. sgp4 also writes the Alpha-5 extension, where a
        leading letter counts as 10-33 ("A0000" is satellite 100000,
        "Z9999" is 339999). Larger numbers raise ValueError.
        """
        return export_tle(satrec=self.to_satrec())

    @classmethod
    def from_tle_lines(cls, line1: str, line2: str) -> "TleInformation":
        verify_checksum(line1, line2)
        return cls.from_satrec(Satrec.twoline2rv(line1, line2))

    @classmethod
    def from_satrec(cls, satrec: Satrec) -> "TleInformation":
        international_designator = (
            InternationalDesignator.from_tle_string(satrec.intldesg)
            if satrec.intldesg
            else None
        )

        return TleInformation(
            argument_of_perigee=satrec.argpo,
            drag_coefficient=satrec.bstar,
            eccentricity=satrec.ecco,
            epoch_days=satrec.jdsatepoch
            - 2433281.5
            + satrec.jdsatepochF,  # JD of 1949-Dec-31 00:00 UT (SGP4 epoch reference)
            inclination=satrec.inclo,
            international_designator=international_designator,
            mean_anomaly=satrec.mo,
            mean_motion=MeanMotion(
                first_derivative=satrec.ndot,
                second_derivative=satrec.nddot,
                value=satrec.no_kozai,
            ),
            revolution_number=satrec.revnum,
            right_ascension_of_ascending_node=satrec.nodeo,
            satellite_number=satrec.satnum,
            classification=satrec.classification,
        )
