"""Celestial observation target."""

from dataclasses import dataclass


@dataclass
class ObservationTarget:
    """A celestial target specified by equatorial coordinates.

    Attributes:
        declination: Declination string (e.g. '7d24m25.426s').
        right_ascension: Right ascension string (e.g. '5h55m10.3s').
    """

    declination: str
    right_ascension: str
