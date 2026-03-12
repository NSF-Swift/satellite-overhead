"""Core data types shared across the SOPP library."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


@dataclass
class Position:
    """A position in the local horizontal coordinate system.

    Attributes:
        altitude: Elevation angle in degrees. 0 is the horizon, 90 is zenith.
            Negative values mean the object is below the horizon.
        azimuth: Azimuth angle in degrees, measured clockwise from geographic
            north (0) through east (90), south (180), and west (270).
        distance_km: Slant range to the object in kilometers, or None.
    """

    altitude: float
    azimuth: float
    distance_km: float | None = None


@dataclass
class TimeWindow:
    """A time interval defined by a start and end time (UTC).

    Attributes:
        begin: Start of the window.
        end: End of the window.
    """

    begin: datetime
    end: datetime

    @property
    def duration(self) -> timedelta:
        return self.end - self.begin

    def overlaps(self, time_window: "TimeWindow"):
        return self.begin < time_window.end and self.end > time_window.begin

    def __str__(self):
        return (
            f"{self.__class__.__name__}:\n"
            f"  Begin:              {self.begin}\n"
            f"  End:                {self.end}"
        )


class CoordinatesJsonKey(Enum):
    latitude = "latitude"
    longitude = "longitude"


@dataclass
class Coordinates:
    """Geographic coordinates in decimal degrees.

    Attributes:
        latitude: Latitude in decimal degrees (positive north).
        longitude: Longitude in decimal degrees (positive east).
    """

    latitude: float
    longitude: float

    @classmethod
    def from_json(cls, info: dict) -> "Coordinates":
        return cls(
            latitude=info[CoordinatesJsonKey.latitude.value],
            longitude=info[CoordinatesJsonKey.longitude.value],
        )


@dataclass
class FrequencyRange:
    """A frequency band defined by center frequency and bandwidth.

    Used for both telescope observations and satellite transmissions.

    Attributes:
        frequency: Center frequency in MHz.
        bandwidth: Total bandwidth in MHz.
        status: Optional metadata (e.g. 'active', 'inactive').
    """

    frequency: float
    bandwidth: float
    status: str | None = None

    @property
    def low_mhz(self) -> float:
        return self.frequency - (self.bandwidth / 2.0)

    @property
    def high_mhz(self) -> float:
        return self.frequency + (self.bandwidth / 2.0)

    def overlaps(self, other: "FrequencyRange") -> bool:
        """Return True if this frequency range overlaps with another."""
        return (self.low_mhz < other.high_mhz) and (self.high_mhz > other.low_mhz)

    def __str__(self):
        return (
            f"{self.__class__.__name__}:\n"
            f"  Frequency:          {self.frequency} MHz\n"
            f"  Bandwidth:          {self.bandwidth} MHz"
        )
