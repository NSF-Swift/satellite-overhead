from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


@dataclass
class Position:
    """
    Represents a position relative to an observer on Earth.

    Attributes:
    + altitude (float): The altitude angle of the object in degrees. It ranges
      from 0° at the horizon to 90° directly overhead at the zenith. A negative
      altitude means the satellite is below the horizon.
    + azimuth (float): The azimuth angle of the object in degrees, measured
      clockwise around the horizon. It runs from 0° (geographic north) through
      east (90°), south (180°), and west (270°) before returning to the north.
    + distance (Optional[float]): The straight-line distance between the
      object and the observer in kilometers. If not provided, it is set to
      None.
    """

    altitude: float
    azimuth: float
    distance_km: float | None = None


@dataclass
class TimeWindow:
    """
    The TimeWindow class is used to store the beginning and end time of events. The duration function returns a time delta for the
    duration of the event and the overlaps function determines if the TimeWindow overlaps with another TimeWindow
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
    """
    Represents a frequency band (center frequency + bandwidth).
    Used for both Telescope Observations and Satellite Transmissions.
    """

    frequency: float  # Center frequency in MHz
    bandwidth: float  # Total bandwidth in MHz
    status: str | None = None  # Metadata (e.g. 'active', 'inactive')

    @property
    def low_mhz(self) -> float:
        return self.frequency - (self.bandwidth / 2.0)

    @property
    def high_mhz(self) -> float:
        return self.frequency + (self.bandwidth / 2.0)

    def overlaps(self, other: "FrequencyRange") -> bool:
        """
        Determines if this frequency range overlaps with another.
        Logic: startA < endB AND endA > startB
        """
        return (self.low_mhz < other.high_mhz) and (self.high_mhz > other.low_mhz)

    def __str__(self):
        return (
            f"{self.__class__.__name__}:\n"
            f"  Frequency:          {self.frequency} MHz\n"
            f"  Bandwidth:          {self.bandwidth} MHz"
        )
