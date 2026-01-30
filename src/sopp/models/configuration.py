from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sopp.models.ground.config import AntennaConfig

if TYPE_CHECKING:
    from sopp.models.reservation import Reservation
    from sopp.models.satellite.satellite import Satellite


@dataclass
class RuntimeSettings:
    """
    Configuration settings for controlling the execution and precision of the EventFinder.

    Attributes:
        time_resolution_seconds (float): The step size (granularity) of the simulation
            grid in seconds.
            (Default: 1.0)
        concurrency_level (int): The number of parallel processes to spawn for
            calculating satellite windows.
            (Default: 1)
        min_altitude (float): The minimum elevation angle (in degrees) required for
            a satellite to be considered visible or "above the horizon".
            (Default: 0.0)
    """

    time_resolution_seconds: float = field(default=1)
    concurrency_level: int = field(default=1)
    min_altitude: float = field(default=0.0)

    def __str__(self):
        return (
            f"{self.__class__.__name__}:\n"
            f"  Time Interval:      {self.time_resolution_seconds}\n"
            f"  Concurrency:        {self.concurrency_level}"
            f"  Min. Altitude:      {self.min_altitude}"
        )


@dataclass
class Configuration:
    reservation: Reservation
    satellites: list[Satellite]
    antenna_config: AntennaConfig
    runtime_settings: RuntimeSettings = field(default_factory=RuntimeSettings)

    def __post_init__(self):
        """
        Validates the configuration.
        Raises ValueError if state is invalid.
        """
        self._validate_satellites()
        self._validate_reservation()
        self._validate_settings()
        self._validate_antenna_config()

    def _validate_satellites(self):
        if not self.satellites:
            raise ValueError("Satellites list cannot be empty.")

    def _validate_reservation(self):
        if self.reservation.time.begin >= self.reservation.time.end:
            raise ValueError(
                f"Reservation start ({self.reservation.time.begin}) "
                f"must be before end ({self.reservation.time.end})"
            )
        if self.reservation.facility.beamwidth <= 0:
            raise ValueError(
                f"Beamwidth must be > 0, provided: {self.reservation.facility.beamwidth}"
            )

    def _validate_settings(self):
        if self.runtime_settings.time_resolution_seconds <= 0:
            raise ValueError("Time resolution must be greater than 0 seconds.")
        if self.runtime_settings.concurrency_level < 1:
            raise ValueError("Concurrency level must be at least 1.")
        if self.runtime_settings.min_altitude < 0:
            raise ValueError("Minimum altitude must be non-negative.")

    def _validate_antenna_config(self):
        if not isinstance(self.antenna_config, AntennaConfig):
            raise ValueError("Invalid antenna configuration.")

    def __str__(self):
        return (
            f"{self.__class__.__name__}:\n"
            f"{self.reservation}\n"
            f"{self.runtime_settings}\n"
            f"Satellites:           {len(self.satellites)} total"
        )
