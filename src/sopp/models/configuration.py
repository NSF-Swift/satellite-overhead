from dataclasses import dataclass, field

from sopp.models.antenna_trajectory import AntennaTrajectory
from sopp.models.reservation import Reservation
from sopp.models.runtime_settings import RuntimeSettings
from sopp.models.satellite.satellite import Satellite


@dataclass
class Configuration:
    reservation: Reservation
    satellites: list[Satellite]
    antenna_trajectory: AntennaTrajectory
    runtime_settings: RuntimeSettings = field(default_factory=RuntimeSettings)

    def __post_init__(self):
        """
        Validates the configuration.
        Raises ValueError if state is invalid.
        """
        self._validate_satellites()
        self._validate_reservation()
        self._validate_settings()
        self._validate_trajectory()

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

    def _validate_trajectory(self):
        if not self.antenna_trajectory or len(self.antenna_trajectory) == 0:
            raise ValueError("Antenna trajectory is empty.")

        if len(self.antenna_trajectory) > 1:
            if self.antenna_trajectory.times[-1] <= self.antenna_trajectory.times[0]:
                raise ValueError("Antenna trajectory times are not increasing.")

    def __str__(self):
        return (
            f"{self.__class__.__name__}:\n"
            f"{self.reservation}\n"
            f"{self.runtime_settings}\n"
            f"Satellites:           {len(self.satellites)} total"
        )
