from dataclasses import dataclass, field

from sopp.models import AntennaTrajectory, Reservation, RuntimeSettings, Satellite


@dataclass
class Configuration:
    reservation: Reservation
    satellites: list[Satellite]
    antenna_trajectory: AntennaTrajectory
    runtime_settings: RuntimeSettings = field(default_factory=RuntimeSettings)

    def __str__(self):
        return (
            f"{self.__class__.__name__}:\n"
            f"{self.reservation}\n"
            f"{self.runtime_settings}\n"
            f"Satellites:           {len(self.satellites)} total"
        )
