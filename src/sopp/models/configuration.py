from dataclasses import dataclass, field

from sopp.models.position_time import PositionTime
from sopp.models.reservation import Reservation
from sopp.models.runtime_settings import RuntimeSettings
from sopp.models.satellite.satellite import Satellite


@dataclass
class Configuration:
    reservation: Reservation
    satellites: list[Satellite]
    antenna_direction_path: list[PositionTime]
    runtime_settings: RuntimeSettings = field(default_factory=RuntimeSettings)

    def __str__(self):
        return (
            f"{self.__class__.__name__}:\n"
            f"{self.reservation}\n"
            f"{self.runtime_settings}\n"
            f"Satellites:           {len(self.satellites)} total"
        )
