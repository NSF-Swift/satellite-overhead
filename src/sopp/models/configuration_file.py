from dataclasses import dataclass, field

from sopp.models.antenna_trajectory import AntennaTrajectory
from sopp.models.observation_target import ObservationTarget
from sopp.models.position import Position
from sopp.models.reservation import Reservation
from sopp.models.runtime_settings import RuntimeSettings


@dataclass
class ConfigurationFile:
    reservation: Reservation
    runtime_settings: RuntimeSettings = field(default_factory=RuntimeSettings)
    antenna_trajectory: AntennaTrajectory | None = None
    observation_target: ObservationTarget | None = None
    static_antenna_position: Position | None = None
