from dataclasses import dataclass, field

from sopp.models.observation_target import ObservationTarget
from sopp.models.position import Position
from sopp.models.position_time import PositionTime
from sopp.models.reservation import Reservation
from sopp.models.runtime_settings import RuntimeSettings


@dataclass
class ConfigurationFile:
    reservation: Reservation
    runtime_settings: RuntimeSettings = field(default_factory=RuntimeSettings)
    antenna_position_times: list[PositionTime] | None = None
    observation_target: ObservationTarget | None = None
    static_antenna_position: Position | None = None
