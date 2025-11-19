from dataclasses import dataclass, field

from sopp.custom_dataclasses.observation_target import ObservationTarget
from sopp.custom_dataclasses.position import Position
from sopp.custom_dataclasses.position_time import PositionTime
from sopp.custom_dataclasses.reservation import Reservation
from sopp.custom_dataclasses.runtime_settings import RuntimeSettings


@dataclass
class ConfigurationFile:
    reservation: Reservation
    runtime_settings: RuntimeSettings = field(default_factory=RuntimeSettings)
    antenna_position_times: list[PositionTime] | None = None
    observation_target: ObservationTarget | None = None
    static_antenna_position: Position | None = None
