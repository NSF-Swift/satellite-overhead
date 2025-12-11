from dataclasses import dataclass
from sopp.models.position import Position
from sopp.models.observation_target import ObservationTarget
from sopp.models.antenna_trajectory import AntennaTrajectory


@dataclass
class StaticPointingConfig:
    position: Position


@dataclass
class CelestialTrackingConfig:
    target: ObservationTarget


@dataclass
class CustomTrajectoryConfig:
    trajectory: AntennaTrajectory


AntennaConfig = StaticPointingConfig | CelestialTrackingConfig | CustomTrajectoryConfig
