from dataclasses import dataclass

from sopp.models.core import Position
from sopp.models.ground.target import ObservationTarget
from sopp.models.ground.trajectory import AntennaTrajectory


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
