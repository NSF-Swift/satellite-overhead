"""Antenna pointing configuration variants.

Each variant tells the engine how to determine where the antenna points
during an observation.
"""

from dataclasses import dataclass

from sopp.models.core import Position
from sopp.models.ground.target import ObservationTarget
from sopp.models.ground.trajectory import AntennaTrajectory


@dataclass
class StaticPointingConfig:
    """Fixed azimuth/elevation pointing for the entire observation."""

    position: Position


@dataclass
class CelestialTrackingConfig:
    """Track a celestial object (RA/Dec) across the sky."""

    target: ObservationTarget


@dataclass
class CustomTrajectoryConfig:
    """User-provided antenna trajectory with explicit az/alt at each time step."""

    trajectory: AntennaTrajectory


AntennaConfig = StaticPointingConfig | CelestialTrackingConfig | CustomTrajectoryConfig
