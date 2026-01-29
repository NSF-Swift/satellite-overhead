from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sopp.analysis.geometry import calculate_angular_separation_sq
from sopp.models.satellite.trajectory import SatelliteTrajectory

if TYPE_CHECKING:
    from sopp.models.core import FrequencyRange
    from sopp.models.ground.facility import Facility
    from sopp.models.ground.trajectory import AntennaTrajectory


@dataclass
class InterferenceResult:
    """The output of any interference strategy.

    Wraps a SatelliteTrajectory (the interfering segment) with optional
    quantitative interference data. All strategies produce this same
    structure, making them interchangeable.
    """

    trajectory: SatelliteTrajectory

    # Quantitative interference level (strategy-dependent, may be None).
    # For geometric: None (binary detection only).
    # For gain-based: antenna gain in dB.
    # For link budget: received power in dBW.
    interference_level: np.ndarray | None = None
    level_units: str | None = None
    metadata: dict | None = None


class InterferenceStrategy(ABC):
    """Abstract base class for interference detection strategies.

    Each strategy implements a different method of determining whether
    a satellite causes interference and optionally how much.

    The ``calculate`` method receives data that is common to every
    trajectory in a simulation run: the antenna pointing, telescope
    facility, and observation frequency.  Model-specific configuration
    (transmitter characteristics, atmospheric profiles, antenna gain
    patterns, link-budget functions, etc.) should be provided via the
    strategy's ``__init__`` and stored on ``self``.
    """

    @abstractmethod
    def calculate(
        self,
        satellite_trajectory: SatelliteTrajectory,
        antenna_trajectory: AntennaTrajectory,
        facility: Facility,
        frequency: FrequencyRange,
    ) -> InterferenceResult | None:
        """Analyze a single satellite pass for interference.

        Args:
            satellite_trajectory: The satellite's path (times, az, alt, distance).
            antenna_trajectory: Where the antenna is pointing over time.
            facility: Telescope location and parameters.
            frequency: Observation frequency.

        Returns:
            InterferenceResult if interference detected, None otherwise.
        """


class GeometricStrategy(InterferenceStrategy):
    """Binary in/out-of-beam detection.

    Determines interference by checking whether the angular separation
    between the satellite and antenna boresight is less than the beam
    radius. This is the original SOPP behavior.
    """

    def calculate(
        self,
        satellite_trajectory: SatelliteTrajectory,
        antenna_trajectory: AntennaTrajectory,
        facility: Facility,
        frequency: FrequencyRange,
    ) -> InterferenceResult | None:
        ant_az, ant_alt = antenna_trajectory.get_state_at(satellite_trajectory.times)

        sep_sq = calculate_angular_separation_sq(
            az1=satellite_trajectory.azimuth,
            alt1=satellite_trajectory.altitude,
            az2=ant_az,
            alt2=ant_alt,
        )

        mask = sep_sq <= facility.beam_radius**2

        if not np.any(mask):
            return None

        return InterferenceResult(
            trajectory=SatelliteTrajectory(
                satellite=satellite_trajectory.satellite,
                times=satellite_trajectory.times[mask],
                azimuth=satellite_trajectory.azimuth[mask],
                altitude=satellite_trajectory.altitude[mask],
                distance_km=satellite_trajectory.distance_km[mask],
            ),
        )
