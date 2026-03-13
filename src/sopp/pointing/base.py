"""Abstract base for antenna pointing calculators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sopp.models.core import TimeWindow
    from sopp.models.ground.facility import Facility
    from sopp.models.ground.target import ObservationTarget
    from sopp.models.ground.trajectory import AntennaTrajectory


class PointingCalculator(ABC):
    """Computes where the antenna points over time for a given celestial target."""

    def __init__(
        self,
        facility: Facility,
        observation_target: ObservationTarget,
        time_window: TimeWindow,
    ):
        self._facility = facility
        self._observation_target = observation_target
        self._time_window = time_window

    @abstractmethod
    def calculate_trajectory(
        self, resolution_seconds: float = 1.0, time_grid: np.ndarray | None = None
    ) -> AntennaTrajectory:
        """Compute the antenna az/alt trajectory.

        Args:
            resolution_seconds: Time step if generating a new grid.
            time_grid: Pre-computed time grid. If provided, resolution_seconds
                is ignored.

        Returns:
            AntennaTrajectory with positions at each time step.
        """
        pass
