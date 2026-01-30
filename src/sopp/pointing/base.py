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
    """
    Abstract base for engines that calculate where the antenna is pointing over time.
    """

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
        """
        Generates the vector of Az/Alt positions for the antenna.
        """
        pass
