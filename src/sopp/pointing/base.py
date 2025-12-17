from abc import ABC, abstractmethod

import numpy as np

from sopp.models.core import TimeWindow
from sopp.models.ground.facility import Facility
from sopp.models.ground.target import ObservationTarget
from sopp.models.ground.trajectory import AntennaTrajectory


class ObservationPathFinder(ABC):
    """
    The ObservationPathFinder determines the path the telescope will need to follow to track its target and returns
    a list of altitude, azimuth, and timestamp to represent the telescope's movement. It uses the observation
    target's right ascension and declination to determine this path.
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
    def calculate_path(
        self, resolution_seconds: float = 1.0, time_grid: np.ndarray | None = None
    ) -> AntennaTrajectory:
        pass
