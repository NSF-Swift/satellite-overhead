from abc import ABC, abstractmethod
from datetime import datetime

import numpy as np
import numpy.typing as npt

from sopp.models.core import Position, TimeWindow
from sopp.models.ground.facility import Facility
from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.trajectory import SatelliteTrajectory


class EphemerisCalculator(ABC):
    """
    Abstract contract for the physics engine.
    """

    @abstractmethod
    def __init__(
        self,
        facility: Facility,
        datetimes: npt.NDArray[np.object_],  # datetime objects
    ):
        """
        Initializes the calculator.

        Args:
            facility: The location observing the satellites.
            datetimes: A sorted NumPy array of UTC datetime objects representing
                       the simulation master grid.
        """
        pass

    @abstractmethod
    def calculate_visibility_windows(
        self,
        satellite: Satellite,
        min_altitude: float,
        start_time: datetime,
        end_time: datetime,
    ) -> list[TimeWindow]:
        """
        Calculates TimeWindows for Rise/Set events (where alt > min_alt).
        """
        pass

        pass

    @abstractmethod
    def calculate_trajectory(
        self, satellite: Satellite, start: datetime, end: datetime
    ) -> SatelliteTrajectory:
        """
        Calculates the continuous path for a single time window (Vector).
        """
        pass

    @abstractmethod
    def calculate_trajectories(
        self, satellite: Satellite, windows: list[TimeWindow]
    ) -> list[SatelliteTrajectory]:
        """
        Calculates paths for multiple disjoint windows in one batch operation.
        """
        pass

    @abstractmethod
    def calculate_position(self, satellite: Satellite, time: datetime) -> Position:
        """
        Calculates the position for a specific instant (Scalar).
        """
        pass
