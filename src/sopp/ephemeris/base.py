from abc import ABC, abstractmethod
from datetime import datetime

from sopp.models import Position, Satellite, SatelliteTrajectory, TimeWindow


class EphemerisCalculator(ABC):
    """
    Abstract contract for the physics engine.
    """

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
