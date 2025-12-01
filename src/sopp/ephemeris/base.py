from abc import ABC, abstractmethod
from datetime import datetime

from sopp.models import Satellite, PositionTime, TimeWindow


class EphemerisCalculator(ABC):
    """
    Abstract contract for the physics engine.
    Calculates orbital mechanics, coordinates, and events.
    """

    @abstractmethod
    def find_events(
        self,
        satellite: Satellite,
        min_altitude: float,
        start_time: datetime,
        end_time: datetime,
    ) -> list[TimeWindow]:
        """
        Returns TimeWindows for Rise/Set events.
        """
        pass

    @abstractmethod
    def get_position_at(self, satellite: Satellite, time: datetime) -> PositionTime:
        """
        Returns the position of the satellite relative to the facility at a specific instant.
        """
        pass

    @abstractmethod
    def get_positions_window(
        self, satellite: Satellite, start: datetime, end: datetime
    ) -> list[PositionTime]:
        """
        Returns positions for all intervaled time points within the time window.
        """
        pass
