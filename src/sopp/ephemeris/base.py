from abc import ABC, abstractmethod
from datetime import datetime

from sopp.models.facility import Facility
from sopp.models.position_time import PositionTime
from sopp.models.satellite.satellite import Satellite


class SatellitePositionsWithRespectToFacilityRetriever(ABC):
    def __init__(self, facility: Facility, datetimes: list[datetime]):
        self._datetimes = datetimes
        self._facility = facility

    @abstractmethod
    def run(self, satellite: Satellite) -> list[PositionTime]:
        pass
