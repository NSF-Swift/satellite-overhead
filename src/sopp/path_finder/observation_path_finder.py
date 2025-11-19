from abc import ABC, abstractmethod

from sopp.models.facility import Facility
from sopp.models.observation_target import ObservationTarget
from sopp.models.position_time import PositionTime
from sopp.models.time_window import TimeWindow


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
    def calculate_path(self) -> list[PositionTime]:
        pass
