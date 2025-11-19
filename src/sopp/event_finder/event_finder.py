from abc import ABC, abstractmethod

from sopp.models.overhead_window import OverheadWindow
from sopp.models.position_time import PositionTime
from sopp.models.reservation import Reservation
from sopp.models.runtime_settings import RuntimeSettings
from sopp.models.satellite.satellite import Satellite
from sopp.event_finder.event_finder_rhodesmill.support.satellite_positions_with_respect_to_facility_retriever.satellite_positions_with_respect_to_facility_retriever import (
    SatellitePositionsWithRespectToFacilityRetriever,
)


class EventFinder(ABC):
    """
    The EventFinder is the module that determines if a satellite interferes with an RA observation. It has three functions:

      + get_satellites_crossing_main_beam():    determines if a satellite crosses the telescope's main beam as the telescope moves across the sky
                                        by looking for intersections of azimuth and altitude and returning a list of OverheadWindows for
                                        events where this occurs
      + get_satellites_above_horizon():         Determines the satellites visible above the horizon during the search window and returns a list of
                                        OverheadWindows for each event. This can be used to find all satellite visible over the horizon or
                                        to determine events for a stationary observation if an azimuth and altitude is provided

    """

    def __init__(
        self,
        antenna_direction_path: list[PositionTime],
        list_of_satellites: list[Satellite],
        reservation: Reservation,
        satellite_positions_with_respect_to_facility_retriever_class: type[
            SatellitePositionsWithRespectToFacilityRetriever
        ],
        runtime_settings: RuntimeSettings = RuntimeSettings(),
    ):
        self.antenna_direction_path = antenna_direction_path
        self.list_of_satellites = list_of_satellites
        self.reservation = reservation
        self.satellite_positions_with_respect_to_facility_retriever_class = (
            satellite_positions_with_respect_to_facility_retriever_class
        )
        self.runtime_settings = runtime_settings

    @abstractmethod
    def get_satellites_above_horizon(self) -> list[OverheadWindow]:
        pass

    @abstractmethod
    def get_satellites_crossing_main_beam(self) -> list[OverheadWindow]:
        pass
