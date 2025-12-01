from abc import ABC, abstractmethod

from sopp.models import (
    OverheadWindow,
    PositionTime,
    Reservation,
    RuntimeSettings,
    Satellite,
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
        runtime_settings: RuntimeSettings | None = None,
    ):
        self.antenna_direction_path = antenna_direction_path
        self.list_of_satellites = list_of_satellites
        self.reservation = reservation
        self.runtime_settings = (
            runtime_settings if runtime_settings is not None else RuntimeSettings()
        )

    @abstractmethod
    def get_satellites_above_horizon(self) -> list[OverheadWindow]:
        pass

    @abstractmethod
    def get_satellites_crossing_main_beam(self) -> list[OverheadWindow]:
        pass
