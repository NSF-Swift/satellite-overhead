from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import List, Type

from satellite_determination.custom_dataclasses.observation_path import ObservationPath
from satellite_determination.custom_dataclasses.overhead_window import OverheadWindow
from satellite_determination.custom_dataclasses.reservation import Reservation
from satellite_determination.custom_dataclasses.satellite.satellite import Satellite
from satellite_determination.custom_dataclasses.time_window import TimeWindow
from satellite_determination.event_finder.event_finder import EventFinder
from satellite_determination.event_finder.event_finder_rhodesmill import EventFinderRhodesMill
from satellite_determination.frequency_filter.frequency_filter import FrequencyFilter
from satellite_determination.path_finder.observation_path_finder import ObservationPathFinder


@dataclass
class MainResults:
    satellites_above_horizon: List[OverheadWindow]
    interference_windows: List[OverheadWindow]


class Main:
    def __init__(self,
                 reservation: Reservation,
                 search_window: TimeWindow,
                 satellites: List[Satellite]):
        self._reservation = reservation
        self._search_window = search_window
        self._satellites = satellites

    def run(self) -> MainResults:
        return MainResults(
            satellites_above_horizon=self._event_finder.get_overhead_windows(),
            interference_windows=self._event_finder.get_overhead_windows_slew()
        )

    @cached_property
    def _event_finder(self) -> EventFinder:
        return EventFinderRhodesMill(list_of_satellites=self._frequency_filtered_satellites,
                                     reservation=self._reservation,
                                     azimuth_altitude_path=self._altitude_azimuth_pairs,
                                     search_window=self._search_window)

    @property
    def _frequency_filtered_satellites(self) -> List[Satellite]:
        return FrequencyFilter(satellites=self._satellites,
                               observation_frequency=self._reservation.frequency).filter_frequencies()

    @property
    def _altitude_azimuth_pairs(self) -> List[ObservationPath]:
        return ObservationPathFinder(reservation=self._reservation, time_window=self._reservation.time).calculate_path()
