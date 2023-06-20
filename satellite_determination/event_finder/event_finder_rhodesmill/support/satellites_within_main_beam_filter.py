import itertools
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from math import isclose
from typing import List

import numpy

from satellite_determination.custom_dataclasses.facility import Facility
from satellite_determination.custom_dataclasses.position_time import PositionTime
from satellite_determination.custom_dataclasses.time_window import TimeWindow
from satellite_determination.utilities import convert_datetime_to_utc


DEGREES_IN_A_CIRCLE = 360


@dataclass
class AntennaPosition:
    satellite_positions: List[PositionTime]
    antenna_direction: PositionTime


@dataclass
class EnterAndExitEvents:
    enter: List[datetime]
    exit: List[datetime]


class SatellitesWithinMainBeamFilter:
    def __init__(self,
                 facility: Facility,
                 antenna_positions: List[AntennaPosition],
                 cutoff_time: datetime):
        self._cutoff_time = cutoff_time
        self._facility = facility
        self._antenna_positions = antenna_positions
        self._previously_in_view = False

    def run(self) -> List[TimeWindow]:
        enter_events = []
        exit_events = []
        for antenna_position in self._antenna_positions_by_time:
            satellite_positions = self._satellite_position_above_the_horizon(antenna_position=antenna_position)
            for satellite_position in self._sort_satellite_positions_by_time(satellite_positions=satellite_positions):
                timestamp = convert_datetime_to_utc(satellite_position.time)
                is_within_beam_width_altitude = isclose(satellite_position.altitude,
                                                        antenna_position.antenna_direction.altitude,
                                                        abs_tol=self._facility.half_beamwidth)
                now_in_view = is_within_beam_width_altitude \
                              and self._is_within_beam_with_azimuth(satellite_azimuth=satellite_position.azimuth,
                                                                    antenna_azimuth=antenna_position.antenna_direction.azimuth)
                if now_in_view and not self._previously_in_view:
                    enter_events.append(timestamp)
                    self._previously_in_view = True
                elif not now_in_view and self._previously_in_view:
                    exit_events.append(timestamp)
                    self._previously_in_view = False
        exit_events.append(self._cutoff_time)
        return [TimeWindow(begin=convert_datetime_to_utc(begin_event),
                           end=convert_datetime_to_utc(exit_event)) for begin_event, exit_event in zip(enter_events, exit_events)]

    @cached_property
    def _antenna_positions_by_time(self) -> List[AntennaPosition]:
        return sorted(self._antenna_positions, key=lambda x: x.antenna_direction.time)

    def _satellite_position_above_the_horizon(self, antenna_position: AntennaPosition) -> List[PositionTime]:
        return [position for position in antenna_position.satellite_positions
                if position.altitude >= 0 and position.time < self._cutoff_time]

    @staticmethod
    def _sort_satellite_positions_by_time(satellite_positions: List[PositionTime]) -> List[PositionTime]:
        return sorted(satellite_positions, key=lambda x: x.time)

    def _is_within_beam_with_azimuth(self, satellite_azimuth: float, antenna_azimuth: float) -> bool:
        positions_to_compare_original = [satellite_azimuth, antenna_azimuth]
        positions_to_compare_next_modulus = (numpy.array(positions_to_compare_original) + DEGREES_IN_A_CIRCLE).tolist()
        positions_to_compare = itertools.combinations(positions_to_compare_original + positions_to_compare_next_modulus, 2)
        return any([isclose(*positions, abs_tol=self._facility.half_beamwidth) for positions in positions_to_compare])
