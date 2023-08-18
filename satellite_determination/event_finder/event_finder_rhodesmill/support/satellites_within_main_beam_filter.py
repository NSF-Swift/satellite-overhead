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
            for satellite_position in self._sort_satellite_positions_by_time(satellite_positions=antenna_position.satellite_positions):
                if satellite_position.time >= self._cutoff_time:
                    break
                timestamp = convert_datetime_to_utc(satellite_position.time)
                now_in_view = self._is_within_beam_width_altitude(satellite_altitude=satellite_position.position.altitude,
                                                                  antenna_altitude=antenna_position.antenna_direction.position.altitude) \
                              and self._is_within_beam_with_azimuth(satellite_azimuth=satellite_position.position.azimuth,
                                                                    antenna_azimuth=antenna_position.antenna_direction.position.azimuth)
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

    @staticmethod
    def _sort_satellite_positions_by_time(satellite_positions: List[PositionTime]) -> List[PositionTime]:
        return sorted(satellite_positions, key=lambda x: x.time)

    def _is_within_beam_width_altitude(self, satellite_altitude: float, antenna_altitude: float) -> bool:
        # Check if satellite is above horizon
        if satellite_altitude < 0:
            return False 
        else:
            lowest_main_beam_altitude = antenna_altitude - self._facility.half_beamwidth
            return satellite_altitude >= lowest_main_beam_altitude

    def _is_within_beam_with_azimuth(self, satellite_azimuth: float, antenna_azimuth: float) -> bool:
        positions_to_compare_original = [satellite_azimuth, antenna_azimuth]
        positions_to_compare_next_modulus = (numpy.array(positions_to_compare_original) + DEGREES_IN_A_CIRCLE).tolist()
        positions_to_compare = itertools.combinations(positions_to_compare_original + positions_to_compare_next_modulus, 2)
        return any([isclose(*positions, abs_tol=self._facility.half_beamwidth) for positions in positions_to_compare])
