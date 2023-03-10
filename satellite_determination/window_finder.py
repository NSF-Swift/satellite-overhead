from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

from satellite_determination.custom_dataclasses.overhead_window import OverheadWindow
from satellite_determination.custom_dataclasses.reservation import Reservation
from satellite_determination.custom_dataclasses.satellite.satellite import Satellite
from satellite_determination.custom_dataclasses.time_window import TimeWindow
from satellite_determination.event_finder.validator import Validator


@dataclass
class SuggestedReservation:
    ideal_reservation: Reservation
    overhead_satellites: List[OverheadWindow]
    suggested_start_time: datetime


class WindowFinder:
    def __init__(self,
                 ideal_reservation: Reservation,
                 satellites: List[Satellite],
                 validator: Validator,
                 search_window: timedelta = timedelta(weeks=1),
                 start_time_increments: timedelta = timedelta(minutes=15)):
        self._ideal_reservation = ideal_reservation
        self._satellites = satellites
        self._search_window = search_window
        self._start_time_increments = start_time_increments
        self._validator = validator

    def find(self) -> List[SuggestedReservation]:
        potential_time_windows = [TimeWindow(begin=start_time, end=start_time + self._ideal_reservation.time.duration)
                                  for start_time in self._potential_start_times]
        potential_reservations = [Reservation(facility=self._ideal_reservation.facility, time=time) for time in potential_time_windows]
        overhead_satellites = [self._satellites_overhead(reservation=reservation) for reservation in potential_reservations]
        sort_indices = sorted(range(len(potential_reservations)), key=lambda index: len(overhead_satellites[index]))
        return [
            SuggestedReservation(
                ideal_reservation=self._ideal_reservation,
                overhead_satellites=overhead_satellites[index],
                suggested_start_time=potential_reservations[index].time.begin
            )
            for index in sort_indices
        ]

    def _satellites_overhead(self, reservation: Reservation) -> List[OverheadWindow]:
        return self._validator.overhead_list(list_of_satellites=self._satellites,
                                             reservation=reservation)

    @property
    def _potential_start_times(self) -> List[datetime]:
        number_of_slots = int(self._search_window.total_seconds() / self._start_time_increments.total_seconds())
        return [self._get_time_slot(i) for i in range(1, number_of_slots + 1)]

    def _get_time_slot(self, index: int) -> datetime:
        backwards_forwards = (-1) ** index
        multiplier = int(index / 2)
        return self._ideal_reservation.time.begin + backwards_forwards * self._start_time_increments * multiplier
