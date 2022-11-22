from typing import List

from satellite_determination.dataclasses.overhead_window import OverheadWindow
from satellite_determination.dataclasses.reservation import Reservation
from satellite_determination.dataclasses.satellite import Satellite
from satellite_determination.dataclasses.time_window import TimeWindow
from satellite_determination.validator.validator import Validator


class ValidatorSatellitesAreOverheadAtSpecificTimes(Validator):
    def __init__(self, overhead_times: List[TimeWindow]):
        self._overhead_times = overhead_times

    def overhead_list(self, list_of_satellites: List[Satellite], reservation: Reservation) -> List[OverheadWindow]:
        return [OverheadWindow(satellite=satellite, overhead_time=overhead_time)
                for satellite, overhead_time in zip(list_of_satellites, self._overhead_times)
                if overhead_time.overlaps(reservation.time)]
