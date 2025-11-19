from sopp.models.overhead_window import OverheadWindow
from sopp.models.reservation import Reservation
from sopp.models.satellite.satellite import Satellite
from sopp.models.time_window import TimeWindow
from sopp.event_finder.validator import Validator


class ValidatorSatellitesAreOverheadAtSpecificTimes(Validator):
    def __init__(self, overhead_times: list[TimeWindow]):
        self._overhead_times = overhead_times

    def get_overhead_windows(
        self, list_of_satellites: list[Satellite], reservation: Reservation
    ) -> list[OverheadWindow]:
        return [
            OverheadWindow(satellite=satellite, overhead_time=overhead_time)
            for satellite, overhead_time in zip(
                list_of_satellites, self._overhead_times, strict=False
            )
            if overhead_time.overlaps(reservation.time)
        ]
