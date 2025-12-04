from sopp.analysis.event_finders.validator import Validator

from sopp.models.satellite_trajectory import SatelliteTrajectory
from sopp.models.reservation import Reservation
from sopp.models.satellite.satellite import Satellite
from sopp.models.time_window import TimeWindow


class ValidatorSatellitesAreOverheadAtSpecificTimes(Validator):
    def __init__(self, overhead_times: list[TimeWindow]):
        self._overhead_times = overhead_times

    def get_overhead_windows(
        self, list_of_satellites: list[Satellite], reservation: Reservation
    ) -> list[SatelliteTrajectory]:
        return [
            SatelliteTrajectory(satellite=satellite, overhead_time=overhead_time)
            for satellite, overhead_time in zip(
                list_of_satellites, self._overhead_times, strict=False
            )
            if overhead_time.overlaps(reservation.time)
        ]
