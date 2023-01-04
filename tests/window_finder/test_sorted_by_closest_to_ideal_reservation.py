from datetime import datetime, timedelta
from typing import List

from satellite_determination.dataclasses.reservation import Reservation
from satellite_determination.dataclasses.time_window import TimeWindow
from satellite_determination.window_finder import SuggestedReservation, WindowFinder
from tests.window_finder.definitions import ARBITRARY_FACILITY
from tests.window_finder.support.validator_satellites_are_overhead_at_specific_times import \
    ValidatorSatellitesAreOverheadAtSpecificTimes


class TestSortedByClosestToIdealReservation:
    def test(self):
        suggestions = WindowFinder(
            ideal_reservation=self._ideal_reservation,
            satellites=[],
            validator=ValidatorSatellitesAreOverheadAtSpecificTimes(overhead_times=[]),
            start_time_increments=timedelta(days=1),
            search_window=timedelta(weeks=1)
        ).find()
        assert suggestions == self._expected_suggestions

    @property
    def _expected_suggestions(self) -> List[SuggestedReservation]:
        return [
            SuggestedReservation(
                suggested_start_time=start_time,
                ideal_reservation=self._ideal_reservation,
                overhead_satellites=[]
            )
            for start_time in self._expected_suggestion_start_times
        ]

    @property
    def _ideal_reservation(self) -> Reservation:
        return Reservation(
            facility=ARBITRARY_FACILITY,
            time=TimeWindow(begin=datetime(year=2022, month=11, day=20), end=datetime(year=2022, month=11, day=21)))

    @property
    def _expected_suggestion_start_times(self) -> List[datetime]:
        return [
            datetime(year=2022, month=11, day=20),
            datetime(year=2022, month=11, day=21),
            datetime(year=2022, month=11, day=19),
            datetime(year=2022, month=11, day=22),
            datetime(year=2022, month=11, day=18),
            datetime(year=2022, month=11, day=23),
            datetime(year=2022, month=11, day=17)
        ]
