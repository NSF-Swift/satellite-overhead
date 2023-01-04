from datetime import datetime, timedelta
from typing import List

from satellite_determination.dataclasses.frequency_range import FrequencyRange
from satellite_determination.dataclasses.overhead_window import OverheadWindow
from satellite_determination.dataclasses.reservation import Reservation
from satellite_determination.dataclasses.satellite import Satellite
from satellite_determination.dataclasses.time_window import TimeWindow
from satellite_determination.window_finder import SuggestedReservation, WindowFinder
from tests.window_finder.definitions import ARBITRARY_FACILITY
from tests.window_finder.support.validator_satellites_are_overhead_at_specific_times import \
    ValidatorSatellitesAreOverheadAtSpecificTimes


_ARBITRARY_FREQUENCY_RANGE = FrequencyRange(high_in_megahertz=2., low_in_megahertz=1.)
_ARBITRARY_SATELLITE_NUMBER = 'satellite_number'


class TestSortedByLeastNumberOfSatellites:
    def test(self):
        suggestions = WindowFinder(
            ideal_reservation=self._ideal_reservation,
            satellites=[window.satellite for window in self._overhead_windows],
            validator=ValidatorSatellitesAreOverheadAtSpecificTimes(
                overhead_times=[window.overhead_time for window in self._overhead_windows]),
            start_time_increments=timedelta(days=1),
            search_window=timedelta(weeks=1)
        ).find()
        assert suggestions == self._expected_suggestions

    @property
    def _overhead_windows(self) -> List[OverheadWindow]:
        return self._two_overhead_windows_on_ideal_reservation + self._one_overhead_window_on_second_closest_reservation

    @property
    def _expected_suggestions(self) -> List[SuggestedReservation]:
        ideal_reservation_index = len(self._expected_suggestion_start_times) - 1
        second_closest_reservation_index = ideal_reservation_index - 1
        return [
            SuggestedReservation(
                suggested_start_time=start_time,
                ideal_reservation=self._ideal_reservation,
                overhead_satellites=self._two_overhead_windows_on_ideal_reservation
                    if index == ideal_reservation_index
                    else self._one_overhead_window_on_second_closest_reservation
                        if index == second_closest_reservation_index
                        else []
            )
            for index, start_time in enumerate(self._expected_suggestion_start_times)
        ]

    @property
    def _ideal_reservation(self) -> Reservation:
        return Reservation(
            facility=ARBITRARY_FACILITY,
            time=TimeWindow(begin=datetime(year=2022, month=11, day=20), end=datetime(year=2022, month=11, day=21)))

    @property
    def _two_overhead_windows_on_ideal_reservation(self) -> List[OverheadWindow]:
        return [
            OverheadWindow(satellite=Satellite(frequency=_ARBITRARY_FREQUENCY_RANGE,
                                               satellite_number=_ARBITRARY_SATELLITE_NUMBER,
                                               name='name1'),
                           overhead_time=TimeWindow(begin=datetime(year=2022, month=11, day=20),
                                                    end=datetime(year=2022, month=11, day=20, hour=1))),
            OverheadWindow(satellite=Satellite(frequency=_ARBITRARY_FREQUENCY_RANGE,
                                               satellite_number=_ARBITRARY_SATELLITE_NUMBER,
                                               name='name2'),
                           overhead_time=TimeWindow(begin=datetime(year=2022, month=11, day=20),
                                                    end=datetime(year=2022, month=11, day=20, hour=1)))
        ]

    @property
    def _one_overhead_window_on_second_closest_reservation(self) -> List[OverheadWindow]:
        return [
            OverheadWindow(satellite=Satellite(frequency=_ARBITRARY_FREQUENCY_RANGE,
                                               satellite_number=_ARBITRARY_SATELLITE_NUMBER,
                                               name='name3'),
                           overhead_time=TimeWindow(begin=datetime(year=2022, month=11, day=21),
                                                    end=datetime(year=2022, month=11, day=21, hour=1))),
        ]

    @property
    def _expected_suggestion_start_times(self) -> List[datetime]:
        return [
            datetime(year=2022, month=11, day=19),
            datetime(year=2022, month=11, day=22),
            datetime(year=2022, month=11, day=18),
            datetime(year=2022, month=11, day=23),
            datetime(year=2022, month=11, day=17),
            datetime(year=2022, month=11, day=21),
            datetime(year=2022, month=11, day=20)
        ]
