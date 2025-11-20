from datetime import datetime, timezone
from functools import cached_property

from numpy._typing import NDArray
from skyfield.api import load, wgs84
from skyfield.timelib import Time, Timescale

from sopp.models.frequency_range import FrequencyRange
from sopp.models.satellite.international_designator import (
    InternationalDesignator,
)
from sopp.models.satellite.mean_motion import MeanMotion
from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.tle_information import TleInformation


class TestRhodesmillIntegration:
    _MINUTE_BEFORE_ENTERS = 34
    _MINUTES_AFTER_ENTERS = _MINUTE_BEFORE_ENTERS + 1
    _MINUTE_BEFORE_CULMINATES = 40
    _MINUTE_AFTER_CULMINATES = _MINUTE_BEFORE_CULMINATES + 1
    _MINUTE_BEFORE_LEAVES = 47

    def test_events_found_on_window_that_encompasses_only_leaves(self):
        assert self._get_events(
            minute_begin=self._MINUTE_AFTER_CULMINATES,
            minute_end=self._MINUTE_BEFORE_LEAVES,
        ).tolist() == [2]

    def test_events_found_on_window_that_is_between_enter_and_culminates(self):
        assert (
            self._get_events(
                minute_begin=self._MINUTES_AFTER_ENTERS,
                minute_end=self._MINUTE_BEFORE_CULMINATES,
            ).tolist()
            == []
        )

    def test_events_found_on_window_that_encompasses_only_enters(self):
        assert self._get_events(
            minute_begin=self._MINUTE_BEFORE_ENTERS,
            minute_end=self._MINUTE_BEFORE_CULMINATES,
        ).tolist() == [0]

    def test_events_found_on_window_that_encompasses_only_culminates(self):
        assert self._get_events(
            minute_begin=self._MINUTE_BEFORE_CULMINATES,
            minute_end=self._MINUTE_AFTER_CULMINATES,
        ).tolist() == [1]

    def test_events_found_on_window_that_encompasses_culminates_and_leaves(self):
        assert self._get_events(
            minute_begin=self._MINUTE_BEFORE_CULMINATES,
            minute_end=self._MINUTE_BEFORE_LEAVES,
        ).tolist() == [1, 2]

    def test_events_found_on_window_that_encompasses_culminates_and_enters(self):
        assert self._get_events(
            minute_begin=self._MINUTE_BEFORE_ENTERS,
            minute_end=self._MINUTE_AFTER_CULMINATES,
        ).tolist() == [0, 1]

    def test_events_found_on_window_that_encompasses_full_satellite_pass(self):
        assert self._get_events(
            minute_begin=self._MINUTE_BEFORE_ENTERS,
            minute_end=self._MINUTE_BEFORE_LEAVES,
        ).tolist() == [0, 1, 2]

    def _get_events(self, minute_begin: int, minute_end: int) -> NDArray[int]:
        time_begin = self._datetime_to_rhodesmill_time(minute=minute_begin)
        time_end = self._datetime_to_rhodesmill_time(minute=minute_end)
        coordinates = wgs84.latlon(40.8178049, -121.4695413)
        rhodesmill_earthsat = self._satellite.to_rhodesmill()
        event_times, events = rhodesmill_earthsat.find_events(
            topos=coordinates, t0=time_begin, t1=time_end, altitude_degrees=0
        )
        return events

    def _datetime_to_rhodesmill_time(self, minute: int) -> Time:
        return self._rhodesmill_timescale.from_datetime(
            datetime(2023, 3, 30, 12, minute, tzinfo=timezone.utc)
        )

    @cached_property
    def _rhodesmill_timescale(self) -> Timescale:
        return load.timescale()

    @property
    def _satellite(self) -> Satellite:
        return Satellite(
            name="SAUDISAT 2",
            tle_information=TleInformation(
                argument_of_perigee=2.6581678667138995,
                drag_coefficient=8.4378e-05,
                eccentricity=0.0025973,
                epoch_days=26801.46955532,
                inclination=1.7179345640550268,
                international_designator=InternationalDesignator(
                    year=4, launch_number=25, launch_piece="F"
                ),
                mean_anomaly=3.6295308619113436,
                mean_motion=MeanMotion(
                    first_derivative=9.605371056982682e-12,
                    second_derivative=0.0,
                    value=0.06348248105551128,
                ),
                revolution_number=200,
                right_ascension_of_ascending_node=1.7778098293739442,
                satellite_number=28371,
                classification="U",
            ),
            frequency=[
                FrequencyRange(frequency=137.513, bandwidth=None, status="active")
            ],
        )
