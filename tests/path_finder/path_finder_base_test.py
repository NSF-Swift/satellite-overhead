import pytest
from sopp.models.coordinates import Coordinates
from sopp.models.facility import Facility
from sopp.models.observation_target import ObservationTarget
from sopp.models.time_window import TimeWindow
from sopp.utils.helpers import read_datetime_string_as_utc


class PathFinderBaseTest:
    @pytest.mark.parametrize(
        "begin_time, end_time, expected_altitude, expected_azimuth",
        [
            ("2023-09-23T05:53:00.000000", "2023-09-23T05:54:00.000000", 0, 90),
            ("2023-09-23T11:53:00.000000", "2023-09-23T11:54:00.000000", 90, 191),
            ("2023-09-23T17:53:00.000000", "2023-09-23T17:54:00.000000", 0, 270),
        ],
    )
    def test_expected_path(
        self, begin_time, end_time, expected_altitude, expected_azimuth
    ):
        facility = Facility(
            Coordinates(
                latitude=0.0,
                longitude=0.0,
            ),
            elevation=0,
            name="Null Island",
        )

        time_window = TimeWindow(
            begin=read_datetime_string_as_utc(begin_time),
            end=read_datetime_string_as_utc(end_time),
        )

        obs_target = ObservationTarget(declination="0d0m0s", right_ascension="12h0m0s")

        path_finder = self.PathFinderClass(facility, obs_target, time_window)
        path = path_finder.calculate_path()

        self._assert_eq(
            actual=path[0].position.altitude, expected=expected_altitude, tolerance=0.5
        )
        self._assert_eq(
            actual=path[0].position.azimuth, expected=expected_azimuth, tolerance=0.5
        )

    def _assert_eq(self, actual, expected, tolerance=0.05):
        assert abs(actual - expected) <= tolerance
