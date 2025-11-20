from datetime import datetime, timedelta, timezone

from sopp.models.coordinates import Coordinates
from sopp.models.facility import Facility
from sopp.models.overhead_window import OverheadWindow
from sopp.models.position import Position
from sopp.models.position_time import PositionTime

ARBITRARY_FACILITY = Facility(coordinates=Coordinates(latitude=0, longitude=0))

ARBITRARY_ANTENNA_POSITION = PositionTime(
    position=Position(altitude=100, azimuth=100), time=datetime.now(tz=timezone.utc)
)

ARBITRARY_SATELLITE_POSITION = PositionTime(
    position=Position(altitude=100, azimuth=100), time=datetime.now(tz=timezone.utc)
)


def create_overhead_window(satellite, altitude, azimuth, start_time, num_positions):
    positions = [
        PositionTime(
            position=Position(altitude=altitude, azimuth=azimuth),
            time=start_time + timedelta(seconds=i),
        )
        for i in range(num_positions)
    ]

    return OverheadWindow(satellite=satellite, positions=positions)


def create_expected_windows(expected_positions):
    return [
        [position] if isinstance(position, PositionTime) else position
        for position in expected_positions
    ]


def assert_windows_eq(actual_windows, expected_windows):
    for actual_window, expected_window in zip(
        actual_windows, expected_windows, strict=False
    ):
        assert actual_window == expected_window
