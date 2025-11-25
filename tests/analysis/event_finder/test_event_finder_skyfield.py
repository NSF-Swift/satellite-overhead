import pytest
from datetime import timedelta
from sopp.models.position import Position
from sopp.models.position_time import PositionTime
from sopp.models.satellite.satellite import Satellite
from tests.definitions import SMALL_EPSILON

from tests.conftest import ARBITRARY_ALTITUDE, ARBITRARY_AZIMUTH


@pytest.fixture
def static_antenna_path(start_time, time_window_duration):
    """
    An antenna that stays pointed at (0,0) for the whole duration.
    """
    steps = int(time_window_duration.total_seconds()) + 1
    return [
        PositionTime(
            position=Position(altitude=ARBITRARY_ALTITUDE, azimuth=ARBITRARY_AZIMUTH),
            time=start_time + timedelta(seconds=i),
        )
        for i in range(steps)
    ]


def test_single_satellite_match(
    make_event_finder, satellite, static_antenna_path, start_time, reservation
):
    """
    Scenario: 1 Satellite, Antenna matches perfectly for the whole duration.
    """
    finder = make_event_finder(
        satellites=[satellite],
        antenna_path=static_antenna_path,
        reservation=reservation,
    )

    windows = finder.get_satellites_crossing_main_beam()

    assert len(windows) == 1
    assert windows[0].satellite.name == satellite.name
    assert windows[0].overhead_time.begin == start_time


def test_multiple_satellites_match(make_event_finder, static_antenna_path, reservation):
    """
    Scenario: 2 Satellites, Antenna matches perfectly.
    """
    sats = [Satellite(name="SAT-A"), Satellite(name="SAT-B")]
    finder = make_event_finder(
        satellites=sats, antenna_path=static_antenna_path, reservation=reservation
    )

    windows = finder.get_satellites_crossing_main_beam()

    assert len(windows) == 2
    names_found = {w.satellite.name for w in windows}
    assert names_found == {"SAT-A", "SAT-B"}


def test_antenna_movement_breaks_window(
    make_event_finder, satellite, start_time, reservation
):
    """
    Scenario: Antenna moves OUT of alignment in the middle of the pass.
    Timeline:
      T+0: Match (0,0)
      T+1: Mismatch (Gap created)
      T+2: Match (0,0)
    """
    # Calculate an altitude that is definitely outside the beamwidth
    bad_alt = ARBITRARY_ALTITUDE + reservation.facility.half_beamwidth + SMALL_EPSILON

    # Construct specific path for this test
    antenna_path = [
        PositionTime(
            Position(altitude=ARBITRARY_ALTITUDE, azimuth=ARBITRARY_AZIMUTH),
            time=start_time,
        ),
        PositionTime(
            Position(altitude=bad_alt, azimuth=ARBITRARY_AZIMUTH),
            time=start_time + timedelta(seconds=1),
        ),
        PositionTime(
            Position(altitude=ARBITRARY_ALTITUDE, azimuth=ARBITRARY_AZIMUTH),
            time=start_time + timedelta(seconds=2),
        ),
        # Pad end
        PositionTime(
            Position(altitude=ARBITRARY_ALTITUDE, azimuth=ARBITRARY_AZIMUTH),
            time=start_time + timedelta(seconds=3),
        ),
    ]

    finder = make_event_finder(
        satellites=[satellite], antenna_path=antenna_path, reservation=reservation
    )

    windows = finder.get_satellites_crossing_main_beam()

    # We expect 2 windows because of the gap at T+1
    assert len(windows) == 2
    assert windows[0].overhead_time.begin == start_time
    assert windows[1].overhead_time.begin == start_time + timedelta(seconds=2)
