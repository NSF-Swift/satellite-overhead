from datetime import datetime, timezone

import pytest
from skyfield.api import load, wgs84

from sopp.models.frequency_range import FrequencyRange
from sopp.models.satellite.international_designator import InternationalDesignator
from sopp.models.satellite.mean_motion import MeanMotion
from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.tle_information import TleInformation

# Constants
MINUTE_BEFORE_ENTERS = 34
MINUTES_AFTER_ENTERS = MINUTE_BEFORE_ENTERS + 1
MINUTE_BEFORE_CULMINATES = 40
MINUTE_AFTER_CULMINATES = MINUTE_BEFORE_CULMINATES + 1
MINUTE_BEFORE_LEAVES = 47

# Events
RISE = 0
CULMINATE = 1
SET = 2


def test_events_found_on_window_that_encompasses_only_leaves(skyfield_fixture):
    events = _get_events(
        skyfield_fixture,
        minute_begin=MINUTE_AFTER_CULMINATES,
        minute_end=MINUTE_BEFORE_LEAVES,
    )
    assert events.tolist() == [SET]


def test_events_found_on_window_that_is_between_enter_and_culminates(skyfield_fixture):
    events = _get_events(
        skyfield_fixture,
        minute_begin=MINUTES_AFTER_ENTERS,
        minute_end=MINUTE_BEFORE_CULMINATES,
    )
    assert events.tolist() == []


def test_events_found_on_window_that_encompasses_only_enters(skyfield_fixture):
    events = _get_events(
        skyfield_fixture,
        minute_begin=MINUTE_BEFORE_ENTERS,
        minute_end=MINUTE_BEFORE_CULMINATES,
    )
    assert events.tolist() == [RISE]


def test_events_found_on_window_that_encompasses_only_culminates(skyfield_fixture):
    events = _get_events(
        skyfield_fixture,
        minute_begin=MINUTE_BEFORE_CULMINATES,
        minute_end=MINUTE_AFTER_CULMINATES,
    )
    assert events.tolist() == [CULMINATE]


def test_events_found_on_window_that_encompasses_culminates_and_leaves(
    skyfield_fixture,
):
    events = _get_events(
        skyfield_fixture,
        minute_begin=MINUTE_BEFORE_CULMINATES,
        minute_end=MINUTE_BEFORE_LEAVES,
    )
    assert events.tolist() == [CULMINATE, SET]


def test_events_found_on_window_that_encompasses_culminates_and_enters(
    skyfield_fixture,
):
    events = _get_events(
        skyfield_fixture,
        minute_begin=MINUTE_BEFORE_ENTERS,
        minute_end=MINUTE_AFTER_CULMINATES,
    )
    assert events.tolist() == [RISE, CULMINATE]


def test_events_found_on_window_that_encompasses_full_satellite_pass(skyfield_fixture):
    events = _get_events(
        skyfield_fixture,
        minute_begin=MINUTE_BEFORE_ENTERS,
        minute_end=MINUTE_BEFORE_LEAVES,
    )
    assert events.tolist() == [RISE, CULMINATE, SET]


@pytest.fixture(scope="module")
def skyfield_fixture():
    """Calculates heavy objects (Timescale, Sat) once per module."""
    ts = load.timescale()
    sat = Satellite(
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
        frequency=[FrequencyRange(frequency=137.513, bandwidth=None, status="active")],
    )
    return {"ts": ts, "sat": sat}


def _get_events(fixture, minute_begin: int, minute_end: int):
    ts = fixture["ts"]
    sat = fixture["sat"]

    t0 = ts.from_datetime(datetime(2023, 3, 30, 12, minute_begin, tzinfo=timezone.utc))
    t1 = ts.from_datetime(datetime(2023, 3, 30, 12, minute_end, tzinfo=timezone.utc))

    coords = wgs84.latlon(40.8178049, -121.4695413)
    sky_sat = sat.to_skyfield()

    _, events = sky_sat.find_events(coords, t0, t1, altitude_degrees=0)
    return events
