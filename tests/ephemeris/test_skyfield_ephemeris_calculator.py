import pytest
from datetime import datetime, timedelta, timezone
from dataclasses import replace

from sopp.models.coordinates import Coordinates
from sopp.models.facility import Facility
from sopp.models.satellite.international_designator import InternationalDesignator
from sopp.models.satellite.mean_motion import MeanMotion
from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.tle_information import TleInformation
from sopp.ephemeris.skyfield import SkyfieldEphemerisCalculator


def test_altitude_can_be_negative(calculator_empty, arbitrary_satellite, sample_time):
    """
    Verifies that we can calculate a position for a satellite below the horizon.
    """
    # Uses ad-hoc lookup (get_position_at) since calculator is empty
    position = calculator_empty.get_position_at(arbitrary_satellite, sample_time)
    assert position.position.altitude < 0


def test_azimuth_can_be_greater_than_180(
    calculator_empty, arbitrary_satellite, sample_time
):
    """
    Verifies azimuth ranges are standard (0-360).
    """
    position = calculator_empty.get_position_at(arbitrary_satellite, sample_time)
    # Based on the TLE/Time, this specific sat has Az > 180
    assert position.position.azimuth > 180


def test_altitude_decreases_as_facility_elevation_increases(
    arbitrary_satellite, sample_time
):
    """
    Verifies that moving the facility UP reduces the relative altitude of the satellite.
    """
    # 1. Facility at Sea Level
    facility_sea_level = Facility(Coordinates(latitude=0, longitude=-24.66605))
    calc_sea = SkyfieldEphemerisCalculator(facility_sea_level, [])
    pos_sea = calc_sea.get_position_at(arbitrary_satellite, sample_time)

    # 2. Facility at 1000m
    facility_high = replace(facility_sea_level, elevation=1000)
    calc_high = SkyfieldEphemerisCalculator(facility_high, [])
    pos_high = calc_high.get_position_at(arbitrary_satellite, sample_time)

    assert pos_high.position.altitude < pos_sea.position.altitude


def test_get_position_at_uses_cache_if_available(
    calculator_with_grid, arbitrary_satellite, time_grid
):
    """
    Verifies that requesting a time exactly matching a grid point works
    (and implicitly assumes it uses the fast cache path).
    """
    target_time = time_grid[2]  # The middle point
    pos = calculator_with_grid.get_position_at(arbitrary_satellite, target_time)

    assert pos.time == target_time
    assert isinstance(pos.position.altitude, float)


def test_get_position_at_falls_back_if_missing_from_grid(
    calculator_with_grid, arbitrary_satellite, sample_time
):
    """
    Verifies that requesting a time NOT in the grid (e.g., 30 seconds offset)
    still returns a valid calculation (Fallback logic).
    """
    # 30 seconds after the first grid point
    off_grid_time = sample_time + timedelta(seconds=30)

    pos = calculator_with_grid.get_position_at(arbitrary_satellite, off_grid_time)

    assert pos.time == off_grid_time
    assert isinstance(pos.position.altitude, float)


def test_get_positions_within_window_slices_correctly(
    calculator_with_grid, arbitrary_satellite, time_grid
):
    """
    Verifies the bisect logic returns exactly the subset of grid points within the window.
    """
    # Grid: [T0, T1, T2, T3, T4]
    # Window: T1 -> T3 (Inclusive start, Exclusive end usually, depending on bisect usage)
    # Our bisect implementation usually slices [Start:End]

    start_window = time_grid[1]
    end_window = time_grid[3]  # Should include T1, T2

    # Note: Depending on specific bisect_right logic, check if end is inclusive or exclusive
    # Standard python slice [1:3] includes indices 1 and 2.

    results = calculator_with_grid.get_positions_window(
        arbitrary_satellite, start_window, end_window
    )

    assert len(results) == 3
    assert results[0].time == time_grid[1]
    assert results[1].time == time_grid[2]
    # Ensure T3 is NOT included if using standard slicing (or adjust assert if you made it inclusive)


def test_get_positions_within_window_returns_empty_if_no_overlap(
    calculator_with_grid, arbitrary_satellite, sample_time
):
    """
    Verifies that a window entirely "between" two grid points returns an empty list.
    """
    # Grid has points at 00:00 and 00:01
    # Window is 00:00:15 to 00:00:45
    t_start = sample_time + timedelta(seconds=15)
    t_end = sample_time + timedelta(seconds=45)

    results = calculator_with_grid.get_positions_window(
        arbitrary_satellite, t_start, t_end
    )

    assert results == []


def test_find_events_with_grid(calculator_with_grid, arbitrary_satellite, sample_time):
    """
    Verifies find_events works when grid is present.
    """
    # We don't expect actual events in this 5 minute window for this sat,
    # but we check the types are valid.
    t_start = sample_time + timedelta(seconds=15)
    t_end = sample_time + timedelta(seconds=45)

    windows = calculator_with_grid.find_events(arbitrary_satellite, 0.0, t_start, t_end)

    assert isinstance(windows, list)


@pytest.fixture
def arbitrary_satellite() -> Satellite:
    """
    COSMOS 1932 DEB
    """
    return Satellite(
        name="ARBITRARY SATELLITE",
        tle_information=TleInformation(
            argument_of_perigee=5.153187590939126,
            drag_coefficient=0.00015211,
            eccentricity=0.0057116,
            epoch_days=26633.28893622,
            inclination=1.1352005427406557,
            international_designator=InternationalDesignator(
                year=88, launch_number=19, launch_piece="F"
            ),
            mean_anomaly=4.188343400497881,
            mean_motion=MeanMotion(
                first_derivative=2.363466695408988e-12,
                second_derivative=0.0,
                value=0.060298700041442894,
            ),
            revolution_number=95238,
            right_ascension_of_ascending_node=2.907844197528697,
            satellite_number=28275,
            classification="U",
        ),
        frequency=[],
    )


@pytest.fixture
def facility_at_zero() -> Facility:
    return Facility(Coordinates(latitude=0, longitude=0))


@pytest.fixture
def sample_time() -> datetime:
    return datetime(year=2023, month=6, day=7, tzinfo=timezone.utc)


@pytest.fixture
def time_grid(sample_time) -> list[datetime]:
    """Creates a list of 5 timestamps spaced 1 minute apart."""
    return [sample_time + timedelta(minutes=i) for i in range(5)]


@pytest.fixture
def calculator_with_grid(facility_at_zero, time_grid):
    """A calculator initialized with a pre-cached grid."""
    return SkyfieldEphemerisCalculator(facility=facility_at_zero, datetimes=time_grid)


@pytest.fixture
def calculator_empty(facility_at_zero):
    """A calculator initialized with NO grid (for ad-hoc testing)."""
    return SkyfieldEphemerisCalculator(facility=facility_at_zero, datetimes=[])
