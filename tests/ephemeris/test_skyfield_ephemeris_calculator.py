import pytest
import numpy as np
from datetime import datetime, timedelta, timezone
from dataclasses import replace

from sopp.ephemeris.skyfield import SkyfieldEphemerisCalculator
from sopp.models.satellite.international_designator import InternationalDesignator
from sopp.models.satellite.mean_motion import MeanMotion
from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.tle_information import TleInformation
from sopp.models import Facility, TimeWindow, Coordinates


def test_init_accepts_numpy_array(facility_at_zero, time_grid):
    """
    Verifies that the calculator correctly handles NumPy arrays in __init__,
    converting/storing them correctly.
    """
    # Case 1: Standard List (Already covered by fixture, but explicit here)
    calc_list = SkyfieldEphemerisCalculator(facility_at_zero, time_grid)
    assert len(calc_list._datetimes) == 5

    # Case 2: NumPy Array
    np_grid = np.array(time_grid)
    calc_np = SkyfieldEphemerisCalculator(facility_at_zero, np_grid)
    assert len(calc_np._datetimes) == 5
    # Ensure it didn't crash and physics are cached
    assert "M" in calc_np._grid_timescale.__dict__


def test_altitude_can_be_negative(
    calculator_with_grid, arbitrary_satellite, sample_time
):
    """
    Verifies that we can calculate a position for a satellite below the horizon.
    """
    position = calculator_with_grid.calculate_position(arbitrary_satellite, sample_time)
    assert position.altitude < 0


def test_azimuth_can_be_greater_than_180(
    calculator_with_grid, arbitrary_satellite, sample_time
):
    """
    Verifies azimuth ranges are standard (0-360).
    """
    position = calculator_with_grid.calculate_position(arbitrary_satellite, sample_time)
    # Based on the TLE/Time, this specific sat has Az > 180
    assert position.azimuth > 180


def test_altitude_decreases_as_facility_elevation_increases(
    arbitrary_satellite, sample_time
):
    """
    Verifies that moving the facility UP reduces the relative altitude of the satellite.
    """
    # 1. Facility at Sea Level
    facility_sea_level = Facility(Coordinates(latitude=0, longitude=-24.66605))
    calc_sea = SkyfieldEphemerisCalculator(facility_sea_level, [sample_time])
    pos_sea = calc_sea.calculate_position(arbitrary_satellite, sample_time)

    # 2. Facility at 1000m
    facility_high = replace(facility_sea_level, elevation=1000)
    calc_high = SkyfieldEphemerisCalculator(facility_high, [sample_time])
    pos_high = calc_high.calculate_position(arbitrary_satellite, sample_time)

    assert pos_high.altitude < pos_sea.altitude


def test_calculate_position_uses_cache_if_available(
    calculator_with_grid, arbitrary_satellite, time_grid
):
    """
    Verifies that requesting a time exactly matching a grid point works
    (and implicitly assumes it uses the fast cache path).
    """
    target_time = time_grid[2]  # The middle point
    pos = calculator_with_grid.calculate_position(arbitrary_satellite, target_time)

    assert isinstance(pos.altitude, float)


def test_calculate_position_falls_back_if_missing_from_grid(
    calculator_with_grid, arbitrary_satellite, sample_time
):
    """
    Verifies that requesting a time NOT in the grid (e.g., 30 seconds offset)
    still returns a valid calculation (Fallback logic).
    """
    # 30 seconds after the first grid point
    off_grid_time = sample_time + timedelta(seconds=30)

    pos = calculator_with_grid.calculate_position(arbitrary_satellite, off_grid_time)

    assert isinstance(pos.altitude, float)


def test_calculate_trajectory_slices_correctly(
    calculator_with_grid, arbitrary_satellite, time_grid
):
    """
    Verifies the bisect logic returns exactly the subset of grid points within the window.
    """
    # Grid: [T0, T1, T2, T3, T4]

    start_window = time_grid[1]
    end_window = time_grid[3]  # Should include T1, T2

    results = calculator_with_grid.calculate_trajectory(
        arbitrary_satellite, start_window, end_window
    )

    # Should contain T1 and T2 (Inclusive-Exclusive behavior depending on implementation,
    # but based on previous tests we established len=3 means Inclusive T3?)
    # Re-verifying logic:
    # If calculate_trajectory uses generate_time_grid or slicing:
    # If the grid exists, we slice. [1:4] -> 1, 2, 3.
    assert len(results) == 3
    assert results.times[0] == time_grid[1]
    assert results.times[1] == time_grid[2]
    assert results.times[2] == time_grid[3]


def test_calculate_trajectory_returns_empty_if_no_overlap(
    calculator_with_grid, arbitrary_satellite, sample_time
):
    """
    Verifies that a window entirely "between" two grid points returns an empty list.
    """
    # Grid has points at 00:00 and 00:01
    # Window is 00:00:15 to 00:00:45
    t_start = sample_time + timedelta(seconds=15)
    t_end = sample_time + timedelta(seconds=45)

    results = calculator_with_grid.calculate_trajectory(
        arbitrary_satellite, t_start, t_end
    )

    assert len(results.times) == 0


def test_calculate_trajectories_with_out_of_bounds_window(
    calculator_with_grid, arbitrary_satellite, sample_time
):
    """
    Verifies that the batch method handles windows completely outside the grid range
    gracefully (returning empty trajectories) rather than crashing.
    """
    # Window way in the future
    future = sample_time + timedelta(days=100)
    win = TimeWindow(future, future + timedelta(seconds=1))

    results = calculator_with_grid.calculate_trajectories(arbitrary_satellite, [win])
    assert len(results) == 0


def test_calculate_visibility_windows_with_grid(
    calculator_with_grid, arbitrary_satellite, sample_time
):
    """
    Verifies calculate_visibility_windows works when grid is present.
    """
    t_start = sample_time + timedelta(seconds=15)
    t_end = sample_time + timedelta(seconds=45)

    windows = calculator_with_grid.calculate_visibility_windows(
        arbitrary_satellite, 0.0, t_start, t_end
    )

    assert isinstance(windows, list)


def test_calculate_trajectories_handles_multiple_windows(
    calculator_with_grid, arbitrary_satellite, time_grid
):
    """
    Verifies that the batch engine correctly processes multiple disjoint windows
    and returns a list of separate trajectories.
    """
    # Grid: [T0, T1, T2, T3, T4]

    # Window A: T0 -> T1
    win_a = TimeWindow(time_grid[0], time_grid[1])

    # Window B: T3 -> T4 (skipping T2 to ensure 'gaps' work)
    win_b = TimeWindow(time_grid[3], time_grid[4])

    results = calculator_with_grid.calculate_trajectories(
        arbitrary_satellite, [win_a, win_b]
    )

    assert len(results) == 2

    # Check Window A
    traj_a = results[0]
    assert len(traj_a) == 2
    assert traj_a.times[0] == time_grid[0]
    assert traj_a.times[1] == time_grid[1]

    # Check Window B
    traj_b = results[1]
    assert len(traj_b) == 2
    assert traj_b.times[0] == time_grid[3]
    assert traj_b.times[1] == time_grid[4]

    # Check that the GAP (T2) did not leak into results
    # Checking times directly using numpy isin or set intersection
    all_times = np.concatenate([traj_a.times, traj_b.times])
    assert time_grid[2] not in all_times


def test_trajectory_data_alignment(
    calculator_with_grid, arbitrary_satellite, time_grid
):
    """
    Verifies that the vector arrays in the trajectory are all the same length.
    """
    win = TimeWindow(time_grid[0], time_grid[-1])
    traj = calculator_with_grid.calculate_trajectory(
        arbitrary_satellite, win.begin, win.end
    )

    n = len(traj.times)
    assert len(traj.azimuth) == n
    assert len(traj.altitude) == n
    assert len(traj.distance_km) == n
    assert n > 0


def test_scalar_and_vector_methods_are_consistent(
    calculator_with_grid, arbitrary_satellite, time_grid
):
    """
    Verifies that getting a single point via calculate_position returns
    the same values as extracting that point from a trajectory.
    """
    target_time = time_grid[2]

    # 1. Scalar Method
    pos_scalar = calculator_with_grid.calculate_position(
        arbitrary_satellite, target_time
    )

    # 2. Vector Method
    traj = calculator_with_grid.calculate_trajectory(
        arbitrary_satellite, target_time, target_time
    )

    assert len(traj) >= 1

    # Compare
    assert pos_scalar.azimuth == pytest.approx(traj.azimuth[0])
    assert pos_scalar.altitude == pytest.approx(traj.altitude[0])


@pytest.fixture
def arbitrary_satellite() -> Satellite:
    """COSMOS 1932 DEB"""
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
