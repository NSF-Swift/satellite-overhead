import pytest
import numpy as np
from datetime import datetime, timedelta, timezone
from sopp.utils.time import generate_time_grid


@pytest.fixture
def t0():
    return datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_generate_time_grid_standard(t0):
    """
    Scenario: Standard 1-second resolution over a 10-second window.
    Expected: 11 points (inclusive of start and end: 0, 1, ..., 10).
    """
    t_end = t0 + timedelta(seconds=10)

    grid = generate_time_grid(t0, t_end, resolution_seconds=1.0)

    assert isinstance(grid, np.ndarray)
    assert len(grid) == 11
    assert grid[0] == t0
    assert grid[-1] == t_end
    # Verify spacing
    assert grid[1] == t0 + timedelta(seconds=1)


def test_generate_time_grid_sub_second(t0):
    """
    Scenario: 0.5 second resolution over a 2-second window.
    Expected: 5 points (0.0, 0.5, 1.0, 1.5, 2.0).
    """
    t_end = t0 + timedelta(seconds=2)

    grid = generate_time_grid(t0, t_end, resolution_seconds=0.5)

    assert len(grid) == 5
    assert grid[0] == t0
    assert grid[1] == t0 + timedelta(milliseconds=500)
    assert grid[-1] == t_end


def test_generate_time_grid_uneven_duration(t0):
    """
    Scenario: Duration (2.5s) is not a multiple of resolution (1.0s).
    Logic Check: steps = int(2.5 / 1.0) + 1 = 3 steps (0, 1, 2).
    Expected: Should capture points up to the floor, but not exceed end time.
    """
    t_end = t0 + timedelta(seconds=2.5)

    grid = generate_time_grid(t0, t_end, resolution_seconds=1.0)

    assert len(grid) == 3
    assert grid[-1] == t0 + timedelta(seconds=2)
    # Ensure we didn't generate a point past the end time
    assert grid[-1] < t_end


def test_generate_time_grid_large_step(t0):
    """
    Scenario: Step size is larger than duration.
    Expected: Should return just the start time (1 step).
    """
    t_end = t0 + timedelta(seconds=5)
    resolution = 10.0  # Step is 10s

    grid = generate_time_grid(t0, t_end, resolution_seconds=resolution)

    # int(5 / 10) + 1 = 1 step
    assert len(grid) == 1
    assert grid[0] == t0


def test_zero_duration_returns_single_point(t0):
    """
    Scenario: Start time equals End time.
    Expected: Array containing exactly one point (start time).
    """
    grid = generate_time_grid(t0, t0, resolution_seconds=1.0)

    assert len(grid) == 1
    assert grid[0] == t0


def test_negative_duration_returns_single_point(t0):
    """
    Scenario: End time is before Start time.
    Expected: Array containing exactly one point (start time).
    """
    t_past = t0 - timedelta(seconds=10)
    grid = generate_time_grid(t0, t_past, resolution_seconds=1.0)

    assert len(grid) == 1
    assert grid[0] == t0


def test_returns_numpy_array_of_objects(t0):
    """
    Verifies that the returned structure is compatible with Skyfield
    (NumPy array of python datetime objects, not np.datetime64).
    """
    t_end = t0 + timedelta(seconds=1)
    grid = generate_time_grid(t0, t_end)

    assert isinstance(grid, np.ndarray)
    assert grid.dtype == object
    assert isinstance(grid[0], datetime)
