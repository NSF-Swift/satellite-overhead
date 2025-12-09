import numpy as np
import pytest

from sopp.models import Configuration, RuntimeSettings


def test_validate_empty_satellites_list(antenna_trajectory, reservation):
    with pytest.raises(ValueError) as _:
        _ = Configuration(
            satellites=[],
            antenna_trajectory=antenna_trajectory,
            reservation=reservation,
        )


def test_validate_runtime_settings_time_resolution(
    satellite, reservation, antenna_trajectory
):
    runtime_settings = RuntimeSettings(time_resolution_seconds=-1)

    with pytest.raises(ValueError) as _:
        _ = Configuration(
            satellites=[satellite],
            antenna_trajectory=antenna_trajectory,
            reservation=reservation,
            runtime_settings=runtime_settings,
        )


def test_validate_runtime_settings_concurrency(
    satellite, antenna_trajectory, reservation
):
    runtime_settings = RuntimeSettings(concurrency_level=0)
    with pytest.raises(ValueError) as _:
        _ = Configuration(
            satellites=[satellite],
            antenna_trajectory=antenna_trajectory,
            reservation=reservation,
            runtime_settings=runtime_settings,
        )


def test_validate_minimum_altitude(satellite, antenna_trajectory, reservation):
    runtime_settings = RuntimeSettings(min_altitude=-1)
    with pytest.raises(ValueError) as _:
        _ = Configuration(
            satellites=[satellite],
            antenna_trajectory=antenna_trajectory,
            reservation=reservation,
            runtime_settings=runtime_settings,
        )


def test_validate_reservation_time_window(satellite, antenna_trajectory, reservation):
    reservation.time.begin = reservation.time.end
    with pytest.raises(ValueError) as _:
        _ = Configuration(
            satellites=[satellite],
            antenna_trajectory=antenna_trajectory,
            reservation=reservation,
            runtime_settings=RuntimeSettings(),
        )


def test_validate_reservation_beamwidth(satellite, antenna_trajectory, reservation):
    reservation.facility.beamwidth = 0
    with pytest.raises(ValueError) as _:
        _ = Configuration(
            satellites=[satellite],
            antenna_trajectory=antenna_trajectory,
            reservation=reservation,
            runtime_settings=RuntimeSettings(),
        )


def test_validate_empty_antenna_trajectory(satellite, reservation):
    antenna_trajectory = []
    with pytest.raises(ValueError) as _:
        _ = Configuration(
            satellites=[satellite],
            antenna_trajectory=antenna_trajectory,
            reservation=reservation,
            runtime_settings=RuntimeSettings(),
        )


def test_validate_antenna_trajectory_increasing_times(
    satellite, reservation, antenna_trajectory
):
    antenna_trajectory.times = np.append(
        antenna_trajectory.times, antenna_trajectory.times[0]
    )
    with pytest.raises(ValueError) as _:
        _ = Configuration(
            satellites=[satellite],
            antenna_trajectory=antenna_trajectory,
            reservation=reservation,
            runtime_settings=RuntimeSettings(),
        )
