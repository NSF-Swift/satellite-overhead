from datetime import timedelta

import numpy as np
from tests.conftest import ARBITRARY_ALTITUDE, ARBITRARY_AZIMUTH

from sopp.analysis.strategies import GeometricStrategy, InterferenceResult
from sopp.models.core import FrequencyRange
from sopp.models.ground.trajectory import AntennaTrajectory
from sopp.models.satellite.trajectory import SatelliteTrajectory
from sopp.utils.time import generate_time_grid


def test_geometric_strategy_detects_interference(
    arbitrary_datetime, satellite, facility
):
    """
    Scenario: Pre-computed trajectory at exact antenna position.
    Expected: GeometricStrategy returns an InterferenceResult.
    """
    times = generate_time_grid(arbitrary_datetime, arbitrary_datetime, 1)
    frequency = FrequencyRange(frequency=10, bandwidth=10)

    sat_traj = SatelliteTrajectory(
        satellite=satellite,
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
        distance_km=np.full(len(times), 500.0),
    )

    ant_traj = AntennaTrajectory(
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
    )

    strategy = GeometricStrategy()
    result = strategy.calculate(sat_traj, ant_traj, facility, frequency)

    assert result is not None
    assert isinstance(result, InterferenceResult)
    assert result.trajectory.satellite.name == satellite.name
    assert len(result.trajectory) == len(times)


def test_geometric_strategy_returns_none_outside_beam(
    arbitrary_datetime, satellite, facility
):
    """
    Scenario: Pre-computed trajectory far from antenna pointing.
    Expected: GeometricStrategy returns None.
    """
    times = generate_time_grid(arbitrary_datetime, arbitrary_datetime, 1)
    frequency = FrequencyRange(frequency=10, bandwidth=10)

    sat_traj = SatelliteTrajectory(
        satellite=satellite,
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
        distance_km=np.full(len(times), 500.0),
    )

    offset = facility.beamwidth + 1.0
    ant_traj = AntennaTrajectory(
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH + offset),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
    )

    strategy = GeometricStrategy()
    result = strategy.calculate(sat_traj, ant_traj, facility, frequency)

    assert result is None


def test_geometric_strategy_masks_partial_crossing(
    arbitrary_datetime, satellite, facility
):
    """
    Scenario:
        T=0: Satellite outside beam
        T=1: Satellite inside beam
        T=2: Satellite outside beam
    Expected: Result contains only T=1.
    """
    t_end = arbitrary_datetime + timedelta(seconds=2)
    times = generate_time_grid(arbitrary_datetime, t_end, resolution_seconds=1)
    frequency = FrequencyRange(frequency=10, bandwidth=10)

    sat_traj = SatelliteTrajectory(
        satellite=satellite,
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
        distance_km=np.full(len(times), 500.0),
    )

    # Antenna: far at T=0, aligned at T=1, far at T=2
    offset = facility.beamwidth + 1.0
    ant_traj = AntennaTrajectory(
        times=times,
        azimuth=np.array(
            [
                ARBITRARY_AZIMUTH + offset,
                ARBITRARY_AZIMUTH,
                ARBITRARY_AZIMUTH + offset,
            ]
        ),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
    )

    strategy = GeometricStrategy()
    result = strategy.calculate(sat_traj, ant_traj, facility, frequency)

    assert result is not None
    assert len(result.trajectory) == 1
    assert result.trajectory.times[0] == arbitrary_datetime + timedelta(seconds=1)


def test_interference_result_wraps_trajectory(arbitrary_datetime, satellite, facility):
    """
    Verify InterferenceResult.trajectory is a valid SatelliteTrajectory
    with all expected fields.
    """
    times = generate_time_grid(arbitrary_datetime, arbitrary_datetime, 1)
    frequency = FrequencyRange(frequency=10, bandwidth=10)

    sat_traj = SatelliteTrajectory(
        satellite=satellite,
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
        distance_km=np.full(len(times), 500.0),
    )

    ant_traj = AntennaTrajectory(
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
    )

    strategy = GeometricStrategy()
    result = strategy.calculate(sat_traj, ant_traj, facility, frequency)

    assert result is not None

    traj = result.trajectory
    assert isinstance(traj, SatelliteTrajectory)
    assert traj.satellite is satellite
    np.testing.assert_array_equal(traj.times, times)
    np.testing.assert_allclose(traj.azimuth, np.full(len(times), ARBITRARY_AZIMUTH))
    np.testing.assert_allclose(traj.altitude, np.full(len(times), ARBITRARY_ALTITUDE))
    np.testing.assert_allclose(traj.distance_km, np.full(len(times), 500.0))

    # GeometricStrategy produces no quantitative level
    assert result.interference_level is None
    assert result.level_units is None
