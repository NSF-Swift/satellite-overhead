from datetime import timedelta

import numpy as np
from tests.conftest import ARBITRARY_ALTITUDE, ARBITRARY_AZIMUTH

from sopp.analysis.interference import analyze_interference
from sopp.analysis.strategies import GeometricStrategy
from sopp.analysis.visibility import find_satellites_above_horizon
from sopp.models.core import FrequencyRange
from sopp.models.ground.trajectory import AntennaTrajectory
from sopp.models.satellite.trajectory import SatelliteTrajectory
from sopp.utils.time import generate_time_grid


def test_satellite_inside_beam_is_detected(
    arbitrary_datetime, satellite, make_reservation, ephemeris_stub
):
    """
    Scenario: Satellite is positioned exactly where the antenna is pointing.
    Expected: An InterferenceResult should be returned.
    """
    reservation = make_reservation(start_time=arbitrary_datetime, duration_seconds=1)

    times = generate_time_grid(arbitrary_datetime, arbitrary_datetime, 1)
    ant_traj = AntennaTrajectory(
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
    )

    trajectories = find_satellites_above_horizon(
        reservation, [satellite], ephemeris_stub()
    )
    results = analyze_interference(
        trajectories=trajectories,
        antenna_trajectory=ant_traj,
        strategy=GeometricStrategy(),
        facility=reservation.facility,
        frequency=reservation.frequency,
    )

    assert len(results) == 1
    assert results[0].trajectory.satellite.name == satellite.name


def test_satellite_outside_beam_is_ignored(
    arbitrary_datetime,
    satellite,
    make_reservation,
    facility,
    ephemeris_stub,
):
    """
    Scenario: Satellite is just outside the beamwidth.
    Expected: No results returned.
    """
    reservation = make_reservation(start_time=arbitrary_datetime, duration_seconds=1)

    offset = facility.beamwidth + 1.0
    times = generate_time_grid(arbitrary_datetime, arbitrary_datetime, 1)

    ant_traj = AntennaTrajectory(
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH + offset),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
    )

    trajectories = find_satellites_above_horizon(
        reservation, [satellite], ephemeris_stub()
    )
    results = analyze_interference(
        trajectories=trajectories,
        antenna_trajectory=ant_traj,
        strategy=GeometricStrategy(),
        facility=reservation.facility,
        frequency=reservation.frequency,
    )

    assert len(results) == 0


def test_satellite_enters_and_exits_beam(
    arbitrary_datetime, satellite, make_reservation, facility, ephemeris_stub
):
    """
    Scenario:
        T=0: Satellite Outside
        T=1: Satellite Inside
        T=2: Satellite Outside
    Expected: One result returned containing only the data at T=1.
    """
    reservation = make_reservation(start_time=arbitrary_datetime, duration_seconds=3)

    t_end = arbitrary_datetime + timedelta(seconds=2)
    times = generate_time_grid(arbitrary_datetime, t_end, resolution_seconds=1)

    offset = facility.beamwidth + 1.0
    azimuths = np.array(
        [
            ARBITRARY_AZIMUTH + offset,  # T=0
            ARBITRARY_AZIMUTH,  # T=1 (Hit)
            ARBITRARY_AZIMUTH + offset,  # T=2
        ]
    )

    ant_traj = AntennaTrajectory(
        times=times, azimuth=azimuths, altitude=np.full(len(times), ARBITRARY_ALTITUDE)
    )

    trajectories = find_satellites_above_horizon(
        reservation, [satellite], ephemeris_stub()
    )
    results = analyze_interference(
        trajectories=trajectories,
        antenna_trajectory=ant_traj,
        strategy=GeometricStrategy(),
        facility=reservation.facility,
        frequency=reservation.frequency,
    )

    assert len(results) == 1

    interfering_traj = results[0].trajectory
    assert len(interfering_traj) == 1
    assert interfering_traj.times[0] == arbitrary_datetime + timedelta(seconds=1)


def test_satellite_always_above_horizon_is_returned(
    satellite, reservation, ephemeris_stub
):
    """
    Scenario: Satellite is visible for the entire window.
    Expected: One trajectory returned covering the full duration.
    """
    trajectories = find_satellites_above_horizon(
        reservation, [satellite], ephemeris_stub()
    )

    assert len(trajectories) == 1
    assert trajectories[0].satellite.name == satellite.name
    # Verify we got data points
    assert len(trajectories[0]) > 0


# --- analyze_interference tests ---


def test_analyze_interference_applies_strategy(arbitrary_datetime, satellite, facility):
    """
    Verify that analyze_interference applies the strategy to each trajectory
    and collects results.
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

    results = analyze_interference(
        trajectories=[sat_traj],
        antenna_trajectory=ant_traj,
        strategy=GeometricStrategy(),
        facility=facility,
        frequency=frequency,
    )

    assert len(results) == 1
    assert results[0].trajectory.satellite.name == satellite.name
