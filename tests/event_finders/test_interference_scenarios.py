from datetime import timedelta

import numpy as np

from sopp.models.antenna_trajectory import AntennaTrajectory
from sopp.utils.time import generate_time_grid
from tests.conftest import ARBITRARY_ALTITUDE, ARBITRARY_AZIMUTH


def test_satellite_inside_beam_is_detected(
    arbitrary_datetime, satellite, make_reservation, make_event_finder, facility
):
    """
    Scenario: Satellite is positioned exactly where the antenna is pointing.
    Expected: A trajectory should be returned containing the interference data.
    """
    reservation = make_reservation(start_time=arbitrary_datetime, duration_seconds=1)

    # Setup: Antenna matches Satellite exactly
    times = generate_time_grid(arbitrary_datetime, arbitrary_datetime, 1)
    traj = AntennaTrajectory(
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
    )

    finder = make_event_finder(
        reservation=reservation, satellites=[satellite], antenna_trajectory=traj
    )

    windows = finder.get_satellites_crossing_main_beam()
    assert len(windows) == 1
    assert windows[0].satellite.name == satellite.name


def test_satellite_outside_beam_is_ignored(
    arbitrary_datetime, satellite, make_reservation, make_event_finder, facility
):
    """
    Scenario: Satellite is just outside the beamwidth.
    Expected: No trajectories returned.
    """
    reservation = make_reservation(start_time=arbitrary_datetime, duration_seconds=1)

    # Setup: Antenna pointed far away (beamwidth + 1 degree)
    offset = facility.beamwidth + 1.0
    times = generate_time_grid(arbitrary_datetime, arbitrary_datetime, 1)

    traj = AntennaTrajectory(
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH + offset),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
    )

    finder = make_event_finder(
        reservation=reservation, satellites=[satellite], antenna_trajectory=traj
    )

    windows = finder.get_satellites_crossing_main_beam()
    assert len(windows) == 0


def test_satellite_enters_and_exits_beam(
    arbitrary_datetime, satellite, make_reservation, make_event_finder, facility
):
    """
    Scenario:
        T=0: Satellite Outside
        T=1: Satellite Inside
        T=2: Satellite Outside
    Expected: One trajectory returned containing only the data at T=1.
    """
    reservation = make_reservation(start_time=arbitrary_datetime, duration_seconds=3)

    # Grid: T0, T1, T2
    t_end = arbitrary_datetime + timedelta(seconds=2)
    times = generate_time_grid(arbitrary_datetime, t_end, resolution_seconds=1)

    # Antenna moves: [Far, Hit, Far]
    # We simulate this by moving the antenna, assuming the satellite stays at ARBITRARY_AZIMUTH.
    offset = facility.beamwidth + 1.0
    azimuths = np.array(
        [
            ARBITRARY_AZIMUTH + offset,  # T=0
            ARBITRARY_AZIMUTH,  # T=1 (Hit)
            ARBITRARY_AZIMUTH + offset,  # T=2
        ]
    )

    traj = AntennaTrajectory(
        times=times, azimuth=azimuths, altitude=np.full(len(times), ARBITRARY_ALTITUDE)
    )

    finder = make_event_finder(
        reservation=reservation, satellites=[satellite], antenna_trajectory=traj
    )

    windows = finder.get_satellites_crossing_main_beam()

    assert len(windows) == 1

    # The resulting trajectory should only contain the point where interference occurred
    interfering_traj = windows[0]
    assert len(interfering_traj) == 1
    assert interfering_traj.times[0] == arbitrary_datetime + timedelta(seconds=1)
