from datetime import timedelta

import numpy as np

from sopp.models.antenna_trajectory import AntennaTrajectory
from sopp.utils.time import generate_time_grid
from tests.conftest import ARBITRARY_ALTITUDE, ARBITRARY_AZIMUTH


def test_reservation_begins_part_way_through_antenna_position_time(
    arbitrary_datetime, satellite, make_reservation, make_event_finder
):
    """
    Scenario: Antenna points at satellite starting at T-1s. Reservation starts at T=0.
    Expected: The EventFinder should detect interference only starting at T=0.
    """
    # 1. Setup Reservation (Start = T0)
    reservation = make_reservation(start_time=arbitrary_datetime, duration_seconds=2)

    # 2. Setup Mock Antenna Path (Starts at T-1)
    # We create a grid from T-1 to T+2
    t_start_ant = arbitrary_datetime - timedelta(seconds=1)
    t_end_ant = arbitrary_datetime + timedelta(seconds=2)

    times = generate_time_grid(t_start_ant, t_end_ant, resolution_seconds=1)
    n = len(times)

    # Antenna is STATIC, pointing exactly at ARBITRARY_AZIMUTH/ALTITUDE
    trajectory = AntennaTrajectory(
        times=times,
        azimuth=np.full(n, ARBITRARY_AZIMUTH),
        altitude=np.full(n, ARBITRARY_ALTITUDE),
    )

    # 3. Setup EventFinder
    event_finder = make_event_finder(
        reservation=reservation, satellites=[satellite], antenna_trajectory=trajectory
    )

    # 4. Execute
    trajectories = event_finder.get_satellites_crossing_main_beam()

    # 5. Verify
    assert len(trajectories) == 1
    traj = trajectories[0]

    # Verify the satellite is correct
    assert traj.satellite.name == satellite.name

    # Verify the start time is clipped to the Reservation Start (T0)
    assert traj.times[0] == reservation.time.begin
    assert len(traj) > 0


def test_antenna_positions_that_end_before_reservation_starts_are_not_included(
    arbitrary_datetime,
    satellite,
    make_reservation,
    make_event_finder,
    facility,
):
    """
    Scenario: Antenna points at satellite at T-1, but moves away at T=0.
    Expected: No interference found because the overlap happened before the reservation.
    """
    # 1. Setup Reservation (Start = T0)
    reservation = make_reservation(start_time=arbitrary_datetime, duration_seconds=2)

    # 2. Setup Mock Antenna Path
    # Grid: T-1 to T+1
    t_minus_1 = arbitrary_datetime - timedelta(seconds=1)
    t_plus_1 = arbitrary_datetime + timedelta(seconds=1)
    times = generate_time_grid(t_minus_1, t_plus_1, resolution_seconds=1)

    # Define Altitudes
    # T-1: Hits Satellite
    # T=0+: Misses Satellite (Shifted by beamwidth + epsilon)
    alt_hit = ARBITRARY_ALTITUDE
    alt_miss = ARBITRARY_ALTITUDE + facility.beamwidth

    # Create array: [Hit, Miss, Miss]
    altitudes = np.array([alt_hit, alt_miss, alt_miss])
    azimuths = np.full(len(times), ARBITRARY_AZIMUTH)

    trajectory = AntennaTrajectory(times=times, azimuth=azimuths, altitude=altitudes)

    # 3. Setup EventFinder
    event_finder = make_event_finder(
        reservation=reservation, satellites=[satellite], antenna_trajectory=trajectory
    )

    # 4. Execute
    trajectories = event_finder.get_satellites_crossing_main_beam()

    # 5. Verify
    assert len(trajectories) == 0
