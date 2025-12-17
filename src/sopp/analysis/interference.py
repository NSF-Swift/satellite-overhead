import numpy as np

from sopp.analysis.geometry import calculate_angular_separation_sq
from sopp.ephemeris.base import EphemerisCalculator
from sopp.models.ground.trajectory import AntennaTrajectory
from sopp.models.reservation import Reservation
from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.trajectory import SatelliteTrajectory


def find_satellites_above_horizon(
    reservation: Reservation,
    satellites: list[Satellite],
    ephemeris_calculator: EphemerisCalculator,
    min_altitude: float = 0.0,
) -> list[SatelliteTrajectory]:
    trajectories = []

    for satellite in satellites:
        time_windows = ephemeris_calculator.calculate_visibility_windows(
            satellite,
            min_altitude,
            reservation.time.begin,
            reservation.time.end,
        )

        if not time_windows:
            continue

        satellite_trajectories = ephemeris_calculator.calculate_trajectories(
            satellite, time_windows
        )

        trajectories.extend(satellite_trajectories)

    return trajectories


def find_satellites_crossing_main_beam(
    reservation: Reservation,
    satellites: list[Satellite],
    ephemeris_calculator: EphemerisCalculator,
    antenna_trajectory: AntennaTrajectory,
    min_altitude: float = 0.0,
) -> list[SatelliteTrajectory]:
    beam_radius_sq = reservation.facility.beam_radius**2

    interfering_trajectories = []

    for satellite in satellites:
        # Get visibility windows
        windows = ephemeris_calculator.calculate_visibility_windows(
            satellite,
            min_altitude,
            reservation.time.begin,
            reservation.time.end,
        )
        if not windows:
            continue

        # Get satellite path
        sat_trajectories = ephemeris_calculator.calculate_trajectories(
            satellite, windows
        )

        for sat_traj in sat_trajectories:
            # Interpolate Antenna position to match Satellite times
            ant_az, ant_alt = antenna_trajectory.get_state_at(sat_traj.times)

            sep_sq = calculate_angular_separation_sq(
                az1=sat_traj.azimuth,
                alt1=sat_traj.altitude,
                az2=ant_az,
                alt2=ant_alt,
            )

            # Check if separation < radius
            mask = sep_sq <= beam_radius_sq

            if np.any(mask):
                # Extract the interfering path segment
                interfering_trajectories.append(
                    SatelliteTrajectory(
                        satellite=satellite,
                        times=sat_traj.times[mask],
                        azimuth=sat_traj.azimuth[mask],
                        altitude=sat_traj.altitude[mask],
                        distance_km=sat_traj.distance_km[mask],
                    )
                )

    return interfering_trajectories
