import numpy as np

from sopp.ephemeris.base import EphemerisCalculator
from sopp.event_finders.base import EventFinder
from sopp.models.antenna_trajectory import AntennaTrajectory
from sopp.models.reservation import Reservation
from sopp.models.runtime_settings import RuntimeSettings
from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite_trajectory import SatelliteTrajectory
from sopp.utils.geometry import calculate_angular_separation_sq


class EventFinderSkyfield(EventFinder):
    def __init__(
        self,
        antenna_trajectory: AntennaTrajectory,
        list_of_satellites: list[Satellite],
        reservation: Reservation,
        ephemeris_calculator: EphemerisCalculator,
        runtime_settings: RuntimeSettings | None = None,
    ):
        super().__init__(
            antenna_trajectory=antenna_trajectory,
            list_of_satellites=list_of_satellites,
            reservation=reservation,
            runtime_settings=runtime_settings,
        )
        self.ephemeris_calculator = ephemeris_calculator

    def get_satellites_above_horizon(self) -> list[SatelliteTrajectory]:
        trajectories = []

        for satellite in self.list_of_satellites:
            time_windows = self.ephemeris_calculator.calculate_visibility_windows(
                satellite,
                self.runtime_settings.min_altitude,
                self.reservation.time.begin,
                self.reservation.time.end,
            )

            if not time_windows:
                continue

            satellite_trajectories = self.ephemeris_calculator.calculate_trajectories(
                satellite, time_windows
            )

            trajectories.extend(satellite_trajectories)

        return trajectories

    def get_satellites_crossing_main_beam(self) -> list[SatelliteTrajectory]:
        beam_radius_sq = self.reservation.facility.beam_radius**2

        interfering_trajectories = []

        for satellite in self.list_of_satellites:
            # Get visibility windows
            windows = self.ephemeris_calculator.calculate_visibility_windows(
                satellite,
                self.runtime_settings.min_altitude,
                self.reservation.time.begin,
                self.reservation.time.end,
            )
            if not windows:
                continue

            # Get satellite path
            sat_trajectories = self.ephemeris_calculator.calculate_trajectories(
                satellite, windows
            )

            for sat_traj in sat_trajectories:
                # Interpolate Antenna position to match Satellite times
                ant_az, ant_alt = self.antenna_trajectory.get_state_at(sat_traj.times)

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

    # def _get_satellites_interference(self) -> list[SatelliteTrajectory]:
    #    processes = (
    #        int(self.runtime_settings.concurrency_level)
    #        if self.runtime_settings.concurrency_level > 1
    #        else 1
    #    )
    #    pool = multiprocessing.Pool(processes=processes)
    #    results = pool.map(
    #        self._get_satellite_overhead_windows, self.list_of_satellites
    #    )
    #    pool.close()
    #    pool.join()

    #    return [overhead_window for result in results for overhead_window in result]
