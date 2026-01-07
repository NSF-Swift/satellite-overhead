from functools import cached_property
from concurrent.futures import ProcessPoolExecutor
from typing import Type, List, Tuple

import numpy as np

from sopp.analysis.interference import (
    find_satellites_above_horizon,
    find_satellites_crossing_main_beam,
)
from sopp.ephemeris.base import EphemerisCalculator
from sopp.ephemeris.skyfield import SkyfieldEphemerisCalculator
from sopp.models.configuration import Configuration
from sopp.models.ground.config import (
    CelestialTrackingConfig,
    CustomTrajectoryConfig,
    StaticPointingConfig,
)
from sopp.models.ground.trajectory import AntennaTrajectory
from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.trajectory import SatelliteTrajectory
from sopp.pointing.base import PointingCalculator
from sopp.pointing.skyfield import PointingCalculatorSkyfield
from sopp.utils.time import generate_time_grid


def _parallel_horizon_worker(payload: Tuple) -> List[SatelliteTrajectory]:
    """
    Rehydrates the simulation context on a worker process and calculates visibility.
    """
    satellites, reservation, runtime_settings, calc_class = payload

    # Regenerate Time Grid
    datetimes = generate_time_grid(
        start=reservation.time.begin,
        end=reservation.time.end,
        resolution_seconds=runtime_settings.time_resolution_seconds,
    )

    # Initialize Calculator
    calculator = calc_class(facility=reservation.facility, datetimes=datetimes)

    return find_satellites_above_horizon(
        reservation=reservation,
        satellites=satellites,
        ephemeris_calculator=calculator,
        min_altitude=runtime_settings.min_altitude,
    )


def _parallel_beam_worker(payload: Tuple) -> List[SatelliteTrajectory]:
    """
    Rehydrates context and calculates beam crossings.
    """
    satellites, reservation, runtime_settings, calc_class, antenna_traj = payload

    # Regenerate Time Grid
    datetimes = generate_time_grid(
        start=reservation.time.begin,
        end=reservation.time.end,
        resolution_seconds=runtime_settings.time_resolution_seconds,
    )

    # Initialize Calculator
    calculator = calc_class(facility=reservation.facility, datetimes=datetimes)

    return find_satellites_crossing_main_beam(
        reservation=reservation,
        satellites=satellites,
        ephemeris_calculator=calculator,
        antenna_trajectory=antenna_traj,
        min_altitude=runtime_settings.min_altitude,
    )


class Sopp:
    """
    The main entry point for the SOPP simulation engine.
    """

    def __init__(
        self,
        configuration: Configuration,
        ephemeris_calculator_class: Type[
            EphemerisCalculator
        ] = SkyfieldEphemerisCalculator,
        pointing_calculator_class: Type[
            PointingCalculator
        ] = PointingCalculatorSkyfield,
    ):
        self.configuration = configuration
        self._pointing_calculator_class = pointing_calculator_class
        self._ephemeris_calculator_class = ephemeris_calculator_class

    def get_satellites_above_horizon(self) -> List[SatelliteTrajectory]:
        """
        Returns trajectories for all satellites that rise above the minimum altitude.
        """
        satellites = self.configuration.satellites
        concurrency = self.configuration.runtime_settings.concurrency_level

        # Serial Fallback
        if concurrency <= 1:
            return find_satellites_above_horizon(
                reservation=self.configuration.reservation,
                satellites=satellites,
                ephemeris_calculator=self.ephemeris_calculator,
                min_altitude=self.configuration.runtime_settings.min_altitude,
            )

        # Parallel Execution
        return self._run_parallel(
            worker_func=_parallel_horizon_worker,
            satellites=satellites,
            include_antenna=False,
        )

    def get_satellites_crossing_main_beam(self) -> List[SatelliteTrajectory]:
        """
        Returns trajectories for satellites that cross the antenna's main beam.
        """
        satellites = self.configuration.satellites
        concurrency = self.configuration.runtime_settings.concurrency_level

        # Serial Fallback
        if concurrency <= 1:
            return find_satellites_crossing_main_beam(
                reservation=self.configuration.reservation,
                satellites=satellites,
                ephemeris_calculator=self.ephemeris_calculator,
                antenna_trajectory=self.antenna_trajectory,
                min_altitude=self.configuration.runtime_settings.min_altitude,
            )

        # Parallel Execution
        return self._run_parallel(
            worker_func=_parallel_beam_worker,
            satellites=satellites,
            include_antenna=True,
        )

    def _run_parallel(
        self, worker_func, satellites: List[Satellite], include_antenna: bool
    ) -> List[SatelliteTrajectory]:
        """
        Orchestrates the distribution of work to the process pool.
        """
        concurrency = self.configuration.runtime_settings.concurrency_level

        # Divide satellites into N chunks
        chunk_size = int(np.ceil(len(satellites) / concurrency))
        if chunk_size == 0:
            return []

        chunks = [
            satellites[i : i + chunk_size]
            for i in range(0, len(satellites), chunk_size)
        ]

        # Pass base data classes
        base_args = [
            self.configuration.reservation,
            self.configuration.runtime_settings,
            self._ephemeris_calculator_class,
        ]

        if include_antenna:
            base_args.append(self.antenna_trajectory)

        tasks = [(chunk, *base_args) for chunk in chunks]

        results = []
        with ProcessPoolExecutor(max_workers=concurrency) as executor:
            for batch_result in executor.map(worker_func, tasks):
                results.extend(batch_result)

        return results

    @cached_property
    def master_time_grid(self) -> np.ndarray:
        return generate_time_grid(
            start=self.configuration.reservation.time.begin,
            end=self.configuration.reservation.time.end,
            resolution_seconds=self.configuration.runtime_settings.time_resolution_seconds,
        )

    @cached_property
    def ephemeris_calculator(self) -> EphemerisCalculator:
        # Only used for Serial execution
        return self._ephemeris_calculator_class(
            facility=self.configuration.reservation.facility,
            datetimes=self.master_time_grid,
        )

    @cached_property
    def antenna_trajectory(self) -> AntennaTrajectory:
        config = self.configuration.antenna_config
        times = self.master_time_grid

        match config:
            case CustomTrajectoryConfig(trajectory=traj):
                return traj

            case CelestialTrackingConfig(target=target):
                path_finder = self._pointing_calculator_class(
                    facility=self.configuration.reservation.facility,
                    observation_target=target,
                    time_window=self.configuration.reservation.time,
                )
                return path_finder.calculate_trajectory(time_grid=times)

            case StaticPointingConfig(position=pos):
                n = len(times)
                return AntennaTrajectory(
                    times=times,
                    azimuth=np.full(n, pos.azimuth),
                    altitude=np.full(n, pos.altitude),
                )

            case _:
                raise ValueError(f"Unknown antenna configuration: {config}")
