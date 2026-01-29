from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sopp.analysis.interference import analyze_interference
from sopp.analysis.strategies import GeometricStrategy, InterferenceResult
from sopp.analysis.visibility import find_satellites_above_horizon
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

if TYPE_CHECKING:
    from sopp.analysis.strategies import InterferenceStrategy
    from sopp.io.formats.base import TrajectoryFormat


def _parallel_horizon_worker(payload: tuple) -> list[SatelliteTrajectory]:
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


class Sopp:
    """
    The main entry point for the SOPP simulation engine.
    """

    def __init__(
        self,
        configuration: Configuration,
        ephemeris_calculator_class: type[
            EphemerisCalculator
        ] = SkyfieldEphemerisCalculator,
        pointing_calculator_class: type[
            PointingCalculator
        ] = PointingCalculatorSkyfield,
    ):
        self.configuration = configuration
        self._pointing_calculator_class = pointing_calculator_class
        self._ephemeris_calculator_class = ephemeris_calculator_class

    def get_satellites_above_horizon(self) -> list[SatelliteTrajectory]:
        """
        Returns trajectories for all satellites that rise above the minimum altitude.
        """
        satellites = self.configuration.satellites
        concurrency = self.configuration.runtime_settings.concurrency_level

        if concurrency <= 1:
            return find_satellites_above_horizon(
                reservation=self.configuration.reservation,
                satellites=satellites,
                ephemeris_calculator=self.ephemeris_calculator,
                min_altitude=self.configuration.runtime_settings.min_altitude,
            )

        return self._compute_trajectories_parallel(satellites=satellites)

    def get_satellites_crossing_main_beam(self) -> list[SatelliteTrajectory]:
        """
        Returns trajectories for satellites that cross the antenna's main beam.
        """
        trajectories = self.get_satellites_above_horizon()
        results = self.analyze(trajectories, GeometricStrategy())
        return [r.trajectory for r in results]

    def analyze(
        self,
        trajectories: list[SatelliteTrajectory],
        strategy: InterferenceStrategy,
    ) -> list[InterferenceResult]:
        """Apply an interference strategy to pre-computed trajectories.

        Use this when trajectories have already been computed or loaded from
        disk and you want to run interference analysis without recalculating
        orbital positions.

        Args:
            trajectories: Pre-computed satellite trajectories.
            strategy: The interference detection strategy to apply.

        Returns:
            List of InterferenceResult for trajectories where interference
            was detected.

        Example:
            from sopp.analysis.strategies import GeometricStrategy
            from sopp.io import load_trajectories

            engine = Sopp(configuration)
            cached = load_trajectories("trajectories.arrow")
            results = engine.analyze(cached, GeometricStrategy())
        """
        return analyze_interference(
            trajectories=trajectories,
            antenna_trajectory=self.antenna_trajectory,
            strategy=strategy,
            facility=self.configuration.reservation.facility,
            frequency=self.configuration.reservation.frequency,
        )

    def _compute_trajectories_parallel(
        self, satellites: list[Satellite]
    ) -> list[SatelliteTrajectory]:
        """
        Distributes trajectory computation across a process pool.
        """
        concurrency = self.configuration.runtime_settings.concurrency_level

        chunk_size = int(np.ceil(len(satellites) / concurrency))
        if chunk_size == 0:
            return []

        chunks = [
            satellites[i : i + chunk_size]
            for i in range(0, len(satellites), chunk_size)
        ]

        base_args = [
            self.configuration.reservation,
            self.configuration.runtime_settings,
            self._ephemeris_calculator_class,
        ]

        tasks = [(chunk, *base_args) for chunk in chunks]

        results = []
        with ProcessPoolExecutor(max_workers=concurrency) as executor:
            for batch_result in executor.map(_parallel_horizon_worker, tasks):
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

    def save_trajectories(
        self,
        trajectories: list[SatelliteTrajectory],
        path: str | Path,
        *,
        format: TrajectoryFormat | None = None,
    ) -> Path:
        """Save computed trajectories to a file.

        Convenience method that saves trajectories with observer information
        from the current configuration.

        Args:
            trajectories: List of trajectories to save (from get_satellites_above_horizon
                         or get_satellites_crossing_main_beam).
            path: Output file path.
            format: File format handler. Defaults to ArrowFormat.

        Returns:
            Path to the saved file.

        Example:
            engine = Sopp(configuration)
            trajectories = engine.get_satellites_above_horizon()
            engine.save_trajectories(trajectories, "cache/trajectories.arrow")
        """
        facility = self.configuration.reservation.facility

        from sopp.io import save_trajectories

        return save_trajectories(
            trajectories,
            path,
            format=format,
            observer_name=facility.name,
            observer_lat=facility.coordinates.latitude,
            observer_lon=facility.coordinates.longitude,
        )
