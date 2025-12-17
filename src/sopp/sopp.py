from functools import cached_property

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
from sopp.models.satellite.trajectory import SatelliteTrajectory
from sopp.pointing.base import PointingCalculator
from sopp.pointing.skyfield import PointingCalculatorSkyfield
from sopp.utils.time import generate_time_grid


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
        path_finder_class: type[PointingCalculator] = PointingCalculatorSkyfield,
    ):
        self.configuration = configuration
        self._path_finder_class = path_finder_class
        self._ephemeris_calculator_class = ephemeris_calculator_class

    def get_satellites_above_horizon(self) -> list[SatelliteTrajectory]:
        """
        Returns trajectories for all satellites that rise above the minimum altitude.
        """
        return find_satellites_above_horizon(
            reservation=self.configuration.reservation,
            satellites=self.configuration.satellites,
            ephemeris_calculator=self.ephemeris_calculator,
            min_altitude=self.configuration.runtime_settings.min_altitude,
        )

    def get_satellites_crossing_main_beam(self) -> list[SatelliteTrajectory]:
        """
        Returns trajectories for satellites that cross the antenna's main beam.
        """
        return find_satellites_crossing_main_beam(
            reservation=self.configuration.reservation,
            satellites=self.configuration.satellites,
            ephemeris_calculator=self.ephemeris_calculator,
            antenna_trajectory=self.antenna_trajectory,
            min_altitude=self.configuration.runtime_settings.min_altitude,
        )

    @cached_property
    def master_time_grid(self) -> np.ndarray:
        return generate_time_grid(
            start=self.configuration.reservation.time.begin,
            end=self.configuration.reservation.time.end,
            resolution_seconds=self.configuration.runtime_settings.time_resolution_seconds,
        )

    @cached_property
    def ephemeris_calculator(self) -> EphemerisCalculator:
        return self._ephemeris_calculator_class(
            facility=self.configuration.reservation.facility,
            datetimes=self.master_time_grid,
        )

    @cached_property
    def antenna_trajectory(self) -> AntennaTrajectory:
        """
        Builds the antenna trajectory based on the configuration variant.
        """
        config = self.configuration.antenna_config
        times = self.master_time_grid

        match config:
            case CustomTrajectoryConfig(trajectory=traj):
                return traj

            case CelestialTrackingConfig(target=target):
                path_finder = self._path_finder_class(
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
