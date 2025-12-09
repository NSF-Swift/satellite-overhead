from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from sopp.config.base import ConfigFileLoaderBase
from sopp.config.json_loader import ConfigFileLoaderJson
from sopp.io.satellites_loader import (
    SatellitesLoaderFromFiles,
)
from sopp.models.antenna_trajectory import AntennaTrajectory
from sopp.models.configuration import Configuration
from sopp.models.coordinates import Coordinates
from sopp.models.facility import Facility
from sopp.models.frequency_range import FrequencyRange
from sopp.models.observation_target import ObservationTarget
from sopp.models.position import Position
from sopp.models.reservation import Reservation
from sopp.models.runtime_settings import RuntimeSettings
from sopp.models.satellite.satellite import Satellite
from sopp.models.time_window import TimeWindow
from sopp.path_finders.base import ObservationPathFinder
from sopp.path_finders.skyfield import (
    ObservationPathFinderSkyfield,
)
from sopp.satellite_selection.filterer import Filterer
from sopp.utils.helpers import parse_time_and_convert_to_utc
from sopp.utils.time import generate_time_grid


class ConfigurationBuilder:
    def __init__(
        self,
        path_finder_class: type[ObservationPathFinder] = ObservationPathFinderSkyfield,
        config_file_loader_class: type[ConfigFileLoaderBase] = ConfigFileLoaderJson,
    ):
        self.facility: Facility | None = None
        self.time_window: TimeWindow | None = None
        self.frequency_range: FrequencyRange | None = None

        self._path_finder_class = path_finder_class
        self._config_file_loader_class = config_file_loader_class

        self._filterer: Filterer = Filterer()

        self._observation_target: ObservationTarget | None = None
        self._static_pointing: Position | None = None
        self._custom_antenna_trajectory: AntennaTrajectory | None = None

        self.antenna_trajectory: AntennaTrajectory | None = None
        self.satellites: list[Satellite] | None = None
        self.reservation: Reservation | None = None
        self.runtime_settings: RuntimeSettings = RuntimeSettings()

    def set_facility(
        self,
        latitude: float,
        longitude: float,
        elevation: float,
        name: str,
        beamwidth: float,
    ) -> "ConfigurationBuilder":
        self.facility = Facility(
            Coordinates(latitude=latitude, longitude=longitude),
            elevation=elevation,
            beamwidth=beamwidth,
            name=name,
        )
        return self

    def set_frequency_range(self, bandwidth: float, frequency: float):
        self.frequency_range = FrequencyRange(
            bandwidth=bandwidth,
            frequency=frequency,
        )
        return self

    def set_time_window(
        self,
        begin: str | datetime,
        end: str | datetime,
    ) -> "ConfigurationBuilder":
        self.time_window = TimeWindow(
            begin=parse_time_and_convert_to_utc(begin),
            end=parse_time_and_convert_to_utc(end),
        )
        return self

    def set_observation_target(
        self,
        declination: str | None = None,
        right_ascension: str | None = None,
        altitude: float | None = None,
        azimuth: float | None = None,
        custom_antenna_trajectory: AntennaTrajectory | None = None,
    ) -> "ConfigurationBuilder":
        # Option 1: Pre-calculated Trajectory
        if custom_antenna_trajectory:
            self._custom_antenna_trajectory = custom_antenna_trajectory

        # Option 2: Static Pointing (Alt/Az)
        elif altitude is not None and azimuth is not None:
            self._static_pointing = Position(altitude=altitude, azimuth=azimuth)

        # Option 3: Track Celestial Object (RA/Dec)
        elif declination is not None and right_ascension is not None:
            self._observation_target = ObservationTarget(
                declination=declination,
                right_ascension=right_ascension,
            )
        else:
            raise ValueError(
                "Invalid observation target configuration. Provide one of: "
                "(custom_antenna_trajectory), (altitude, azimuth), or (ra, dec)."
            )
        return self

    def set_satellites(
        self, tle_file: str, frequency_file: str | None = None
    ) -> "ConfigurationBuilder":
        self.satellites = SatellitesLoaderFromFiles(
            tle_file=tle_file,
            frequency_file=frequency_file,
        ).load_satellites()
        return self

    def set_runtime_settings(
        self,
        time_resolution_seconds: float = 1,
        concurrency_level: int = 1,
        min_altitude: float = 0.0,
    ) -> "ConfigurationBuilder":
        self.runtime_settings = RuntimeSettings(
            concurrency_level=concurrency_level,
            time_resolution_seconds=time_resolution_seconds,
            min_altitude=min_altitude,
        )
        return self

    def set_from_config_file(
        self, config_file: Path | None = None
    ) -> "ConfigurationBuilder":
        config = self._config_file_loader_class(filepath=config_file).configuration
        self.frequency_range = config.reservation.frequency
        self.facility = config.reservation.facility
        self.time_window = config.reservation.time
        self.runtime_settings = config.runtime_settings

        if config.antenna_trajectory:
            self._custom_observation_path = config.antenna_position_times
        elif config.static_antenna_position:
            self._static_pointing = config.static_antenna_position
        else:
            self._observation_target = config.observation_target
        return self

    def set_satellites_filter(self, filterer: type[Filterer]) -> "ConfigurationBuilder":
        self._filterer = filterer
        return self

    def add_filter(self, filter_fn: Callable[[Satellite, Any], bool]):
        self._filterer.add_filter(filter_fn)
        return self

    def _build_antenna_trajectory(self) -> AntennaTrajectory:
        if not self.time_window:
            raise ValueError("Time window must be set before building trajectory.")

        res_seconds = self.runtime_settings.time_resolution_seconds

        if self._custom_antenna_trajectory:
            return self._custom_antenna_trajectory

        if self._observation_target and self.facility:
            path_finder = self._path_finder_class(
                self.facility, self._observation_target, self.time_window
            )
            return path_finder.calculate_path(resolution_seconds=res_seconds)

        static_az = 0.0
        static_alt = 90.0

        if self._static_pointing:
            static_az = self._static_pointing.azimuth
            static_alt = self._static_pointing.altitude

        times = generate_time_grid(
            self.time_window.begin,
            self.time_window.end,
            resolution_seconds=res_seconds,
        )
        n = len(times)

        return AntennaTrajectory(
            times=times,
            azimuth=np.full(n, static_az),
            altitude=np.full(n, static_alt),
        )

    def build(self) -> Configuration:
        facility = self.facility
        time_window = self.time_window
        frequency = self.frequency_range
        satellites = self.satellites

        if facility is None:
            raise ValueError("Configuration invalid: Facility is not set.")
        if time_window is None:
            raise ValueError("Configuration invalid: Time Window is not set.")
        if frequency is None:
            raise ValueError("Configuration invalid: Frequency Range is not set.")
        if satellites is None:
            raise ValueError("Configuration invalid: Satellites are not loaded.")

        filtered_satellites = self._filterer.apply_filters(satellites)

        reservation = Reservation(
            facility=facility,
            time=time_window,
            frequency=frequency,
        )

        trajectory = self._build_antenna_trajectory()

        return Configuration(
            reservation=reservation,
            satellites=filtered_satellites,
            antenna_trajectory=trajectory,
            runtime_settings=self.runtime_settings,
        )
