from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from sopp.analysis.path_finders.base import ObservationPathFinder
from sopp.analysis.path_finders.skyfield import (
    ObservationPathFinderSkyfield,
)
from sopp.config.base import ConfigFileLoaderBase
from sopp.config.json_loader import ConfigFileLoaderJson
from sopp.io.satellites_loader import (
    SatellitesLoaderFromFiles,
)
from sopp.models import (
    Configuration,
    Coordinates,
    Facility,
    FrequencyRange,
    ObservationTarget,
    Position,
    PositionTime,
    Reservation,
    RuntimeSettings,
    Satellite,
    TimeWindow,
)
from sopp.satellite_selection.filterer import Filterer
from sopp.utils.helpers import parse_time_and_convert_to_utc


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
        self._static_observation_target: Position | None = None
        self._custom_observation_path: list[PositionTime] | None = None

        self.antenna_direction_path: list[PositionTime] | None = None
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
        custom_path: list[PositionTime] | None = None,
    ) -> "ConfigurationBuilder":
        if custom_path:
            self._custom_observation_path = custom_path
        elif altitude is not None and azimuth is not None:
            self._static_observation_target = Position(
                altitude=altitude, azimuth=azimuth
            )
        elif declination is not None and right_ascension is not None:
            self._observation_target = ObservationTarget(
                declination=declination,
                right_ascension=right_ascension,
            )
        else:
            raise ValueError(
                "Specify at least one way to set the observation target. "
                "Valid combinations are: "
                "1. right_ascension and declination, "
                "2. altitude and azimuth, or "
                "3. custom_path."
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
        time_continuity_resolution: int | None = 1,
        concurrency_level: int | None = 1,
        min_altitude: float | None = 0.0,
    ) -> "ConfigurationBuilder":
        self.runtime_settings = RuntimeSettings(
            concurrency_level=concurrency_level,
            time_continuity_resolution=time_continuity_resolution,
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

        if config.antenna_position_times:
            self._custom_observation_path = config.antenna_position_times
        elif config.static_antenna_position:
            self._static_observation_target = config.static_antenna_position
        else:
            self._observation_target = config.observation_target
        return self

    def set_satellites_filter(self, filterer: type[Filterer]) -> "ConfigurationBuilder":
        self._filterer = filterer
        return self

    def add_filter(self, filter_fn: Callable[[Satellite, Any], bool]):
        self._filterer.add_filter(filter_fn)
        return self

    def _filter_satellites(self):
        self.satellites = self._filterer.apply_filters(self.satellites)

    def _build_reservation(self):
        self.reservation = Reservation(
            facility=self.facility,
            time=self.time_window,
            frequency=self.frequency_range,
        )

    def _build_antenna_direction_path(self) -> "ConfigurationBuilder":
        if self._custom_observation_path:
            self._antenna_direction_path = self._custom_observation_path
        elif self._static_observation_target:
            self._antenna_direction_path = [
                PositionTime(
                    position=self._static_observation_target,
                    time=self.time_window.begin,
                )
            ]
        elif self._observation_target:
            self._antenna_direction_path = self._path_finder_class(
                self.facility, self._observation_target, self.time_window
            ).calculate_path()
        else:
            self._antenna_direction_path = [
                PositionTime(
                    position=Position(altitude=90, azimuth=0),
                    time=self.reservation.time.begin,
                ),
            ]

    def build(self) -> "Configuration":
        if not (
            all(
                [self.facility, self.time_window, self.frequency_range, self.satellites]
            )
        ):
            raise ValueError(
                "Incomplete configuration. Ensure that the following are called: "
                "set_facility, set_time_window, set_frequency_range, "
                "set_satellites, or set_from_config_file."
            )

        self._filter_satellites()
        self._build_reservation()
        self._build_antenna_direction_path()

        configuration = Configuration(
            reservation=self.reservation,
            satellites=self.satellites,
            antenna_direction_path=self._antenna_direction_path,
            runtime_settings=self.runtime_settings,
        )
        return configuration
