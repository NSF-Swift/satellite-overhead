"""Fluent builder for constructing simulation configurations."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from sopp.config.loaders import ConfigFileLoaderBase, ConfigFileLoaderJson
from sopp.filtering.filterer import Filterer
from sopp.io.tle import load_satellites
from sopp.models.configuration import Configuration, RuntimeSettings
from sopp.models.core import Coordinates, FrequencyRange, Position, TimeWindow
from sopp.models.ground.config import (
    AntennaConfig,
    CelestialTrackingConfig,
    CustomTrajectoryConfig,
    StaticPointingConfig,
)
from sopp.models.ground.facility import Facility
from sopp.models.ground.receiver import Receiver
from sopp.models.ground.target import ObservationTarget
from sopp.models.reservation import Reservation
from sopp.utils.helpers import parse_time_and_convert_to_utc

if TYPE_CHECKING:
    from sopp.models.ground.trajectory import AntennaTrajectory
    from sopp.models.satellite.satellite import Satellite


class ConfigurationBuilder:
    """Fluent API for constructing a Configuration.

    Call setter methods in any order, then call ``build()`` to produce
    a validated Configuration. All setter methods return ``self`` for chaining.

    Example::

        config = (
            ConfigurationBuilder()
            .set_facility(lat, lon, elev, name, Receiver(beamwidth=3))
            .set_frequency_range(bandwidth=10, frequency=135)
            .set_time_window(begin="2024-01-01T00:00:00", end="2024-01-01T01:00:00")
            .set_observation_target(declination="7d24m25s", right_ascension="5h55m10s")
            .load_satellites(tle_file="satellites.tle")
            .build()
        )
    """

    def __init__(
        self,
    ):
        self.facility: Facility | None = None
        self.time_window: TimeWindow | None = None
        self.frequency_range: FrequencyRange | None = None

        self._filterer: Filterer = Filterer()

        self.antenna_config: AntennaConfig | None = None
        self.satellites: list[Satellite] | None = None
        self.reservation: Reservation | None = None
        self.runtime_settings: RuntimeSettings = RuntimeSettings()

    def set_facility(
        self,
        latitude: float,
        longitude: float,
        elevation: float,
        name: str,
        receiver: Receiver,
    ) -> ConfigurationBuilder:
        """Set the observation facility location and receiver characteristics."""
        self.facility = Facility(
            coordinates=Coordinates(latitude=latitude, longitude=longitude),
            receiver=receiver,
            elevation=elevation,
            name=name,
        )
        return self

    def set_frequency_range(self, bandwidth: float, frequency: float):
        """Set the observation frequency band in MHz."""
        self.frequency_range = FrequencyRange(
            bandwidth=bandwidth,
            frequency=frequency,
        )
        return self

    def set_time_window(
        self,
        begin: str | datetime,
        end: str | datetime,
    ) -> ConfigurationBuilder:
        """Set the observation time window. Strings are parsed as ISO 8601 UTC."""
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
    ) -> ConfigurationBuilder:
        """Set the antenna pointing mode.

        Provide exactly one of:
            - ``custom_antenna_trajectory``: explicit az/alt path over time.
            - ``altitude`` and ``azimuth``: fixed pointing direction.
            - ``declination`` and ``right_ascension``: celestial target to track.
        """
        # Option 1: Custom
        if custom_antenna_trajectory:
            self.antenna_config = CustomTrajectoryConfig(custom_antenna_trajectory)

        # Option 2: Static
        elif altitude is not None and azimuth is not None:
            self.antenna_config = StaticPointingConfig(
                Position(altitude=altitude, azimuth=azimuth)
            )

        # Option 3: Tracking
        elif declination is not None and right_ascension is not None:
            target = ObservationTarget(
                declination=declination, right_ascension=right_ascension
            )
            self.antenna_config = CelestialTrackingConfig(target)

        else:
            raise ValueError(
                "Invalid observation target configuration. Provide one of: "
                "(custom_antenna_trajectory), (altitude, azimuth), or (ra, dec)."
            )
        return self

    def set_satellites(self, satellites: list[Satellite]) -> ConfigurationBuilder:
        """Set the satellite list directly."""
        self.satellites = satellites
        return self

    def load_satellites(
        self, tle_file: str | Path, frequency_file: str | Path | None = None
    ) -> ConfigurationBuilder:
        """Load satellites from a TLE file, optionally attaching frequency data."""
        self.satellites = load_satellites(
            tle_file=Path(tle_file),
            frequency_file=Path(frequency_file) if frequency_file else None,
        )
        return self

    def set_runtime_settings(
        self,
        time_resolution_seconds: float = 1,
        concurrency_level: int = 1,
        min_altitude: float = 0.0,
    ) -> ConfigurationBuilder:
        """Set simulation resolution, parallelism, and minimum altitude."""
        self.runtime_settings = RuntimeSettings(
            concurrency_level=concurrency_level,
            time_resolution_seconds=time_resolution_seconds,
            min_altitude=min_altitude,
        )
        return self

    def set_from_config_file(
        self,
        config_file: Path,
        loader_class: type[ConfigFileLoaderBase] = ConfigFileLoaderJson,
    ) -> ConfigurationBuilder:
        """Load facility, frequency, time, and antenna settings from a config file."""
        loader = loader_class(filepath=config_file)

        self.frequency_range = loader.frequency_range
        self.facility = loader.facility
        self.time_window = loader.time_window
        self.runtime_settings = loader.runtime_settings
        self.antenna_config = loader.antenna_config

        return self

    def set_satellites_filter(self, filterer: Filterer) -> ConfigurationBuilder:
        """Replace the satellite filter with a pre-built Filterer."""
        self._filterer = filterer
        return self

    def add_filter(self, filter_fn: Callable[[Satellite], bool]):
        """Add a single satellite filter function."""
        self._filterer.add_filter(filter_fn)
        return self

    def build(self) -> Configuration:
        """Validate all settings and produce a Configuration.

        Raises:
            ValueError: If any required setting is missing.
        """
        facility = self.facility
        time_window = self.time_window
        frequency = self.frequency_range
        satellites = self.satellites
        antenna_config = self.antenna_config

        if facility is None:
            raise ValueError("Configuration invalid: Facility is not set.")
        if time_window is None:
            raise ValueError("Configuration invalid: Time Window is not set.")
        if satellites is None:
            raise ValueError("Configuration invalid: Satellites are not loaded.")

        filtered_satellites = self._filterer.apply_filters(satellites)

        reservation = Reservation(
            facility=facility,
            time=time_window,
            frequency=frequency,
        )

        return Configuration(
            reservation=reservation,
            satellites=filtered_satellites,
            antenna_config=antenna_config,
            runtime_settings=self.runtime_settings,
        )
