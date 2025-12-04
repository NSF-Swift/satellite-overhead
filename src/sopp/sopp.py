from datetime import timedelta
from functools import cached_property

from sopp.analysis.event_finders.base import EventFinder
from sopp.analysis.event_finders.skyfield import (
    EventFinderSkyfield,
)
from sopp.ephemeris.skyfield import SkyfieldEphemerisCalculator
from sopp.models import Configuration, SatelliteTrajectory
from sopp.utils.time import generate_time_grid


class Sopp:
    def __init__(
        self,
        configuration: Configuration,
        event_finder: EventFinder | None = None,
    ):
        self._configuration = configuration
        self._injected_event_finder = event_finder

    def get_satellites_above_horizon(self) -> list[SatelliteTrajectory]:
        return self._event_finder.get_satellites_above_horizon()

    def get_satellites_crossing_main_beam(self) -> list[SatelliteTrajectory]:
        return self._event_finder.get_satellites_crossing_main_beam()

    @cached_property
    def _event_finder(self) -> EventFinder:
        if self._injected_event_finder:
            return self._injected_event_finder

        self._validate_configuration()

        datetimes = generate_time_grid(
            start=self._configuration.reservation.time.begin,
            end=self._configuration.reservation.time.end,
            resolution_seconds=self._configuration.runtime_settings.time_resolution_seconds,
        )

        ephemeris_calculator = SkyfieldEphemerisCalculator(
            facility=self._configuration.reservation.facility, datetimes=datetimes
        )

        return EventFinderSkyfield(
            list_of_satellites=self._configuration.satellites,
            reservation=self._configuration.reservation,
            antenna_direction_path=self._configuration.antenna_direction_path,
            ephemeris_calculator=ephemeris_calculator,
            runtime_settings=self._configuration.runtime_settings,
        )

    def _validate_configuration(self):
        self._validate_satellites()
        self._validate_runtime_settings()
        self._validate_reservation()
        self._validate_antenna_direction_path()

    def _validate_satellites(self):
        satellites = self._configuration.satellites
        if not satellites:
            raise ValueError("Satellites list empty.")

    def _validate_runtime_settings(self):
        runtime_settings = self._configuration.runtime_settings
        if runtime_settings.time_resolution_seconds <= 0.0:
            raise ValueError(
                f"time_resolution_seconds must be at least 1 second, provided: {runtime_settings.time_resolution_seconds}"
            )
        if runtime_settings.concurrency_level < 1:
            raise ValueError(
                f"concurrency_level must be at least 1, provided: {runtime_settings.concurrency_level}"
            )
        if runtime_settings.min_altitude < 0.0:
            raise ValueError(
                f"min_altitude must be non-negative, provided: {runtime_settings.min_altitude}"
            )

    def _validate_reservation(self):
        reservation = self._configuration.reservation
        if reservation.time.begin >= reservation.time.end:
            raise ValueError(
                f"reservation.time.begin time is later than or equal to end time, provided begin: {reservation.time.begin} end: {reservation.time.end}"
            )
        if reservation.facility.beamwidth <= 0:
            raise ValueError(
                f"reservation.facility.beamwidth must be greater than 0, provided: {reservation.facility.beamwidth}"
            )

    def _validate_antenna_direction_path(self):
        antenna_direction_path = self._configuration.antenna_direction_path
        if not antenna_direction_path:
            raise ValueError("No antenna direction path provided.")

        for current_time, next_time in zip(
            antenna_direction_path, antenna_direction_path[1:], strict=False
        ):
            if current_time.time >= next_time.time:
                raise ValueError("Times in antenna_direction_path must be increasing.")
