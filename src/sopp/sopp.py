from functools import cached_property

from sopp.ephemeris.skyfield import SkyfieldEphemerisCalculator
from sopp.event_finders.base import EventFinder
from sopp.event_finders.skyfield import (
    EventFinderSkyfield,
)
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
            antenna_trajectory=self._configuration.antenna_trajectory,
            ephemeris_calculator=ephemeris_calculator,
            runtime_settings=self._configuration.runtime_settings,
        )
