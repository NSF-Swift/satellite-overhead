import re
from datetime import timedelta

from skyfield.api import load
from skyfield.starlib import Star
from skyfield.toposlib import wgs84

from sopp.analysis.path_finders.base import ObservationPathFinder
from sopp.models.facility import Facility
from sopp.models.observation_target import ObservationTarget
from sopp.models.position import Position
from sopp.models.position_time import PositionTime
from sopp.models.time_window import TimeWindow


class ObservationPathFinderRhodesmill(ObservationPathFinder):
    def __init__(
        self,
        facility: Facility,
        observation_target: ObservationTarget,
        time_window: TimeWindow,
    ) -> list[PositionTime]:
        self._facility = facility
        self._observation_target = observation_target
        self._time_window = time_window

    def calculate_path(self) -> list[PositionTime]:
        observation_path = []
        observing_location = wgs84.latlon(
            latitude_degrees=self._facility.coordinates.latitude,
            longitude_degrees=self._facility.coordinates.longitude,
            elevation_m=self._facility.elevation,
        )

        ts = load.timescale()
        eph = load("de421.bsp")
        earth = eph["earth"]

        target_coordinates = Star(
            ra_hours=ObservationPathFinderRhodesmill.right_ascension_to_rhodesmill(
                self._observation_target
            ),
            dec_degrees=ObservationPathFinderRhodesmill.declination_to_rhodesmill(
                self._observation_target
            ),
        )
        start_time = self._time_window.begin
        end_time = self._time_window.end

        while start_time <= end_time:
            observing_time = ts.from_datetime(start_time)

            astrometric = (
                (earth + observing_location)
                .at(observing_time)
                .observe(target_coordinates)
            )
            position = astrometric.apparent()
            alt, az, _ = position.altaz()

            point = PositionTime(
                position=Position(altitude=alt.degrees, azimuth=az.degrees),
                time=start_time,
            )
            observation_path.append(point)
            start_time += timedelta(minutes=1)

        return observation_path

    @staticmethod
    def _parse_coordinate(coordinate_str: str) -> tuple[float, float, float]:
        parts = [float(part) for part in re.split("[hdms]", coordinate_str) if part]

        return tuple(parts)

    @staticmethod
    def right_ascension_to_rhodesmill(
        observation_target: ObservationTarget,
    ) -> tuple[float, float, float]:
        return ObservationPathFinderRhodesmill._parse_coordinate(
            observation_target.right_ascension
        )

    @staticmethod
    def declination_to_rhodesmill(
        observation_target: ObservationTarget,
    ) -> tuple[float, float, float]:
        return ObservationPathFinderRhodesmill._parse_coordinate(
            observation_target.declination
        )
