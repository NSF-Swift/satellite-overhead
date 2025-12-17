import re

import numpy as np
from skyfield.api import load
from skyfield.starlib import Star
from skyfield.toposlib import wgs84

from sopp.models.core import TimeWindow
from sopp.models.ground.facility import Facility
from sopp.models.ground.target import ObservationTarget
from sopp.models.ground.trajectory import AntennaTrajectory
from sopp.pointing.base import PointingCalculator
from sopp.utils.time import generate_time_grid


class PointingCalculatorSkyfield(PointingCalculator):
    def __init__(
        self,
        facility: Facility,
        observation_target: ObservationTarget,
        time_window: TimeWindow,
    ):
        self._facility = facility
        self._observation_target = observation_target
        self._time_window = time_window

    def calculate_trajectory(
        self,
        resolution_seconds: float = 1.0,
        time_grid: np.ndarray | None = None,
    ) -> AntennaTrajectory:
        if time_grid is None:
            time_grid = generate_time_grid(
                self._time_window.begin, self._time_window.end, resolution_seconds
            )

        # TODO: load only a single time
        ts = load.timescale()
        t_vector = ts.from_datetimes(time_grid)

        observing_location = wgs84.latlon(
            latitude_degrees=self._facility.coordinates.latitude,
            longitude_degrees=self._facility.coordinates.longitude,
            elevation_m=self._facility.elevation,
        )

        eph = load("de421.bsp")
        earth = eph["earth"]

        target_coordinates = Star(
            ra_hours=self.right_ascension_to_skyfield(self._observation_target),
            dec_degrees=self.declination_to_skyfield(self._observation_target),
        )

        astrometric = (
            (earth + observing_location).at(t_vector).observe(target_coordinates)
        )
        apparent = astrometric.apparent()
        alt, az, _ = apparent.altaz()

        return AntennaTrajectory(
            times=time_grid, azimuth=az.degrees, altitude=alt.degrees
        )

    @staticmethod
    def _parse_coordinate(coordinate_str: str) -> tuple[float, ...]:
        parts = [float(part) for part in re.split("[hdms]", coordinate_str) if part]

        return tuple(parts)

    @staticmethod
    def right_ascension_to_skyfield(
        observation_target: ObservationTarget,
    ) -> tuple[float, ...]:
        return PointingCalculatorSkyfield._parse_coordinate(
            observation_target.right_ascension
        )

    @staticmethod
    def declination_to_skyfield(
        observation_target: ObservationTarget,
    ) -> tuple[float, ...]:
        return PointingCalculatorSkyfield._parse_coordinate(
            observation_target.declination
        )
