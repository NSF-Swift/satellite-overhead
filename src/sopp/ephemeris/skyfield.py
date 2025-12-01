import bisect
from datetime import datetime

from skyfield.api import load
from skyfield.toposlib import wgs84

from sopp.ephemeris.base import EphemerisCalculator
from sopp.models import Facility, Position, PositionTime, Satellite, TimeWindow

SKYFIELD_TIMESCALE = load.timescale()
RISE_EVENT = 0
SET_EVENT = 2


class SkyfieldEphemerisCalculator(EphemerisCalculator):
    def __init__(self, facility: Facility, datetimes: list[datetime]):
        self._facility = facility
        # We ensure the list is sorted so bisect works correctly.
        self._datetimes = sorted(datetimes)

        # This creates the vector of time objects once.
        # This is an expensive operation in Skyfield, so we cache it.
        if self._datetimes:
            self._grid_timescale = SKYFIELD_TIMESCALE.from_datetimes(self._datetimes)
        else:
            self._grid_timescale = None

        self._facility_latlon = self._calculate_facility_latlon()

    def find_events(
        self,
        satellite: Satellite,
        min_altitude: float,
        start_time: datetime,
        end_time: datetime,
    ) -> list[TimeWindow]:
        """
        Calculates Rise/Set/Culminate events.
        """
        t0 = SKYFIELD_TIMESCALE.from_datetime(start_time)
        t1 = SKYFIELD_TIMESCALE.from_datetime(end_time)

        sat_skyfield = satellite.to_skyfield()

        times, events = sat_skyfield.find_events(
            self._facility_latlon, t0, t1, altitude_degrees=min_altitude
        )

        windows = []
        current_rise = None
        for t, event in zip(times, events):
            if event == RISE_EVENT:
                current_rise = t.utc_datetime()
            elif event == SET_EVENT:
                if current_rise:
                    windows.append(TimeWindow(current_rise, t.utc_datetime()))
                    current_rise = None
                else:
                    # Started mid-pass
                    windows.append(TimeWindow(start_time, t.utc_datetime()))

        # Rose but didn't set
        if current_rise:
            windows.append(TimeWindow(current_rise, end_time))

        # Check for geostationary
        if not windows:
            # Check one point (start time)
            position_time = self.get_position_at(satellite, start_time)
            alt = position_time.position.altitude
            if alt >= min_altitude:
                windows.append(TimeWindow(start_time, end_time))

        return windows

    def get_positions_window(
        self, satellite: Satellite, start: datetime, end: datetime
    ) -> list[PositionTime]:
        """
        Returns satellite positions for all datetimes that fall
        inclusively within the provided start and end window.
        """

        start_idx = bisect.bisect_left(self._datetimes, start)
        end_idx = bisect.bisect_right(self._datetimes, end)

        if start_idx >= end_idx or self._grid_timescale is None:
            return []

        if start_idx == 0 and end_idx == len(self._datetimes):
            ts_obj = self._grid_timescale
            dt_subset = self._datetimes
        else:
            ts_obj = self._grid_timescale[start_idx:end_idx]
            dt_subset = self._datetimes[start_idx:end_idx]

        sat_skyfield = satellite.to_skyfield()
        difference = sat_skyfield - self._facility_latlon

        topocentric = difference.at(ts_obj)
        alt, az, dist = topocentric.altaz()

        return [
            PositionTime(
                Position(altitude=a, azimuth=z, distance_km=d),
                time=t,
            )
            for a, z, d, t in zip(
                alt.degrees,
                az.degrees,
                dist.km,
                dt_subset,
                strict=False,
            )
        ]

    def get_position_at(self, satellite: Satellite, time: datetime) -> PositionTime:
        """
        Retrieves the position for a single specific time.
        """

        idx = bisect.bisect_left(self._datetimes, time)

        target_t = None

        # Check if the index is valid and the time matches exactly
        if idx < len(self._datetimes) and self._datetimes[idx] == time:
            # Cache Hit: Extract the scalar Time object from our vector
            target_t = self._grid_timescale[idx]
        else:
            # Cache Miss: The requested time is not in our grid.
            target_t = SKYFIELD_TIMESCALE.from_datetime(time)

        sat_skyfield = satellite.to_skyfield()
        difference = sat_skyfield - self._facility_latlon

        topocentric = difference.at(target_t)
        alt, az, dist = topocentric.altaz()

        return PositionTime(
            Position(altitude=alt.degrees, azimuth=az.degrees, distance_km=dist.km),
            time=time,
        )

    def _calculate_facility_latlon(self):
        return wgs84.latlon(
            latitude_degrees=self._facility.coordinates.latitude,
            longitude_degrees=self._facility.coordinates.longitude,
            elevation_m=self._facility.elevation,
        )
