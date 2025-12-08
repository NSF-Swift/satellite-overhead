import bisect
from datetime import datetime

import numpy as np
from skyfield.api import load
from skyfield.timelib import Time
from skyfield.toposlib import wgs84

from sopp.ephemeris.base import EphemerisCalculator
from sopp.models import (
    Facility,
    Position,
    Satellite,
    SatelliteTrajectory,
    TimeWindow,
)

SKYFIELD_TIMESCALE = load.timescale()
RISE_EVENT = 0
SET_EVENT = 2


class SkyfieldEphemerisCalculator(EphemerisCalculator):
    def __init__(self, facility: Facility, datetimes: list[datetime] | np.ndarray):
        self._facility = facility
        self._facility_latlon = self._calculate_facility_latlon()

        if isinstance(datetimes, np.ndarray):
            self._datetimes = np.array(datetimes, dtype=object, copy=True)
        else:
            self._datetimes = np.array(datetimes, dtype=object)

        # Sort so bisect works correctly
        if len(self._datetimes) > 0:
            self._datetimes.sort()

        self._grid_timescale = SKYFIELD_TIMESCALE.from_datetimes(self._datetimes)

        # Calcualte matrices
        if "M" not in self._grid_timescale.__dict__:
            _ = self._grid_timescale.M
        if "gast" not in self._grid_timescale.__dict__:
            _ = self._grid_timescale.gast
        if "gmst" not in self._grid_timescale.__dict__:
            _ = self._grid_timescale.gmst
        if "delta_t" not in self._grid_timescale.__dict__:
            _ = self._grid_timescale.delta_t

    def calculate_visibility_windows(
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
        for t, event in zip(times, events, strict=True):
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
            position = self.calculate_position(satellite, start_time)
            alt = position.altitude
            if alt >= min_altitude:
                windows.append(TimeWindow(start_time, end_time))

        return windows

    def calculate_trajectory(
        self, satellite: Satellite, start: datetime, end: datetime
    ) -> SatelliteTrajectory:
        window = TimeWindow(start, end)
        results = self.calculate_trajectories(satellite, [window])

        if not results:
            return SatelliteTrajectory(
                satellite,
                times=np.array([]),
                azimuth=np.array([]),
                altitude=np.array([]),
                distance_km=np.array([]),
            )

        return results[0]

    def calculate_trajectories(
        self, satellite: Satellite, windows: list[TimeWindow]
    ) -> list[SatelliteTrajectory]:
        if not windows:
            return []

        max_idx = len(self._datetimes)

        indices_list = []
        window_lengths = []

        # Map the continuous TimeWindows to the discrete grid indices.
        for win in windows:
            # Find the index range in the master grid
            start_idx = bisect.bisect_left(self._datetimes, win.begin)
            end_idx = bisect.bisect_right(self._datetimes, win.end)

            # Clamp and Check
            if start_idx >= max_idx or end_idx <= start_idx:
                continue

            # Create the indices for this window
            indices = np.arange(start_idx, end_idx)
            indices_list.append(indices)
            window_lengths.append(len(indices))

        if not indices_list:
            # All windows were out of bounds of the discrete grid
            return []

        # Combine into one master index array for this satellite
        all_indices = np.concatenate(indices_list)

        # Create subset time object and inject earth physics matrices
        ts_subset = self._grid_timescale[all_indices]
        self._inject_earth_physics(ts_subset, all_indices)

        # Positions calculation
        sat_skyfield = satellite.to_skyfield()
        difference = sat_skyfield - self._facility_latlon
        topocentric = difference.at(ts_subset)
        alt, az, dist = topocentric.altaz()

        alt_deg = alt.degrees
        az_deg = az.degrees
        dist_km = dist.km

        results = []
        current_offset = 0

        for length in window_lengths:
            end_offset = current_offset + length

            w_alt = alt_deg[current_offset:end_offset]
            w_az = az_deg[current_offset:end_offset]
            w_dist = dist_km[current_offset:end_offset]
            w_indices = all_indices[current_offset:end_offset]

            w_times = self._datetimes[w_indices]

            results.append(
                SatelliteTrajectory(
                    satellite=satellite,
                    times=w_times,
                    altitude=w_alt,
                    azimuth=w_az,
                    distance_km=w_dist,
                )
            )

            current_offset += length

        return results

    def calculate_position(self, satellite: Satellite, time: datetime) -> Position:
        """
        Retrieves the position for a single specific time.
        """

        idx = bisect.bisect_left(self._datetimes, time)
        target_t = None

        # Check if the index is valid and the time matches exactly
        if idx < len(self._datetimes) and self._datetimes[idx] == time:
            target_t = self._grid_timescale[idx]
            self._inject_earth_physics(target_t, idx)
        else:
            # The requested time is not in our grid.
            target_t = SKYFIELD_TIMESCALE.from_datetime(time)

        sat_skyfield = satellite.to_skyfield()
        difference = sat_skyfield - self._facility_latlon

        topocentric = difference.at(target_t)
        alt, az, dist = topocentric.altaz()

        return Position(altitude=alt.degrees, azimuth=az.degrees, distance_km=dist.km)

    def _inject_earth_physics(self, target_time_obj: Time, indices):
        """
        Injects cached physical matrices and values from the Master Grid
        to a subset time object to avoid recalculation.
        """
        master = self._grid_timescale

        if "M" not in master.__dict__:
            _ = master.M
        if "gast" not in master.__dict__:
            _ = master.gast
        if "gmst" not in master.__dict__:
            _ = master.gmst
        if "delta_t" not in master.__dict__:
            _ = master.delta_t

        d = master.__dict__
        # (Shape: 3, 3, N)
        target_time_obj.__dict__["M"] = d["M"][:, :, indices]

        # (Shape: N)
        target_time_obj.__dict__["gast"] = d["gast"][indices]
        target_time_obj.__dict__["gmst"] = d["gmst"][indices]
        target_time_obj.__dict__["delta_t"] = d["delta_t"][indices]

    def _calculate_facility_latlon(self):
        return wgs84.latlon(
            latitude_degrees=self._facility.coordinates.latitude,
            longitude_degrees=self._facility.coordinates.longitude,
            elevation_m=self._facility.elevation,
        )
