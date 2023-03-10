from datetime import timedelta
from typing import List

import pytz
from skyfield.toposlib import wgs84
from skyfield.api import load

from satellite_determination.azimuth_filter.overhead_window_from_azimuth import OverheadWindowFromAzimuth
from satellite_determination.custom_dataclasses.overhead_window import OverheadWindow
from satellite_determination.custom_dataclasses.reservation import Reservation


class AzimuthFilter:

    def __init__(self, overhead_windows: List[OverheadWindow], reservation: Reservation):
        self._overhead_windows = overhead_windows
        self._reservation = reservation

    def filter_azimuth(self) -> List[OverheadWindow]:
        ts = load.timescale()
        azimuth_filtered_overhead_windows = []
        coordinates = wgs84.latlon(self._reservation.facility.point_coordinates.latitude,
                                   self._reservation.facility.point_coordinates.longitude)
        time_delta = timedelta(seconds=1) #timedelta to check new azimuth, if we move to more granular seconds takes FOREVER to run but get more windows
        for window in self._overhead_windows:
            satellite_azimuth_values = []
            satellite_time_values = []
            rhodesmill_sat = window.satellite.to_rhodesmill()
            difference = rhodesmill_sat - coordinates # get vector sat relative to telescope location
            while window.overhead_time.begin < window.overhead_time.end:  # timestamp for events needs to be changed since when the sat enters cone is not necessarily event timestamp
                topocentric = difference.at(ts.from_datetime(window.overhead_time.begin))
                alt, az, distance = topocentric.altaz()
                satellite_azimuth = min(az.degrees, 360 - az.degrees)
                satellite_azimuth_values.append(satellite_azimuth)
                satellite_time_values.append(window.overhead_time.begin)
                window.overhead_time.begin += time_delta
            azimuth_time_pairs = zip(satellite_azimuth_values, satellite_time_values)
            windows = OverheadWindowFromAzimuth(azimuth_time_pairs, self._reservation, window).get_window_from_azimuth()
            for new_window in windows:
                azimuth_filtered_overhead_windows.append(new_window)
        for new_window in azimuth_filtered_overhead_windows:
            print(new_window.satellite)
            print(new_window.overhead_time.begin.astimezone(pytz.utc))
            print(new_window.overhead_time.end.astimezone(pytz.utc))
        return azimuth_filtered_overhead_windows
