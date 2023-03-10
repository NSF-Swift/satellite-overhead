from datetime import datetime
from pathlib import Path
import pytz
import os
import filecmp
from skyfield.api import load, EarthSatellite, Time

from satellite_determination.azimuth_filter.azimuth_filtering import AzimuthFilter
from tests.utilities import get_script_directory
from satellite_determination.custom_dataclasses.coordinates import Coordinates
from satellite_determination.custom_dataclasses.facility import Facility
from satellite_determination.custom_dataclasses.satellite.satellite import Satellite
from satellite_determination.custom_dataclasses.overhead_window import OverheadWindow
from satellite_determination.custom_dataclasses.reservation import Reservation
from satellite_determination.custom_dataclasses.time_window import TimeWindow


class TestAzimuthFilter:
    def test_single_sat(self):
        windows = AzimuthFilter(overhead_windows=[self._arbitrary_overhead_window], reservation=self._arbitrary_reservation).filter_azimuth()
        with open ("azimuth_filtered_windows.txt", "w") as outfile:
            outfile.writelines(str(windows))
            outfile.close()

        assert filecmp.cmp('./tests/event_finder/test_reference_files/azimuth_filter_reference.txt', 'azimuth_filtered_windows.txt') == 1
        os.remove("azimuth_filtered_windows.txt")

    @property
    def _arbitrary_reservation(self) -> Reservation:
        return Reservation(facility=Facility(angle_of_visibility_cone=0,
                                             point_coordinates=Coordinates(latitude=40.8178049, longitude=-121.4695413),
                                             azimuth=30,
                                             name='name'),
                           time=TimeWindow(begin=datetime(year=2023, month=2, day=1, hour=1),
                                           end=datetime(year=2023, month=2, day=1, hour=6)))
    @property
    def _arbitrary_overhead_window(self) -> OverheadWindow:
        return OverheadWindow(satellite=self._arbitrary_satellite[0], overhead_time=TimeWindow(begin=self._arbitrary_date, end=self._arbitrary_date_two))

    @property
    def _arbitrary_satellite(self) -> EarthSatellite:
        tle_file = Path(get_script_directory(__file__), 'TLEdata', 'single_TLE.txt')
        arbitrary_satellite = Satellite.from_tle_file(tle_file)
        return arbitrary_satellite

    @property
    def _arbitrary_date(self) -> Time:
        return datetime(year=2023, month=1, day=1, hour=1, tzinfo=pytz.utc)

    @property
    def _arbitrary_date_two(self) -> datetime:
        return datetime(year=2023, month=1, day=1, hour=3, tzinfo=pytz.utc)




