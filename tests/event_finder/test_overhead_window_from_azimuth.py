from datetime import datetime, timedelta

from satellite_determination.azimuth_filter.overhead_window_from_azimuth import OverheadWindowFromAzimuth
from satellite_determination.custom_dataclasses.overhead_window import OverheadWindow
from satellite_determination.custom_dataclasses.reservation import Reservation
from satellite_determination.custom_dataclasses.time_window import TimeWindow
from satellite_determination.custom_dataclasses.satellite.satellite import Satellite
from satellite_determination.custom_dataclasses.coordinates import Coordinates
from satellite_determination.custom_dataclasses.facility import Facility


class TestOverheadWindowFromAzimuth:
    def test_one_satellite_enters_and_leaves(self):
        azimuth = [30, 34]
        time = [self._arbitrary_date, self._arbitrary_date_two]
        azimuth_time_pair = zip(azimuth, time)
        overhead_windows = OverheadWindowFromAzimuth(azimuth_time_pair, self._arbitrary_reservation_with_nonzero_timewindow, self._arbitrary_overhead_window).get_window_from_azimuth()
        assert overhead_windows == [
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=self._arbitrary_date,
                    end=self._arbitrary_date_two
                )
            )
        ]

    def test_one_satellite_enters_only(self):
        azimuth = [30]
        time = [self._arbitrary_date]
        azimuth_time_pair = zip(azimuth, time)
        overhead_windows = OverheadWindowFromAzimuth(azimuth_time_pair,
                                                     self._arbitrary_reservation_with_nonzero_timewindow,
                                                     self._arbitrary_overhead_window).get_window_from_azimuth()
        assert overhead_windows == [
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=self._arbitrary_date,
                    end=self._arbitrary_reservation_with_nonzero_timewindow.time.end
                )
            )
        ]

    def test_satellite_enters_leaves_twice(self):
        azimuth = [30, 31, 32, 28.5, 30, 32]
        time = [self._arbitrary_date, self._arbitrary_date_two, (self._arbitrary_date_two+timedelta(hours=1)), (self._arbitrary_date_two+timedelta(hours=2)), (self._arbitrary_date_two+timedelta(hours=3)), (self._arbitrary_date_two+timedelta(hours=4)) ]
        azimuth_time_pair = zip(azimuth, time)
        overhead_windows = OverheadWindowFromAzimuth(azimuth_time_pair,
                                                     self._arbitrary_reservation_with_nonzero_timewindow,
                                                     self._arbitrary_overhead_window).get_window_from_azimuth()
        assert overhead_windows == [
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=self._arbitrary_date,
                    end=self._arbitrary_date_two+timedelta(hours=1)
                )
            ),
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=self._arbitrary_date_two+timedelta(hours=2),
                    end=self._arbitrary_date_two + timedelta(hours=4)
                )
            )
        ]

    def test_enters_late(self):
        azimuth = [12, 22, 30, 32, 50, 52]
        time = [self._arbitrary_date, self._arbitrary_date_two, (self._arbitrary_date_two + timedelta(hours=1)),
                (self._arbitrary_date_two + timedelta(hours=2)), (self._arbitrary_date_two + timedelta(hours=3)),
                (self._arbitrary_date_two + timedelta(hours=4))]
        azimuth_time_pair = zip(azimuth, time)
        overhead_windows = OverheadWindowFromAzimuth(azimuth_time_pair,
                                                     self._arbitrary_reservation_with_nonzero_timewindow,
                                                     self._arbitrary_overhead_window).get_window_from_azimuth()
        assert overhead_windows == [
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=self._arbitrary_date_two + timedelta(hours=1),
                    end=self._arbitrary_date_two + timedelta(hours=2)
                )
            )
        ]

    def test_no_satellite_enters(self):
        azimuth = [12, 22, 50, 65, 90, 180]
        time = [self._arbitrary_date, self._arbitrary_date_two, (self._arbitrary_date_two+timedelta(hours=1)), (self._arbitrary_date_two+timedelta(hours=2)), (self._arbitrary_date_two+timedelta(hours=3)), (self._arbitrary_date_two+timedelta(hours=4)) ]
        azimuth_time_pair = zip(azimuth, time)
        overhead_windows = OverheadWindowFromAzimuth(azimuth_time_pair,
                                                     self._arbitrary_reservation_with_nonzero_timewindow,
                                                     self._arbitrary_overhead_window).get_window_from_azimuth()
        assert overhead_windows == []
    @property
    def _arbitrary_satellite(self) -> Satellite:
        return Satellite(name='name')

    @property
    def _arbitrary_reservation_with_nonzero_timewindow(self) -> Reservation:
        return Reservation(facility=Facility(angle_of_visibility_cone=0,
                                             point_coordinates=Coordinates(latitude=0, longitude=0),
                                             azimuth=30,
                                             name='name'),
                           time=TimeWindow(begin=datetime(year=2001, month=2, day=1, hour=1),
                                           end=datetime(year=2001, month=2, day=1, hour=6)))

    @property
    def _arbitrary_reservation_at(self) -> Reservation:
        return Reservation(facility=Facility(angle_of_visibility_cone=0,
                                             point_coordinates=Coordinates(latitude=0, longitude=0),
                                             azimuth=30,
                                             name='name'),
                           time=TimeWindow(begin=datetime(year=2001, month=2, day=1, hour=1),
                                           end=datetime(year=2001, month=2, day=1, hour=6)))

    @property
    def _arbitrary_date(self) -> datetime:
        return datetime(year=2000, month=1, day=1, hour=1)

    @property
    def _arbitrary_date_two(self) -> datetime:
        return datetime(year=2000, month=3, day=3, hour=3)

    @property
    def _arbitrary_overhead_window(self) -> OverheadWindow:
        return OverheadWindow(satellite=self._arbitrary_satellite, overhead_time=TimeWindow(begin=self._arbitrary_date, end=self._arbitrary_date_two))

