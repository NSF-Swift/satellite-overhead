from typing import List

from satellite_determination.custom_dataclasses.overhead_window import OverheadWindow
from satellite_determination.custom_dataclasses.reservation import Reservation
from satellite_determination.custom_dataclasses.time_window import TimeWindow


class OverheadWindowFromAzimuth:

    def __init__(self, azimuth_time_pairs: [List], reservation: Reservation, window: OverheadWindow):
        self._azimuth_time_pairs = azimuth_time_pairs
        self._reservation = reservation
        self._window = window

    def get_window_from_azimuth(self) -> List[OverheadWindow]:
        enter_events = []
        exit_events = []
        sat_in_view_flag = 0
        for azimuth, t in self._azimuth_time_pairs:
            if (azimuth <= (self._reservation.facility.azimuth + 1.5)) and (azimuth >= (self._reservation.facility.azimuth - 1.5)):
                if sat_in_view_flag == 0:
                    enter_events.append(t)
                    sat_in_view_flag = 1
            elif (azimuth > (self._reservation.facility.azimuth + 1.5)) or (azimuth < (self._reservation.facility.azimuth - 1.5)):
                if sat_in_view_flag == 1:
                    exit_events.append(t)
                    sat_in_view_flag = 0
        if enter_events != exit_events:
            exit_events.append(self._reservation.time.end)
        enter_and_exit_pairs = zip(enter_events, exit_events)
        time_windows = [TimeWindow(begin=begin_event, end=exit_event) for begin_event, exit_event in
                        enter_and_exit_pairs]
        overhead_windows = [OverheadWindow(satellite=self._window.satellite, overhead_time=time_window) for time_window in
                                time_windows]
        return overhead_windows
