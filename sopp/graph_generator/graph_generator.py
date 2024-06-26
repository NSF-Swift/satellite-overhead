from datetime import datetime, timedelta
from functools import cached_property
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from sopp.custom_dataclasses.overhead_window import OverheadWindow
from sopp.custom_dataclasses.time_window import TimeWindow


class GraphGenerator:
    """
    The GraphGenerator generates a bar graph showing when satellites cross the main beam and
    how many total satellites are above the horizon during the search window. It requires the
    search window and a list of satellites that will interfere with the main observation beam
    (generated by the EventFinderRhodesmill.interference_windows) and the satellites
    visible above the horizon (generated by EventFinderRhodesmill.satellites_above_horizon).
    """

    def __init__(self,
                 search_window_start: datetime,
                 search_window_end: datetime,
                 satellites_above_horizon: List[OverheadWindow],
                 interference_windows: List[OverheadWindow]):
        self._satellite_above_horizon = satellites_above_horizon
        self._interference_windows = interference_windows
        self._search_window_start = search_window_start
        self._search_window_end = search_window_end

    def generate_graph(self):
        size_x_axis = np.arange(len(self._x_axis))
        plt.bar(size_x_axis - 0.2, self._number_of_satellites_above_horizon, 0.4, label='Satellites above the horizon')
        plt.bar(size_x_axis + 0.2, self._number_of_satellites_in_main_beam, 0.4, label='Satellites crossing main observation beam')
        plt.xlabel('Hour (UTC)')
        plt.ylabel('Number of satellites')
        title = 'Overhead Satellites at HCRO starting at ' + str(self._search_window_start)
        plt.title(title)
        plt.legend()
        plt.xticks(size_x_axis, self._x_axis)
        plt.show()

    @cached_property
    def _number_of_satellites_above_horizon(self) -> List[int]:
        return self._get_satellites_in_interval_window(overhead_windows=self._satellite_above_horizon)

    @cached_property
    def _number_of_satellites_in_main_beam(self) -> List[int]:
        return self._get_satellites_in_interval_window(overhead_windows=self._interference_windows)

    def _get_satellites_in_interval_window(self, overhead_windows: List[OverheadWindow]) -> List[int]:
        return [len([window for window in overhead_windows if window.overhead_time.overlaps(interval_window)])
                for interval_window in self._interval_windows]

    @cached_property
    def _interval_windows(self) -> List[TimeWindow]:
        return [TimeWindow(
            begin=interval,
            end=interval + timedelta(minutes=59)
        ) for interval in self._search_intervals]

    @cached_property
    def _x_axis(self) -> List[str]:
        return [str(interval.hour) for interval in self._search_intervals]

    @cached_property
    def _search_intervals(self) -> List[datetime]:
        search_intervals = [self._search_window_start]
        while True:
            next_interval = search_intervals[-1] + timedelta(hours=1)
            if next_interval >= self._search_window_end:
                break
            search_intervals.append(next_interval)
        return search_intervals
