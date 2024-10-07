from dataclasses import dataclass, field
from operator import attrgetter
from typing import List
from datetime import datetime
from sopp.custom_dataclasses.satellite.satellite import Satellite
from sopp.custom_dataclasses.time_window import TimeWindow
from sopp.custom_dataclasses.power_time import PowerTime  # Assuming PowerTime class is defined similarly

@dataclass
class PowerWindow:
    satellite: Satellite
    powertimes: List[PowerTime] = field(default_factory=list)

    def __post_init__(self):
        self.powertimes.sort(key=attrgetter('time'))

    @property
    def overhead_time(self):
        if not self.powertimes:
            return None
        begin = self.powertimes[0].time
        end = self.powertimes[-1].time
        return TimeWindow(begin=begin, end=end)
