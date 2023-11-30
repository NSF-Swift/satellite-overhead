from dataclasses import dataclass, field
from datetime import timedelta

'''
The RuntimeSettings class stores the run time settings used in EventFinderRhodesMill
  + time_continutity_resolution: The time step resolution used to calculate satellite positions. (Default 1 second)
  + concurrency_level: The number of cores to use for multiprocessing the satellite position calculations. (Default 2)
'''


@dataclass
class RuntimeSettings:
    time_continuity_resolution: timedelta = field(default=timedelta(seconds=1))
    concurrency_level: int = field(default=1)