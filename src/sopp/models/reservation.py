from dataclasses import dataclass

from sopp.models.core import FrequencyRange, TimeWindow
from sopp.models.ground.facility import Facility

"""
The Reservation class stores the Facility, as well as some additional reservation-specific information, such as reservation start and end times.
  + facility:   Facility object with RA facility and observation parameters
  + time:       TimeWindow that represents the start and end time of the ideal reservation.
  + frequency:  FrequencyRange of the requested observation. This is the frequency that the RA telescope wants to observe at.
"""


@dataclass
class Reservation:
    facility: Facility
    time: TimeWindow
    frequency: FrequencyRange

    def __str__(self):
        return (
            f"{self.__class__.__name__}:\n"
            f"{self.facility}\n"
            f"{self.time}\n"
            f"{self.frequency}"
        )
