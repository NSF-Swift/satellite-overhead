from dataclasses import dataclass, field

from satellite_determination.custom_dataclasses.facility import Facility
from satellite_determination.custom_dataclasses.frequency_range.frequency_range import FrequencyRange
from satellite_determination.custom_dataclasses.time_window import TimeWindow

'''
The Reservation class stores the Facility, as well as some additional reservation-specific information, such as reservation start and end times.
  + facility:   Facility object with RA facility and observation parameters
  + time:       TimeWindow that represents the start and end time of the ideal reservation.
  + frequency:  FrequencyRange of the requested observation. This is the frequency that the RA telescope wants to observe at.
'''


@dataclass
class Reservation:
    facility: Facility
    time: TimeWindow
    frequency: FrequencyRange = field(default_factory=FrequencyRange)
