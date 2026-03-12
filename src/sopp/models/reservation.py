"""Observation reservation combining facility, time, and frequency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sopp.models.core import FrequencyRange, TimeWindow
    from sopp.models.ground.facility import Facility


@dataclass
class Reservation:
    """A scheduled observation at a facility.

    Attributes:
        facility: The radio astronomy facility.
        time: Time window of the observation.
        frequency: Frequency band being observed.
    """

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
