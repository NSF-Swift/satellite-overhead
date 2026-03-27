"""Observation reservation combining facility and time."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sopp.models.core import TimeWindow
    from sopp.models.ground.facility import Facility


@dataclass
class Reservation:
    """A scheduled observation at a facility.

    Attributes:
        facility: The radio astronomy facility.
        time: Time window of the observation.
    """

    facility: Facility
    time: TimeWindow

    def __str__(self):
        return f"{self.__class__.__name__}:\n{self.facility}\n{self.time}"
