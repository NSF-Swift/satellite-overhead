"""Observation reservation combining facility, time, and frequency."""

from __future__ import annotations

from dataclasses import dataclass, field
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
        frequency: Frequency band being observed. Optional for horizon-only queries.
    """

    facility: Facility
    time: TimeWindow
    frequency: FrequencyRange | None = field(default=None)

    def __str__(self):
        parts = [
            f"{self.__class__.__name__}:",
            str(self.facility),
            str(self.time),
        ]
        if self.frequency is not None:
            parts.append(str(self.frequency))
        return "\n".join(parts)
