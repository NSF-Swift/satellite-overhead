from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sopp.models.core import Coordinates
from sopp.models.ground.receiver import Receiver

if TYPE_CHECKING:
    pass


@dataclass
class Facility:
    """The Facility data class contains the observation parameters of the
    facility.

    Attributes:
        coordinates: Location of RA facility.
        receiver: Receive-side antenna characteristics. Defaults to Tier 0
            with beamwidth=3.
        elevation: Ground elevation of the telescope in meters. Defaults to 0.
        name: Name of the facility. Defaults to 'Unnamed Facility'.
    """

    coordinates: Coordinates
    receiver: Receiver = field(default_factory=Receiver)
    elevation: float = 0
    name: str | None = "Unnamed Facility"

    def __str__(self):
        lines = [
            f"{self.__class__.__name__}:",
            f"  Name:               {self.name}",
            f"  Latitude:           {self.coordinates.latitude}",
            f"  Longitude:          {self.coordinates.longitude}",
            f"  Elevation:          {self.elevation} meters",
        ]
        lines.append(str(self.receiver))
        return "\n".join(lines)
