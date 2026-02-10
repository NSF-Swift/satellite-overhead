from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

from sopp.models.core import Coordinates

if TYPE_CHECKING:
    from sopp.models.antenna import AntennaPattern


@dataclass
class Facility:
    """The Facility data class contains the observation parameters of the
    facility and the object it is tracking.

    Supports two modes for antenna gain:

    Tier 1 (simple): Set only `peak_gain_dbi` for worst-case constant gain.
    Tier 2 (detailed): Set `antenna_pattern` for angle-dependent gain lookup.

    Attributes:
        coordinates: Location of RA facility.
        beamwidth: Beamwidth of the telescope in degrees. Defaults to 3.
        elevation: Ground elevation of the telescope in meters. Defaults to 0.
        name: Name of the facility. Defaults to 'Unnamed Facility'.
        peak_gain_dbi: Peak antenna gain in dBi (Tier 1). Defaults to None.
        antenna_pattern: Full antenna gain pattern (Tier 2). Defaults to None.
    """

    coordinates: Coordinates
    beamwidth: float = 3
    elevation: float = 0
    name: str | None = "Unnamed Facility"
    peak_gain_dbi: float | None = None
    antenna_pattern: AntennaPattern | None = None

    @cached_property
    def beam_radius(self) -> float:
        return self.beamwidth / 2.0

    def __str__(self):
        lines = [
            f"{self.__class__.__name__}:",
            f"  Name:               {self.name}",
            f"  Latitude:           {self.coordinates.latitude}",
            f"  Longitude:          {self.coordinates.longitude}",
            f"  Elevation:          {self.elevation} meters",
            f"  Beamwidth:          {self.beamwidth} degrees",
        ]
        if self.antenna_pattern is not None:
            lines.append(
                f"  Peak Gain:          {self.antenna_pattern.peak_gain_dbi} dBi (from pattern)"
            )
        elif self.peak_gain_dbi is not None:
            lines.append(f"  Peak Gain:          {self.peak_gain_dbi} dBi")
        return "\n".join(lines)
