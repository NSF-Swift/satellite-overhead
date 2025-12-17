from dataclasses import dataclass
from functools import cached_property

from sopp.models.core import Coordinates


@dataclass
class Facility:
    """
    The Facility data class contains the observation parameters of the facility and the object it is tracking, including coordinates
    of the RA telescope and its beamwidth, as well as the right ascension and declination values for its observation target:

    -coordinates:       location of RA facility. Coordinates.
    -beamwidth:         beamwidth of the telescope. float. Defaults to 3
    -elevation:         ground elevation of the telescope in meters. float. Defaults to 0
    -name:              name of the facility. String. Defaults to 'Unnamed Facility'
    """

    coordinates: Coordinates
    beamwidth: float = 3
    elevation: float = 0
    name: str | None = "Unnamed Facility"

    @cached_property
    def beam_radius(self) -> float:
        return self.beamwidth / 2.0

    def __str__(self):
        return (
            f"{self.__class__.__name__}:\n"
            f"  Name:               {self.name}\n"
            f"  Latitude:           {self.coordinates.latitude}\n"
            f"  Longitude:          {self.coordinates.longitude}\n"
            f"  Elevation:          {self.elevation} meters\n"
            f"  Beamwidth:          {self.beamwidth} degrees"
        )
