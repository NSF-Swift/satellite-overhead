from sopp.custom_dataclasses.coordinates import Coordinates
from sopp.custom_dataclasses.facility import Facility

ARBITRARY_FACILITY = Facility(
    beamwidth=3.0,
    coordinates=Coordinates(latitude=1.0, longitude=2.0),
    elevation=1.0,
    name="name",
)
