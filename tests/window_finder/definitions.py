from satellite_determination.dataclasses.coordinates import Coordinates
from satellite_determination.dataclasses.facility import Facility

ARBITRARY_FACILITY = Facility(
    angle_of_visibility_cone=1.,
    point_coordinates=Coordinates(latitude=1., longitude=2.),
    name='name'
)