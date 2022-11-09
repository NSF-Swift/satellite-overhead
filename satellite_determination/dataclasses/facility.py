from dataclasses import dataclass
from enum import Enum

from satellite_determination.dataclasses.coordinates import Coordinates


class FacilityJsonKey(Enum):
    angle_of_visibility = 'angle_of_visibility_cone'
    name = 'name'
    point_coordinates = 'point_coordinates'


@dataclass
class Facility:
    angle_of_visibility_cone: float
    point_coordinates: Coordinates
    name: str

    @classmethod
    def from_json(cls, info: dict) -> 'Facility':
        return Facility(
            angle_of_visibility_cone=info[FacilityJsonKey.angle_of_visibility.value],
            point_coordinates=Coordinates.from_json(info[FacilityJsonKey.point_coordinates.value]),
            name=info[FacilityJsonKey.name.value]
        )
