from abc import ABC, abstractmethod
from typing import List

from satellite_determination.custom_dataclasses.overhead_window import OverheadWindow
from satellite_determination.custom_dataclasses.reservation import Reservation
from satellite_determination.custom_dataclasses.satellite.satellite import Satellite


class Validator(ABC):
    @abstractmethod
    def overhead_list(self, list_of_satellites: List[Satellite], reservation: Reservation) -> List[OverheadWindow]:
        pass
