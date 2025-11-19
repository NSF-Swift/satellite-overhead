from abc import ABC, abstractmethod

from sopp.models.satellite.satellite import Satellite


class SatellitesLoader(ABC):
    @abstractmethod
    def load_satellites(self) -> list[Satellite]:
        pass
