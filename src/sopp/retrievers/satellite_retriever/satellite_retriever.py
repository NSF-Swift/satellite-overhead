from sopp.models.satellite.satellite import Satellite
from sopp.retrievers.retriever import Retriever


class SatelliteRetriever(Retriever):
    def retrieve(self) -> list[Satellite]:
        pass
