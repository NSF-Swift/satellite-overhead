from abc import ABC, abstractmethod
from pathlib import Path

from sopp.models.antenna_config import AntennaConfig
from sopp.models.facility import Facility
from sopp.models.frequency_range import FrequencyRange
from sopp.models.runtime_settings import RuntimeSettings
from sopp.models.time_window import TimeWindow


class ConfigFileLoaderBase(ABC):
    def __init__(self, filepath: Path):
        self._filepath = filepath

    @property
    @abstractmethod
    def facility(self) -> Facility:
        pass

    @property
    @abstractmethod
    def time_window(self) -> TimeWindow:
        pass

    @property
    @abstractmethod
    def frequency_range(self) -> FrequencyRange:
        pass

    @property
    @abstractmethod
    def runtime_settings(self) -> RuntimeSettings:
        pass

    @property
    @abstractmethod
    def antenna_config(self) -> AntennaConfig:
        pass

    @classmethod
    @abstractmethod
    def filename_extension(cls) -> str:
        pass
