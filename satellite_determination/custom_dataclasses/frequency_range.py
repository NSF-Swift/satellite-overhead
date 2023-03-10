from dataclasses import dataclass
from enum import Enum


class FrequencyRangeJsonKey(Enum):
    high_in_megahertz = 'high_in_megahertz'
    low_in_megahertz = 'low_in_megahertz'


@dataclass
class FrequencyRange:
    high_in_megahertz: float
    low_in_megahertz: float

    @classmethod
    def from_json(cls, info: dict) -> 'FrequencyRange':
        return cls(
            high_in_megahertz=info[FrequencyRangeJsonKey.high_in_megahertz.name],
            low_in_megahertz=info[FrequencyRangeJsonKey.low_in_megahertz.name]
        )
