from dataclasses import dataclass
from enum import Enum
from typing import List
import csv


class FrequencyRangeJsonKey(Enum):
    high_in_megahertz = 'high_in_megahertz'
    low_in_megahertz = 'low_in_megahertz'


@dataclass
class FrequencyRange:
    frequencies: List[tuple]

    @classmethod
    def from_csv(cls, info: dict) -> 'FrequencyList':
        return cls(
            high_in_megahertz=info[FrequencyRangeJsonKey.high_in_megahertz.name],
            low_in_megahertz=info[FrequencyRangeJsonKey.low_in_megahertz.name]
        )
