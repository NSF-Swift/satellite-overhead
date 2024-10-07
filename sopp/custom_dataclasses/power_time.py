from dataclasses import dataclass
from datetime import datetime

@dataclass
class PowerTime:
    power: float
    time: datetime
