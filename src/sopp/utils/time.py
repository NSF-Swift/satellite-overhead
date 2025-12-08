from datetime import datetime, timedelta

import numpy as np


def generate_time_grid(
    start: datetime, end: datetime, resolution_seconds: float = 1.0
) -> np.ndarray:
    """
    Generates a NumPy array containing Python datetime objects.
    """
    duration = (end - start).total_seconds()

    if duration <= 0:
        return np.array([start], dtype=object)

    steps = int(duration / resolution_seconds) + 1
    offsets = np.arange(steps) * resolution_seconds

    dt_list = [start + timedelta(seconds=x.item()) for x in offsets]

    return np.array(dt_list, dtype=object)
