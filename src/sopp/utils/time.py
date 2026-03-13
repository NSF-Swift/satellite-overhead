"""Time grid generation for simulations."""

from datetime import datetime, timedelta

import numpy as np


def generate_time_grid(
    start: datetime, end: datetime, resolution_seconds: float = 1.0
) -> np.ndarray:
    """Generate a uniformly spaced array of datetime objects.

    Args:
        start: Start time (UTC).
        end: End time (UTC).
        resolution_seconds: Time step in seconds.

    Returns:
        1D numpy array of datetime objects from start to end (inclusive).
    """
    duration = (end - start).total_seconds()

    if duration <= 0:
        return np.array([start], dtype=object)

    steps = int(duration / resolution_seconds) + 1
    offsets = np.arange(steps) * resolution_seconds

    dt_list = [start + timedelta(seconds=x.item()) for x in offsets]

    return np.array(dt_list, dtype=object)
