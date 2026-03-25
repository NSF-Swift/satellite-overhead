"""Plotting utilities for satellite trajectories."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sopp.models.satellite.trajectory_set import TrajectorySet


def plot_trajectories(trajectory_set: TrajectorySet):
    """Plot elevation vs time for a set of satellite trajectories.

    Args:
        trajectory_set: Trajectories to plot.
    """
    from datetime import datetime, timezone

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 5))
    for t in trajectory_set:
        ax.plot(t.times, t.altitude, label=t.satellite.name)
        if t.peak_time is not None:
            ax.plot(t.peak_time, t.peak_elevation, "k.", ms=4)

    ax.axvline(datetime.now(timezone.utc), color="g", ls="--", alpha=0.5, label="now")
    ax.set_xlabel("UTC")
    ax.set_ylabel("Elevation (deg)")
    ax.grid(True, alpha=0.3)
    if len(trajectory_set) <= 15:
        ax.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.show()
