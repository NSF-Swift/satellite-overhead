"""Trajectory I/O operations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sopp.io.formats.base import TrajectoryFormat
    from sopp.models.satellite.trajectory import SatelliteTrajectory


def save_trajectories(
    trajectories: list[SatelliteTrajectory],
    path: str | Path,
    *,
    format: TrajectoryFormat | None = None,
    observer_name: str | None = None,
    observer_lat: float | None = None,
    observer_lon: float | None = None,
) -> Path:
    """Save multiple trajectories to a single file.

    Args:
        trajectories: List of trajectories to save.
        path: Output file path.
        format: File format handler. Defaults to ArrowFormat.
        observer_name: Optional name of the observing facility.
        observer_lat: Optional latitude of the observer.
        observer_lon: Optional longitude of the observer.

    Returns:
        Path to the saved file.
    """
    if format is None:
        from sopp.io.formats.arrow import ArrowFormat

        format = ArrowFormat()

    return format.save(
        trajectories,
        Path(path),
        observer_name=observer_name,
        observer_lat=observer_lat,
        observer_lon=observer_lon,
    )
