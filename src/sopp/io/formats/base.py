"""Base protocol for trajectory file formats."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from datetime import datetime

    from sopp.models.satellite.trajectory import SatelliteTrajectory


@runtime_checkable
class TrajectoryFormat(Protocol):
    """Protocol defining the interface for trajectory file formats.

    Implementations handle serialization/deserialization of satellite trajectories
    to various file formats (Arrow, CSV, etc.).
    """

    @property
    def extension(self) -> str:
        """Return the file extension for this format (e.g., '.arrow')."""
        ...

    def save(
        self,
        trajectories: SatelliteTrajectory | list[SatelliteTrajectory],
        path: Path,
        *,
        observer_name: str | None = None,
        observer_lat: float | None = None,
        observer_lon: float | None = None,
    ) -> Path:
        """Save trajectory(s) to a file.

        Args:
            trajectories: Single trajectory or list of trajectories to save.
            path: Output file path or directory.
            observer_name: Optional name of the observing facility.
            observer_lat: Optional latitude of the observer.
            observer_lon: Optional longitude of the observer.

        Returns:
            The path to the saved file.
        """
        ...

    def load(
        self,
        path: Path,
        *,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[SatelliteTrajectory]:
        """Load trajectories from a file.

        Args:
            path: Path to the trajectory file.
            time_range: Optional tuple of (start, end) to filter trajectory data.

        Returns:
            List of loaded trajectories.
        """
        ...

    def generate_filename(self, start_time: datetime, end_time: datetime) -> str:
        """Generate a standard filename for a trajectory file.

        Args:
            start_time: Start time of the trajectory.
            end_time: End time of the trajectory.

        Returns:
            Generated filename with appropriate extension.
        """
        ...
