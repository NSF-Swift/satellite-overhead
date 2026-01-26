from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from sopp.models.core import TimeWindow
from sopp.models.satellite.satellite import Satellite

if TYPE_CHECKING:
    from sopp.io.formats.base import TrajectoryFormat


@dataclass
class SatelliteTrajectory:
    """
    Represents the computed path (trajectory) of a satellite relative to a facility.

    Attributes:
        satellite (Satellite): The satellite object associated with this trajectory.
        times (np.ndarray): 1D array of datetime objects representing time steps.
        azimuth (np.ndarray): 1D array of azimuth angles in degrees.
        altitude (np.ndarray): 1D array of elevation/altitude angles in degrees.
        distance_km (np.ndarray): 1D array of distances to the satellite in kilometers.

    Example:
        # Save a single trajectory
        trajectory.save("path/to/file.arrow")

        # Save multiple trajectories to one file
        from sopp.io import save_trajectories
        save_trajectories(trajectories, "batch.arrow")

        # Load trajectories (always returns a list)
        trajectories = SatelliteTrajectory.load("path/to/file.arrow")
    """

    satellite: Satellite
    times: npt.NDArray[np.object_]
    azimuth: npt.NDArray[np.float64]
    altitude: npt.NDArray[np.float64]
    distance_km: npt.NDArray[np.float64]

    def __len__(self):
        return len(self.times)

    @property
    def overhead_time(self):
        """
        A TimeWindow representing the duration that the satellite is tracked
        within this trajectory (e.g., entering and exiting view).
        Returns None if the trajectory contains no data.
        """
        if len(self.times) == 0:
            return None

        begin = self.times[0]
        end = self.times[-1]
        return TimeWindow(begin=begin, end=end)

    def save(
        self,
        path: str | Path | None = None,
        *,
        directory: str | Path | None = None,
        format: TrajectoryFormat | None = None,
        observer_name: str | None = None,
        observer_lat: float | None = None,
        observer_lon: float | None = None,
    ) -> Path:
        """Save this trajectory to a file.

        Args:
            path: Output file path. If None, directory must be provided.
            directory: Directory for auto-generated filename. Ignored if path is set.
            format: File format handler. Defaults to ArrowFormat.
            observer_name: Optional name of the observing facility.
            observer_lat: Optional latitude of the observer.
            observer_lon: Optional longitude of the observer.

        Returns:
            Path to the saved file.

        Raises:
            ValueError: If neither path nor directory is provided.
        """
        if path is None and directory is None:
            raise ValueError("Either 'path' or 'directory' must be provided")

        if format is None:
            from sopp.io.formats.arrow import ArrowFormat

            format = ArrowFormat()

        if path is not None:
            output_path = Path(path)
        else:
            output_path = Path(directory)

        return format.save(
            self,
            output_path,
            observer_name=observer_name,
            observer_lat=observer_lat,
            observer_lon=observer_lon,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        format: TrajectoryFormat | None = None,
        time_range: tuple[datetime, datetime] | None = None,
    ) -> list[SatelliteTrajectory]:
        """Load trajectories from a file.

        Args:
            path: Path to the trajectory file.
            format: File format handler. Defaults to ArrowFormat.
            time_range: Optional tuple of (start, end) to filter trajectory data.

        Returns:
            List of loaded trajectories (even for single-trajectory files).
        """
        if format is None:
            from sopp.io.formats.arrow import ArrowFormat

            format = ArrowFormat()

        return format.load(Path(path), time_range=time_range)
