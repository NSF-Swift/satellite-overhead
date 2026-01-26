"""I/O utilities for SOPP.

This module provides functionality for:
- Loading satellites from TLE files
- Fetching TLEs from remote sources
- Saving and loading trajectory data in various formats
"""

from sopp.io.exceptions import (
    TrajectoryFileNotFoundError,
    TrajectoryFormatError,
    TrajectoryIOError,
    TrajectoryMetadataError,
    TrajectoryValidationError,
)
from sopp.io.formats import ArrowFormat, TrajectoryFormat
from sopp.io.tle import fetch_tles, load_satellites
from sopp.io.trajectory import save_trajectories

__all__ = [
    # TLE operations
    "load_satellites",
    "fetch_tles",
    # Trajectory I/O
    "save_trajectories",
    "TrajectoryFormat",
    "ArrowFormat",
    # Exceptions
    "TrajectoryIOError",
    "TrajectoryFileNotFoundError",
    "TrajectoryFormatError",
    "TrajectoryValidationError",
    "TrajectoryMetadataError",
]
