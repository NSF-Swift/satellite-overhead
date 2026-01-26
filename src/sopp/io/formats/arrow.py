from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa

from sopp.__about__ import get_version
from sopp.io.exceptions import (
    TrajectoryFileNotFoundError,
    TrajectoryFormatError,
)

if TYPE_CHECKING:
    from sopp.models.satellite.trajectory import SatelliteTrajectory

# Constants for unit conversion
KM_TO_METERS = 1000.0
METERS_TO_KM = 1.0 / KM_TO_METERS


class ArrowFormat:
    """Arrow file format for trajectory persistence.

    Column Schema (RSC-SIM compatible):
        - times: timestamp[us] - UTC timestamps
        - azimuths: float64 - Azimuth angles in degrees
        - elevations: float64 - Elevation angles in degrees
        - distances: float64 - Distances in meters
        - sat: string - Satellite name (always included)

    Metadata Schema (SOPP-specific):
        - satellite_count: Number of satellites in file
        - satellite_catalog_ids: Comma-separated NORAD catalog numbers
        - observer_name: Name of the observing facility
        - observer_lat: Observer latitude in degrees
        - observer_lon: Observer longitude in degrees
        - sopp_version: Version of SOPP that created the file
        - created_at: ISO 8601 timestamp of file creation
    """

    @property
    def extension(self) -> str:
        """Return the file extension for Arrow format."""
        return ".arrow"

    def save(
        self,
        trajectories: SatelliteTrajectory | list[SatelliteTrajectory],
        path: Path,
        *,
        observer_name: str | None = None,
        observer_lat: float | None = None,
        observer_lon: float | None = None,
    ) -> Path:
        """Save trajectory(s) to an Arrow file.

        Args:
            trajectories: Single trajectory or list of trajectories to save.
            path: File path or directory. If directory, generates filename from
                  first trajectory's overhead time.
            observer_name: Optional observer/facility name for metadata.
            observer_lat: Optional observer latitude in degrees.
            observer_lon: Optional observer longitude in degrees.

        Returns:
            Path to the saved file.
        """
        # Normalize to list
        if not isinstance(trajectories, list):
            trajectories = [trajectories]

        if not trajectories:
            raise TrajectoryFormatError("Cannot save empty trajectory list")

        path = Path(path)

        # Generate filename if path is a directory
        if path.is_dir():
            first = trajectories[0]
            overhead = first.overhead_time
            if overhead is None:
                raise TrajectoryFormatError(
                    "Cannot generate filename for empty trajectory"
                )
            filename = self.generate_filename(overhead.begin, overhead.end)
            path = path / filename

        path.parent.mkdir(parents=True, exist_ok=True)

        # Build dataframe with all trajectories
        dfs = []
        catalog_ids = []

        for traj in trajectories:
            df = pd.DataFrame(
                {
                    "times": pd.to_datetime(traj.times, utc=True),
                    "azimuths": traj.azimuth,
                    "elevations": traj.altitude,
                    "distances": traj.distance_km * KM_TO_METERS,
                    "sat": traj.satellite.name,
                }
            )
            dfs.append(df)
            catalog_ids.append(str(traj.satellite.satellite_number or 0))

        combined_df = pd.concat(dfs, ignore_index=True)

        # Build metadata
        metadata = self._build_metadata(
            catalog_ids=catalog_ids,
            observer_name=observer_name,
            observer_lat=observer_lat,
            observer_lon=observer_lon,
        )

        # Write Arrow file
        table = pa.Table.from_pandas(combined_df)
        table = table.replace_schema_metadata(metadata)

        with pa.OSFile(str(path), "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)

        return path

    def load(
        self,
        path: Path,
        *,
        time_range: tuple[datetime, datetime] | None = None,
        time_col: str = "times",
        azimuth_col: str = "azimuths",
        elevation_col: str = "elevations",
        distance_col: str = "distances",
        sat_col: str = "sat",
    ) -> list[SatelliteTrajectory]:
        """Load trajectories from an Arrow file.

        Args:
            path: Path to the Arrow file.
            time_range: Optional (start, end) tuple to filter data points.
            time_col: Column name for timestamps.
            azimuth_col: Column name for azimuth angles.
            elevation_col: Column name for elevation angles.
            distance_col: Column name for distances.
            sat_col: Column name for satellite identifier.

        Returns:
            List of trajectories (may be length 1 for single-satellite files).
        """
        path = Path(path)
        if not path.exists():
            raise TrajectoryFileNotFoundError(f"File not found: {path}")

        table = self._read_arrow_file(path)
        return self._table_to_trajectories(
            table,
            time_range=time_range,
            time_col=time_col,
            azimuth_col=azimuth_col,
            elevation_col=elevation_col,
            distance_col=distance_col,
            sat_col=sat_col,
        )

    def generate_filename(self, start_time: datetime, end_time: datetime) -> str:
        """Generate a standard filename for a trajectory file."""
        start_str = start_time.strftime("%Y-%m-%dT%H_%M_%S")
        end_str = end_time.strftime("%Y-%m-%dT%H_%M_%S")
        return f"trajectory_{start_str}_{end_str}{self.extension}"

    def _build_metadata(
        self,
        catalog_ids: list[str],
        *,
        observer_name: str | None = None,
        observer_lat: float | None = None,
        observer_lon: float | None = None,
    ) -> dict[bytes, bytes]:
        """Build metadata dict for Arrow schema."""
        metadata = {
            b"sopp_version": get_version().encode(),
            b"created_at": datetime.now(timezone.utc).isoformat().encode(),
            b"satellite_count": str(len(catalog_ids)).encode(),
            b"satellite_catalog_ids": ",".join(catalog_ids).encode(),
        }

        if observer_name:
            metadata[b"observer_name"] = observer_name.encode()
        if observer_lat is not None:
            metadata[b"observer_lat"] = str(observer_lat).encode()
        if observer_lon is not None:
            metadata[b"observer_lon"] = str(observer_lon).encode()

        return metadata

    def _read_arrow_file(self, path: Path) -> pa.Table:
        """Read an Arrow file and return the table."""
        try:
            with pa.memory_map(str(path), "r") as source:
                reader = pa.ipc.open_file(source)
                return reader.read_all()
        except pa.ArrowInvalid as e:
            raise TrajectoryFormatError(f"Invalid Arrow file: {e}") from e
        except Exception as e:
            raise TrajectoryFormatError(f"Error reading Arrow file: {e}") from e

    def _table_to_trajectories(
        self,
        table: pa.Table,
        *,
        time_range: tuple[datetime, datetime] | None = None,
        time_col: str,
        azimuth_col: str,
        elevation_col: str,
        distance_col: str,
        sat_col: str,
    ) -> list[SatelliteTrajectory]:
        """Convert an Arrow table to trajectory objects."""
        df = table.to_pandas()

        # Rename columns to canonical names
        df = df.rename(
            columns={
                time_col: "times",
                azimuth_col: "azimuths",
                elevation_col: "elevations",
                distance_col: "distances",
                sat_col: "sat",
            }
        )

        required_cols = {"times", "azimuths", "elevations", "distances"}
        missing = required_cols - set(df.columns)
        if missing:
            raise TrajectoryFormatError(f"Missing required columns: {missing}")

        # Convert distances from meters to km
        df["distances"] = df["distances"] * METERS_TO_KM

        # Normalize timestamps
        df["times"] = pd.to_datetime(df["times"], utc=True)
        df["times"] = df["times"].dt.tz_localize(None)

        # Apply time range filter
        if time_range is not None:
            start, end = time_range
            df = df[(df["times"] >= start) & (df["times"] <= end)].reset_index(
                drop=True
            )

        # Group by satellite name
        if "sat" in df.columns:
            return self._df_to_trajectories(df)
        else:
            # File without sat column - treat as single trajectory
            metadata = table.schema.metadata or {}
            df["sat"] = metadata.get(b"satellite_name", b"Unknown").decode()
            return self._df_to_trajectories(df)

    def _df_to_trajectories(self, df: pd.DataFrame) -> list[SatelliteTrajectory]:
        """Convert a DataFrame with 'sat' column to trajectories."""
        trajectories = []

        for sat_name, group in df.groupby("sat", sort=False):
            traj = self._create_trajectory(
                sat_name=sat_name,
                times=np.array(
                    [t.to_pydatetime() for t in group["times"]], dtype=object
                ),
                azimuths=group["azimuths"].to_numpy(),
                elevations=group["elevations"].to_numpy(),
                distances_km=group["distances"].to_numpy(),
            )
            trajectories.append(traj)

        return trajectories

    def _create_trajectory(
        self,
        sat_name: str,
        times: np.ndarray,
        azimuths: np.ndarray,
        elevations: np.ndarray,
        distances_km: np.ndarray,
    ) -> SatelliteTrajectory:
        """Create a SatelliteTrajectory from loaded data."""
        from sopp.models.satellite.satellite import Satellite
        from sopp.models.satellite.trajectory import SatelliteTrajectory

        return SatelliteTrajectory(
            satellite=Satellite(name=sat_name),
            times=times,
            azimuth=azimuths.astype(np.float64),
            altitude=elevations.astype(np.float64),
            distance_km=distances_km.astype(np.float64),
        )
