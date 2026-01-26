"""Tests for SatelliteTrajectory save/load methods."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from sopp.io import save_trajectories
from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.trajectory import SatelliteTrajectory


@pytest.fixture
def trajectory():
    """Create a sample trajectory for testing."""
    base_time = datetime(2024, 6, 15, 12, 0, 0)
    n_points = 100

    return SatelliteTrajectory(
        satellite=Satellite(name="ISS (ZARYA)"),
        times=np.array(
            [base_time + timedelta(seconds=i) for i in range(n_points)],
            dtype=object,
        ),
        azimuth=np.linspace(45.0, 135.0, n_points),
        altitude=np.linspace(10.0, 45.0, n_points),
        distance_km=np.linspace(2000.0, 1500.0, n_points),
    )


class TestSatelliteTrajectoryPersistence:
    """Tests for SatelliteTrajectory.save() and SatelliteTrajectory.load()."""

    def test_save_to_path(self, trajectory, tmp_path):
        """Test saving trajectory to a specific path."""
        file_path = tmp_path / "trajectory.arrow"

        result = trajectory.save(file_path)

        assert result == file_path
        assert file_path.exists()

    def test_save_to_directory(self, trajectory, tmp_path):
        """Test saving trajectory to a directory with auto-generated filename."""
        result = trajectory.save(directory=tmp_path)

        assert result.parent == tmp_path
        assert result.exists()
        assert result.name.startswith("trajectory_")

    def test_save_requires_path_or_directory(self, trajectory):
        """Test that save raises error if neither path nor directory provided."""
        with pytest.raises(ValueError) as exc_info:
            trajectory.save()

        assert "path" in str(exc_info.value).lower()
        assert "directory" in str(exc_info.value).lower()

    def test_load_trajectory(self, trajectory, tmp_path):
        """Test loading a saved trajectory."""
        file_path = tmp_path / "trajectory.arrow"
        trajectory.save(file_path)

        loaded_list = SatelliteTrajectory.load(file_path)

        assert len(loaded_list) == 1
        loaded = loaded_list[0]
        assert loaded.satellite.name == trajectory.satellite.name
        assert len(loaded) == len(trajectory)
        np.testing.assert_array_almost_equal(loaded.azimuth, trajectory.azimuth)
        np.testing.assert_array_almost_equal(loaded.altitude, trajectory.altitude)

    def test_load_with_time_range(self, trajectory, tmp_path):
        """Test loading with time range filter."""
        file_path = tmp_path / "trajectory.arrow"
        trajectory.save(file_path)

        start = trajectory.times[25]
        end = trajectory.times[75]

        loaded_list = SatelliteTrajectory.load(file_path, time_range=(start, end))

        assert len(loaded_list) == 1
        loaded = loaded_list[0]
        assert len(loaded) <= 51
        assert all(start <= t <= end for t in loaded.times)

    def test_save_with_observer_info(self, trajectory, tmp_path):
        """Test saving with observer metadata."""
        file_path = tmp_path / "trajectory.arrow"

        trajectory.save(
            file_path,
            observer_name="Green Bank Observatory",
            observer_lat=38.4331,
            observer_lon=-79.8397,
        )

        # Just verify it saves successfully
        assert file_path.exists()

        # Load and verify data is intact
        loaded_list = SatelliteTrajectory.load(file_path)
        assert loaded_list[0].satellite.name == trajectory.satellite.name


class TestSatelliteTrajectoryBatchOperations:
    """Tests for save_trajectories() and SatelliteTrajectory.load()."""

    @pytest.fixture
    def multiple_trajectories(self):
        """Create multiple test trajectories."""
        trajectories = []
        base_time = datetime(2024, 6, 15, 12, 0, 0)

        for i in range(3):
            n_points = 50
            traj = SatelliteTrajectory(
                satellite=Satellite(name=f"STARLINK-{1000 + i}"),
                times=np.array(
                    [base_time + timedelta(seconds=j) for j in range(n_points)],
                    dtype=object,
                ),
                azimuth=np.linspace(i * 30, i * 30 + 90, n_points),
                altitude=np.linspace(10.0, 45.0, n_points),
                distance_km=np.linspace(2000.0, 1500.0, n_points),
            )
            trajectories.append(traj)

        return trajectories

    def test_save_batch(self, multiple_trajectories, tmp_path):
        """Test saving multiple trajectories as a batch."""
        file_path = tmp_path / "batch.arrow"

        result = save_trajectories(multiple_trajectories, file_path)

        assert result == file_path
        assert file_path.exists()

    def test_load_batch(self, multiple_trajectories, tmp_path):
        """Test loading a batch file."""
        file_path = tmp_path / "batch.arrow"
        save_trajectories(multiple_trajectories, file_path)

        loaded = SatelliteTrajectory.load(file_path)

        assert len(loaded) == len(multiple_trajectories)

        # Check satellite names are preserved
        original_names = {t.satellite.name for t in multiple_trajectories}
        loaded_names = {t.satellite.name for t in loaded}
        assert loaded_names == original_names

    def test_batch_roundtrip_preserves_data(self, multiple_trajectories, tmp_path):
        """Test that batch save/load preserves all trajectory data."""
        file_path = tmp_path / "batch.arrow"
        save_trajectories(multiple_trajectories, file_path)

        loaded = SatelliteTrajectory.load(file_path)

        # Create lookup by name
        loaded_by_name = {t.satellite.name: t for t in loaded}

        for orig in multiple_trajectories:
            load = loaded_by_name[orig.satellite.name]

            assert len(load) == len(orig)
            np.testing.assert_array_almost_equal(load.azimuth, orig.azimuth)
            np.testing.assert_array_almost_equal(load.altitude, orig.altitude)
            np.testing.assert_array_almost_equal(
                load.distance_km, orig.distance_km, decimal=6
            )

    def test_load_batch_with_time_range(self, multiple_trajectories, tmp_path):
        """Test batch loading with time filter."""
        file_path = tmp_path / "batch.arrow"
        save_trajectories(multiple_trajectories, file_path)

        start = multiple_trajectories[0].times[10]
        end = multiple_trajectories[0].times[30]

        loaded = SatelliteTrajectory.load(file_path, time_range=(start, end))

        for traj in loaded:
            assert all(start <= t <= end for t in traj.times)


class TestRSCSimCompatibility:
    """Tests for RSC-SIM file format compatibility."""

    @pytest.fixture
    def rsc_sim_file(self):
        """Path to an actual RSC-SIM trajectory file for testing."""
        file_path = Path(__file__).parents[3] / (
            "RSC-SIM/research_tutorials/data/"
            "Starlink_trajectory_Westford_2025-02-18T15_00_00.000_2025-02-18T15_45_00.000.arrow"
        )
        return file_path

    def test_load_rsc_sim_batch_file(self, rsc_sim_file):
        """Test loading an actual RSC-SIM Arrow file."""
        if not rsc_sim_file.exists():
            pytest.skip(f"RSC-SIM test file not found: {rsc_sim_file}")

        from sopp.io.formats.arrow import ArrowFormat

        fmt = ArrowFormat()

        # RSC-SIM uses different column names - specify them explicitly
        import pyarrow as pa

        with pa.memory_map(str(rsc_sim_file), "r") as source:
            table = pa.ipc.open_file(source).read_all()

        # Find the distance column (varies by observer)
        dist_col = next(
            (c for c in table.column_names if c.startswith("ranges_")),
            "distances",
        )

        trajectories = fmt.load(
            rsc_sim_file,
            time_col="timestamp",
            distance_col=dist_col,
        )

        assert len(trajectories) > 0

        for traj in trajectories:
            assert len(traj) > 0
            assert traj.satellite.name is not None
            assert traj.azimuth.dtype == np.float64
            assert traj.altitude.dtype == np.float64
            assert traj.distance_km.dtype == np.float64
            assert np.all(traj.azimuth >= 0) and np.all(traj.azimuth <= 360)
            assert np.all(traj.altitude >= -90) and np.all(traj.altitude <= 90)
            assert np.all(traj.distance_km > 0)

    def test_rsc_sim_column_mapping(self, rsc_sim_file):
        """Test that RSC-SIM column names are correctly mapped."""
        if not rsc_sim_file.exists():
            pytest.skip(f"RSC-SIM test file not found: {rsc_sim_file}")

        import pyarrow as pa

        # Read raw Arrow file to check column names
        with pa.memory_map(str(rsc_sim_file), "r") as source:
            table = pa.ipc.open_file(source).read_all()

        # RSC-SIM uses these column names
        assert "timestamp" in table.column_names
        assert "azimuths" in table.column_names
        assert "elevations" in table.column_names
        assert any(c.startswith("ranges_") for c in table.column_names)

        # Find the distance column
        dist_col = next(c for c in table.column_names if c.startswith("ranges_"))

        # Load via SOPP with explicit column mapping
        from sopp.io.formats.arrow import ArrowFormat

        trajectories = ArrowFormat().load(
            rsc_sim_file,
            time_col="timestamp",
            distance_col=dist_col,
        )
        assert len(trajectories) > 0
