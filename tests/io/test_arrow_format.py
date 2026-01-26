"""Tests for Arrow trajectory file format."""

from datetime import datetime, timedelta

import numpy as np
import pytest

from sopp.io.exceptions import (
    TrajectoryFileNotFoundError,
    TrajectoryFormatError,
)
from sopp.io.formats.arrow import ArrowFormat
from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.trajectory import SatelliteTrajectory


@pytest.fixture
def trajectory():
    """Create a sample trajectory for testing."""
    base_time = datetime(2024, 6, 15, 12, 0, 0)
    n_points = 100

    return SatelliteTrajectory(
        satellite=Satellite(name="TEST-SAT-1"),
        times=np.array(
            [base_time + timedelta(seconds=i) for i in range(n_points)],
            dtype=object,
        ),
        azimuth=np.linspace(45.0, 135.0, n_points),
        altitude=np.concatenate(
            [
                np.linspace(0.0, 45.0, n_points // 2),
                np.linspace(45.0, 0.0, n_points - n_points // 2),
            ]
        ),
        distance_km=np.linspace(2000.0, 1500.0, n_points),
    )


@pytest.fixture
def arrow_format():
    """Create an ArrowFormat instance."""
    return ArrowFormat()


class TestArrowFormatSingleTrajectory:
    """Tests for single trajectory save/load operations."""

    def test_save_and_load_roundtrip(self, trajectory, arrow_format, tmp_path):
        """Test that saving and loading a trajectory preserves data."""
        file_path = tmp_path / "test_trajectory.arrow"

        # Save
        saved_path = arrow_format.save(trajectory, file_path)
        assert saved_path == file_path
        assert file_path.exists()

        # Load (returns list)
        loaded_list = arrow_format.load(file_path)
        assert len(loaded_list) == 1
        loaded = loaded_list[0]

        # Verify data
        assert loaded.satellite.name == trajectory.satellite.name
        assert loaded.satellite.tle_information is None
        assert len(loaded) == len(trajectory)
        np.testing.assert_array_almost_equal(loaded.azimuth, trajectory.azimuth)
        np.testing.assert_array_almost_equal(loaded.altitude, trajectory.altitude)
        np.testing.assert_array_almost_equal(
            loaded.distance_km, trajectory.distance_km, decimal=6
        )

    def test_save_to_directory_generates_filename(
        self, trajectory, arrow_format, tmp_path
    ):
        """Test that saving to a directory auto-generates filename."""
        saved_path = arrow_format.save(trajectory, tmp_path)

        assert saved_path.parent == tmp_path
        assert saved_path.suffix == ".arrow"
        assert saved_path.name.startswith("trajectory_")
        assert saved_path.exists()

    def test_save_creates_parent_directories(self, trajectory, arrow_format, tmp_path):
        """Test that saving creates parent directories if needed."""
        file_path = tmp_path / "nested" / "dir" / "test.arrow"

        saved_path = arrow_format.save(trajectory, file_path)
        assert saved_path.exists()

    def test_load_with_observer_metadata(self, trajectory, arrow_format, tmp_path):
        """Test that observer metadata is preserved."""
        file_path = tmp_path / "test.arrow"

        arrow_format.save(
            trajectory,
            file_path,
            observer_name="Green Bank Observatory",
            observer_lat=38.4331,
            observer_lon=-79.8397,
        )

        # Metadata is stored but not exposed in loaded trajectory
        # (it's in the Arrow schema metadata)
        loaded_list = arrow_format.load(file_path)
        assert len(loaded_list) == 1
        assert loaded_list[0].satellite.name == trajectory.satellite.name

    def test_load_nonexistent_file_raises_error(self, arrow_format, tmp_path):
        """Test that loading a nonexistent file raises appropriate error."""
        with pytest.raises(TrajectoryFileNotFoundError):
            arrow_format.load(tmp_path / "nonexistent.arrow")

    def test_load_with_time_range_filter(self, trajectory, arrow_format, tmp_path):
        """Test loading with time range filter."""
        file_path = tmp_path / "test.arrow"
        arrow_format.save(trajectory, file_path)

        # Load middle portion of trajectory
        start = trajectory.times[25]
        end = trajectory.times[75]

        loaded_list = arrow_format.load(file_path, time_range=(start, end))
        assert len(loaded_list) == 1
        loaded = loaded_list[0]

        assert len(loaded) <= 51  # 75 - 25 + 1
        assert all(start <= t <= end for t in loaded.times)


class TestArrowFormatMultipleTrajectories:
    """Tests for multiple trajectory operations."""

    @pytest.fixture
    def multiple_trajectories(self):
        """Create multiple sample trajectories."""
        base_time = datetime(2024, 6, 15, 12, 0, 0)
        trajectories = []

        for i in range(3):
            n_points = 50
            traj = SatelliteTrajectory(
                satellite=Satellite(name=f"SAT-{i}"),
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

    def test_save_and_load_multiple_roundtrip(
        self, multiple_trajectories, arrow_format, tmp_path
    ):
        """Test saving/loading multiple trajectories preserves all data."""
        file_path = tmp_path / "batch.arrow"

        # Save list of trajectories
        arrow_format.save(multiple_trajectories, file_path)
        assert file_path.exists()

        # Load
        loaded = arrow_format.load(file_path)

        assert len(loaded) == len(multiple_trajectories)

        for orig, load in zip(multiple_trajectories, loaded, strict=True):
            assert load.satellite.name == orig.satellite.name
            assert len(load) == len(orig)

    def test_save_empty_list_raises_error(self, arrow_format, tmp_path):
        """Test that saving empty list raises error."""
        with pytest.raises(TrajectoryFormatError):
            arrow_format.save([], tmp_path / "empty.arrow")

    def test_load_with_time_filter(self, multiple_trajectories, arrow_format, tmp_path):
        """Test loading with time range filter."""
        file_path = tmp_path / "batch.arrow"
        arrow_format.save(multiple_trajectories, file_path)

        start = multiple_trajectories[0].times[10]
        end = multiple_trajectories[0].times[30]

        loaded = arrow_format.load(file_path, time_range=(start, end))

        for traj in loaded:
            assert all(start <= t <= end for t in traj.times)

    def test_single_trajectory_in_list(self, trajectory, arrow_format, tmp_path):
        """Test that a list with one trajectory works the same as a single trajectory."""
        file_path1 = tmp_path / "single.arrow"
        file_path2 = tmp_path / "list_of_one.arrow"

        # Save as single
        arrow_format.save(trajectory, file_path1)

        # Save as list of one
        arrow_format.save([trajectory], file_path2)

        # Both should load identically
        loaded1 = arrow_format.load(file_path1)
        loaded2 = arrow_format.load(file_path2)

        assert len(loaded1) == len(loaded2) == 1
        assert loaded1[0].satellite.name == loaded2[0].satellite.name
        np.testing.assert_array_equal(loaded1[0].azimuth, loaded2[0].azimuth)


class TestArrowFormatFilename:
    """Tests for filename generation."""

    def test_generate_filename(self, arrow_format):
        """Test filename generation uses timestamps."""
        start = datetime(2024, 6, 15, 12, 0, 0)
        end = datetime(2024, 6, 15, 13, 0, 0)

        filename = arrow_format.generate_filename(start, end)

        assert filename == "trajectory_2024-06-15T12_00_00_2024-06-15T13_00_00.arrow"


class TestArrowFormatUnitConversion:
    """Tests for unit conversion (km <-> meters)."""

    def test_distance_conversion_roundtrip(self, trajectory, arrow_format, tmp_path):
        """Test that km to meters conversion is accurate."""
        file_path = tmp_path / "test.arrow"

        original_distances = trajectory.distance_km.copy()
        arrow_format.save(trajectory, file_path)
        loaded_list = arrow_format.load(file_path)

        # Should match within floating point tolerance
        np.testing.assert_array_almost_equal(
            loaded_list[0].distance_km, original_distances, decimal=6
        )
