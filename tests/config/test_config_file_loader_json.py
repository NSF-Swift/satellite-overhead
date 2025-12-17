from datetime import datetime, timezone
from pathlib import Path

import pytest

from sopp.config.factory import get_config_file_object
from sopp.models.ground.config import (
    CelestialTrackingConfig,
    CustomTrajectoryConfig,
    StaticPointingConfig,
)
from sopp.models.ground.trajectory import AntennaTrajectory
from sopp.models.core import Coordinates
from sopp.models.ground.facility import Facility
from sopp.models.core import FrequencyRange
from sopp.models.ground.target import ObservationTarget
from sopp.models.core import Position
from sopp.models.configuration import RuntimeSettings
from sopp.models.core import TimeWindow


@pytest.fixture
def config_dir():
    """Returns the directory containing the JSON test files."""
    return Path(__file__).resolve().parent / "config_file_json"


@pytest.fixture
def load_config(config_dir):
    """Helper fixture to load a specific file from the test data dir."""

    def _loader(filename: str):
        filepath = config_dir / filename
        return get_config_file_object(config_filepath=filepath)

    return _loader


# --- Tests ---


def test_reads_standard_inputs_correctly(load_config):
    """
    Tests a standard config file ("arbitrary_config_file.json").
    Verifies Facility, Reservation, and prioritization of Static Pointing.
    """
    loader = load_config("arbitrary_config_file.json")

    # 1. Facility
    assert loader.facility == Facility(
        coordinates=Coordinates(latitude=40.8178049, longitude=-121.4695413),
        name="ARBITRARY_1",
        elevation=1000,
    )

    # 2. Time Window
    assert loader.time_window == TimeWindow(
        begin=datetime(2023, 3, 30, 10, 0, tzinfo=timezone.utc),
        end=datetime(2023, 3, 30, 11, 0, tzinfo=timezone.utc),
    )

    # 3. Frequency
    assert loader.frequency_range == FrequencyRange(frequency=135, bandwidth=10)

    # 4. Antenna Config
    # This file has 'staticAntennaPosition' AND 'observationTarget'.
    # Loader logic should prioritize Static.
    assert isinstance(loader.antenna_config, StaticPointingConfig)
    assert loader.antenna_config.position == Position(altitude=0.2, azimuth=0.3)


def test_custom_trajectory_config(load_config):
    """
    Tests a config with 'antennaPositionTimes' (CustomTrajectoryConfig).
    """
    loader = load_config("arbitrary_config_file_with_antenna_position_times.json")

    config = loader.antenna_config
    assert isinstance(config, CustomTrajectoryConfig)

    traj = config.trajectory
    assert isinstance(traj, AntennaTrajectory)
    assert len(traj) == 2

    # Check first point
    assert traj.altitude[0] == 0.0
    assert traj.azimuth[0] == 0.1
    assert traj.times[0] == datetime(2023, 3, 30, 10, 1, tzinfo=timezone.utc)

    # Check second point
    assert traj.altitude[1] == 0.1
    assert traj.azimuth[1] == 0.2


def test_runtime_settings_parsing(load_config):
    """
    Tests parsing of the runtimeSettings section.
    """
    loader = load_config("arbitrary_config_file_runtime_settings.json")

    expected = RuntimeSettings(
        time_resolution_seconds=5.0, concurrency_level=6, min_altitude=0.0
    )
    assert loader.runtime_settings == expected


def test_observation_target_config(load_config):
    """
    Tests a config that ONLY has 'observationTarget' (no static/custom).
    Should result in CelestialTrackingConfig.
    """
    loader = load_config("arbitrary_config_file_no_static_antenna_position.json")

    config = loader.antenna_config
    assert isinstance(config, CelestialTrackingConfig)

    assert config.target == ObservationTarget(
        declination="-38d6m50.8s", right_ascension="4h42m"
    )


def test_validation_partial_observation_target(load_config):
    """
    Tests that a partial observationTarget (missing fields) raises ValueError.
    """
    loader = load_config("arbitrary_config_file_partial_observation_target.json")

    with pytest.raises(ValueError, match="Missing field"):
        _ = loader.antenna_config


def test_validation_partial_static_position(load_config):
    """
    Tests that a partial staticAntennaPosition (missing fields) raises ValueError.
    """
    loader = load_config("arbitrary_config_file_partial_static_antenna_position.json")

    with pytest.raises(ValueError, match="Missing field"):
        _ = loader.antenna_config


def test_validation_no_antenna_info(load_config):
    """
    Tests a config file that has NO antenna info at all.
    """
    loader = load_config("arbitrary_config_file_no_observation_target.json")

    # If it falls through all checks:
    with pytest.raises(ValueError, match="must contain one of"):
        _ = loader.antenna_config
