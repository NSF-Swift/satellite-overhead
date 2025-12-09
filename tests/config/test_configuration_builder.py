from datetime import datetime, timezone

import numpy as np
import pytest

from sopp.config.builder import ConfigurationBuilder
from sopp.models import (
    AntennaTrajectory,
    ConfigurationFile,
    Coordinates,
    Facility,
    FrequencyRange,
    ObservationTarget,
    Position,
    Reservation,
    RuntimeSettings,
    Satellite,
    TimeWindow,
)
from sopp.satellite_selection.filterer import Filterer


def expected_trajectory():
    times = np.array([datetime(2023, 11, 15, 8, 0, tzinfo=timezone.utc)], dtype=object)
    return AntennaTrajectory(
        times=times, azimuth=np.array([0.1]), altitude=np.array([0.0])
    )


def expected_reservation():
    return Reservation(
        facility=Facility(
            coordinates=Coordinates(longitude=-1, latitude=1),
            beamwidth=3,
            elevation=1,
            name="HCRO",
        ),
        time=TimeWindow(
            begin=datetime(2023, 11, 15, 8, 0, tzinfo=timezone.utc),
            end=datetime(2023, 11, 15, 8, 30, tzinfo=timezone.utc),
        ),
        frequency=FrequencyRange(
            bandwidth=10,
            frequency=135,
        ),
    )


class StubConfigFileLoader:
    def __init__(self, filepath):
        self.configuration = ConfigurationFile(
            reservation=expected_reservation(),
            observation_target="target",
        )


class StubPathFinder:
    def __init__(self, facility, target, window):
        pass

    def calculate_path(self, resolution_seconds=1.0):
        # Return the vectorized object
        return expected_trajectory()


def mock_satellite_loader(monkeypatch):
    def mock(tle_file, frequency_file=None):
        return [Satellite(name="TestSatellite")]

    monkeypatch.setattr(
        "sopp.io.satellites_loader.SatellitesLoaderFromFiles.load_satellites",
        mock,
    )


# --- Tests ---


def test_set_facility():
    builder = ConfigurationBuilder()
    builder.set_facility(
        latitude=40,
        longitude=-121,
        elevation=100,
        name="HCRO",
        beamwidth=3,
    )
    assert builder.facility == Facility(
        Coordinates(latitude=40, longitude=-121),
        elevation=100,
        beamwidth=3,
        name="HCRO",
    )


def test_set_frequency_range():
    builder = ConfigurationBuilder()
    builder.set_frequency_range(bandwidth=10, frequency=135)
    assert builder.frequency_range == FrequencyRange(
        bandwidth=10,
        frequency=135,
    )


def test_set_observation_target_error():
    builder = ConfigurationBuilder()
    # Providing only altitude without azimuth should raise error
    with pytest.raises(ValueError):
        builder.set_observation_target(altitude=1)


def test_set_observation_target():
    builder = ConfigurationBuilder()
    builder.set_observation_target(
        declination="1d1m1s",
        right_ascension="1h1m1s",
    )
    # Check internal state matches
    assert builder._observation_target == ObservationTarget(
        declination="1d1m1s",
        right_ascension="1h1m1s",
    )


def test_set_observation_target_static():
    builder = ConfigurationBuilder()
    builder.set_observation_target(altitude=1, azimuth=1)

    assert builder._static_pointing == Position(
        altitude=1,
        azimuth=1,
    )


def test_set_observation_target_custom():
    builder = ConfigurationBuilder()
    traj = expected_trajectory()
    builder.set_observation_target(custom_antenna_trajectory=traj)

    assert builder._custom_antenna_trajectory == traj


def test_set_runtime_settings():
    builder = ConfigurationBuilder()
    builder.set_runtime_settings(concurrency_level=1, time_resolution_seconds=1)

    assert builder.runtime_settings == RuntimeSettings(
        concurrency_level=1,
        time_resolution_seconds=1,
    )


def test_set_time_window_str():
    builder = ConfigurationBuilder()
    builder.set_time_window(
        begin="2023-11-15T08:00:00.0",
        end="2023-11-15T08:30:00.0",
    )

    assert builder.time_window == TimeWindow(
        begin=datetime(2023, 11, 15, 8, 0, tzinfo=timezone.utc),
        end=datetime(2023, 11, 15, 8, 30, tzinfo=timezone.utc),
    )


def test_set_time_window_datetime():
    builder = ConfigurationBuilder()
    builder.set_time_window(
        begin=datetime(2023, 11, 15, 8, 0, tzinfo=timezone.utc),
        end=datetime(2023, 11, 15, 8, 30, tzinfo=timezone.utc),
    )

    assert builder.time_window == TimeWindow(
        begin=datetime(2023, 11, 15, 8, 0, tzinfo=timezone.utc),
        end=datetime(2023, 11, 15, 8, 30, tzinfo=timezone.utc),
    )


def test_set_satellites(monkeypatch):
    mock_satellite_loader(monkeypatch)
    builder = ConfigurationBuilder()
    builder.set_satellites("/mock/tle", "mock/frequency")

    assert builder.satellites == [Satellite(name="TestSatellite")]


def test_set_satellites_filter(monkeypatch):
    # This test verifies the filter logic integration
    mock_satellite_loader(monkeypatch)
    builder = ConfigurationBuilder()
    builder.satellites = [Satellite(name="TestSatellite")]

    # Filter that removes everything containing "Test"
    filterer = Filterer().add_filter(lambda sat: "Test" not in sat.name)

    builder.set_satellites_filter(filterer)

    # Manually trigger filter since we aren't calling build()
    builder.satellites = builder._filterer.apply_filters(builder.satellites)
    assert builder.satellites == []


def test_add_satellites_filter(monkeypatch):
    mock_satellite_loader(monkeypatch)
    builder = ConfigurationBuilder()
    builder.satellites = [Satellite(name="TestSatellite")]

    builder.add_filter(lambda sat: "Test" not in sat.name)
    builder.satellites = builder._filterer.apply_filters(builder.satellites)

    assert builder.satellites == []


def test_build_antenna_trajectory_target():
    builder = ConfigurationBuilder(path_finder_class=StubPathFinder)
    # Set required time window for trajectory generation
    builder.set_time_window(
        begin="2023-11-15T08:00:00.0",
        end="2023-11-15T08:30:00.0",
    )
    # Set dummy facility required for path finding
    builder.set_facility(40, -120, 0, "Test", 3)

    # Set target to trigger the path finder
    builder._observation_target = "mock"

    traj = builder._build_antenna_trajectory()
    expected = expected_trajectory()

    # Compare vector objects (using custom equality logic or just length/properties)
    assert len(traj) == len(expected)
    assert traj.times[0] == expected.times[0]


def test_build_antenna_trajectory_static():
    builder = ConfigurationBuilder()
    # Manually set internal state to simulate "Static Pointing"
    builder._static_pointing = Position(altitude=90, azimuth=0)
    builder.set_time_window(
        begin="2023-11-15T08:00:00.0",
        end="2023-11-15T08:30:00.0",
    )

    traj = builder._build_antenna_trajectory()

    # Should generate a vector covering the window (30 mins = 1801 points @ 1s)
    assert len(traj) == 1801
    assert traj.altitude[0] == 90


def test_build_antenna_trajectory_custom():
    builder = ConfigurationBuilder()
    builder.set_time_window(
        begin="2023-11-15T08:00:00.0",
        end="2023-11-15T08:30:00.0",
    )
    custom = expected_trajectory()
    builder._custom_antenna_trajectory = custom

    traj = builder._build_antenna_trajectory()
    assert traj == custom


def test_build_error_incomplete():
    builder = ConfigurationBuilder()
    with pytest.raises(ValueError):
        builder.build()


def test_build_from_config_file(monkeypatch):
    mock_satellite_loader(monkeypatch)

    builder = ConfigurationBuilder(
        config_file_loader_class=StubConfigFileLoader,
        path_finder_class=StubPathFinder,
    )
    builder.set_from_config_file(config_file="mock/path")
    # Need to load satellites as file loader usually doesn't do TLE I/O
    builder.set_satellites(tle_file="./path/satellites.tle")

    configuration = builder.build()

    # Verify the built configuration matches expected values
    assert configuration.reservation == expected_reservation()
    assert len(configuration.satellites) == 1
    # Check trajectory was built
    assert len(configuration.antenna_trajectory) > 0


def test_build_full_flow(monkeypatch):
    mock_satellite_loader(monkeypatch)

    builder = ConfigurationBuilder(path_finder_class=StubPathFinder)
    builder.set_facility(
        latitude=1,
        longitude=-1,
        elevation=1,
        name="HCRO",
        beamwidth=3,
    )
    builder.set_frequency_range(bandwidth=10, frequency=135)
    builder.set_time_window(begin="2023-11-15T08:00:00.0", end="2023-11-15T08:30:00.0")
    builder.set_observation_target(declination="1d1m1s", right_ascension="1h1m1s")
    builder.set_satellites(tle_file="./path/satellites.tle")

    configuration = builder.build()

    assert configuration.reservation == expected_reservation()
    assert len(configuration.satellites) == 1
    assert len(configuration.antenna_trajectory) > 0
