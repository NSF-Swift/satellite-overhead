from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from sopp.config.builder import ConfigurationBuilder
from sopp.config.loaders import ConfigFileLoaderBase
from sopp.models.core import Coordinates
from sopp.models.ground.facility import Facility
from sopp.models.core import FrequencyRange
from sopp.models.core import Position
from sopp.models.reservation import Reservation
from sopp.models.configuration import RuntimeSettings
from sopp.models.satellite.satellite import Satellite
from sopp.models.core import TimeWindow
from sopp.models.ground.config import (
    CelestialTrackingConfig,
    CustomTrajectoryConfig,
    StaticPointingConfig,
)
from sopp.models.ground.trajectory import AntennaTrajectory
from sopp.models.ground.target import ObservationTarget
from sopp.filtering.filterer import Filterer

# --- Helpers & Stubs ---


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


class StubConfigFileLoader(ConfigFileLoaderBase):
    """
    Mock loader that complies with the new Base Class structure.
    """

    @property
    def facility(self):
        return expected_reservation().facility

    @property
    def time_window(self):
        return expected_reservation().time

    @property
    def frequency_range(self):
        return expected_reservation().frequency

    @property
    def runtime_settings(self):
        return RuntimeSettings()

    @property
    def antenna_config(self):
        # Return a dummy tracking config
        return CelestialTrackingConfig(
            target=ObservationTarget(declination="1d", right_ascension="1h")
        )

    @classmethod
    def filename_extension(cls):
        return ".json"


@pytest.fixture
def mock_satellite_loader(monkeypatch):
    def _patch(satellites_to_return: list[Satellite]):
        target_path = "sopp.config.builder.load_satellites"

        def mock_impl(*args, **kwargs):
            return satellites_to_return

        monkeypatch.setattr(target_path, mock_impl)

    return _patch


@pytest.fixture
def test_satellite(satellite) -> Satellite:
    satellite.name = "TestSatellite"
    return satellite


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


def test_set_observation_target_tracking():
    builder = ConfigurationBuilder()
    builder.set_observation_target(
        declination="1d1m1s",
        right_ascension="1h1m1s",
    )

    # Verify it created the correct Variant
    expected_target = ObservationTarget(declination="1d1m1s", right_ascension="1h1m1s")
    assert builder.antenna_config == CelestialTrackingConfig(target=expected_target)


def test_set_observation_target_static():
    builder = ConfigurationBuilder()
    builder.set_observation_target(altitude=1, azimuth=1)

    # Verify it created the correct Variant
    expected_pos = Position(altitude=1, azimuth=1)
    assert builder.antenna_config == StaticPointingConfig(position=expected_pos)


def test_set_observation_target_custom():
    builder = ConfigurationBuilder()
    traj = expected_trajectory()
    builder.set_observation_target(custom_antenna_trajectory=traj)

    # Verify it created the correct Variant
    assert builder.antenna_config == CustomTrajectoryConfig(trajectory=traj)


def test_set_runtime_settings():
    builder = ConfigurationBuilder()
    builder.set_runtime_settings(concurrency_level=1, time_resolution_seconds=1)

    assert builder.runtime_settings == RuntimeSettings(
        concurrency_level=1,
        time_resolution_seconds=1.0,
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


def test_load_satellites(mock_satellite_loader, satellite):
    mock_satellite_loader([satellite])
    builder = ConfigurationBuilder()
    builder.load_satellites("mock/tle")

    assert builder.satellites == [satellite]


def test_set_satellites_filter(test_satellite):
    # This test verifies the filter logic integration
    builder = ConfigurationBuilder()
    builder.satellites = [test_satellite]

    # Filter that removes everything containing "Test"
    filterer = Filterer().add_filter(lambda sat: "Test" not in sat.name)

    builder.set_satellites_filter(filterer)

    # Manually trigger filter since we aren't calling build()
    builder.satellites = builder._filterer.apply_filters(builder.satellites)
    assert builder.satellites == []


def test_build_error_incomplete():
    """Verifies build() fails if required fields are missing."""
    builder = ConfigurationBuilder()
    with pytest.raises(ValueError):
        builder.build()


def test_build_from_config_file(satellite):
    """
    Verifies that set_from_config_file delegates to the Loader
    and populates the Builder's state correctly.
    """
    builder = ConfigurationBuilder()
    # The path is ignored by our Stub
    builder.set_from_config_file(
        config_file=Path("mock/path"),
        loader_class=StubConfigFileLoader,
    )

    # Load satellites (required for build)
    builder.set_satellites([satellite])

    configuration = builder.build()

    # Verify the built configuration matches expected values from the stub
    assert configuration.reservation == expected_reservation()
    assert len(configuration.satellites) == 1

    # The stub returns a CelestialTrackingConfig
    assert isinstance(configuration.antenna_config, CelestialTrackingConfig)


def test_build_full_flow(mock_satellite_loader, satellite):
    """
    Integration test for the whole builder chain.
    """
    mock_satellite_loader([satellite])

    builder = ConfigurationBuilder()
    builder.set_facility(
        latitude=1, longitude=-1, elevation=1, name="HCRO", beamwidth=3
    )
    builder.set_frequency_range(bandwidth=10, frequency=135)
    builder.set_time_window(begin="2023-11-15T08:00:00.0", end="2023-11-15T08:30:00.0")
    builder.set_observation_target(declination="1d1m1s", right_ascension="1h1m1s")
    builder.load_satellites(tle_file="./path/satellites.tle")

    configuration = builder.build()

    assert configuration.reservation == expected_reservation()
    assert len(configuration.satellites) == 1

    # Verify intent was preserved
    assert isinstance(configuration.antenna_config, CelestialTrackingConfig)
