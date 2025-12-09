from datetime import datetime, timezone
from pathlib import Path

import pytest

from sopp.config.base import ConfigFileLoaderBase
from sopp.config.factory import get_config_file_object
from sopp.models.antenna_trajectory import AntennaTrajectory
from sopp.models.configuration_file import ConfigurationFile
from sopp.models.coordinates import Coordinates
from sopp.models.facility import Facility
from sopp.models.frequency_range import FrequencyRange
from sopp.models.observation_target import ObservationTarget
from sopp.models.position import Position
from sopp.models.reservation import Reservation
from sopp.models.runtime_settings import RuntimeSettings
from sopp.models.time_window import TimeWindow
from sopp.utils.helpers import get_script_directory


class TestConfigFileProvidedArgument:
    @pytest.fixture(params=["config_file_json/arbitrary_config_file.json"])
    def config_arbitrary(self, request):
        yield self._get_config_file_object(config_filename=request.param)

    def test_reads_inputs_of_provided_config_file_correctly(self, config_arbitrary):
        assert config_arbitrary.configuration == ConfigurationFile(
            reservation=Reservation(
                facility=Facility(
                    coordinates=Coordinates(
                        latitude=40.8178049, longitude=-121.4695413
                    ),
                    name="ARBITRARY_1",
                    elevation=1000,
                ),
                time=TimeWindow(
                    begin=datetime(
                        year=2023, month=3, day=30, hour=10, tzinfo=timezone.utc
                    ),
                    end=datetime(
                        year=2023, month=3, day=30, hour=11, tzinfo=timezone.utc
                    ),
                ),
                frequency=FrequencyRange(frequency=135, bandwidth=10),
            ),
            antenna_trajectory=None,
            observation_target=ObservationTarget(
                declination="-38d6m50.8s", right_ascension="4h42m"
            ),
            static_antenna_position=Position(altitude=0.2, azimuth=0.3),
        )

    def test_json_allows_antenna_position_times(self):
        config = self._get_config_file_object(
            config_filename="config_file_json/arbitrary_config_file_with_antenna_position_times.json"
        )
        traj = config.configuration.antenna_trajectory

        assert isinstance(traj, AntennaTrajectory)
        assert len(traj) == 2

        assert traj.altitude[0] == 0.0
        assert traj.azimuth[0] == 0.1
        assert traj.times[0] == datetime(
            year=2023, month=3, day=30, hour=10, minute=1, tzinfo=timezone.utc
        )

        assert traj.altitude[1] == 0.1
        assert traj.azimuth[1] == 0.2
        assert traj.times[1] == datetime(
            year=2023, month=3, day=30, hour=10, minute=2, tzinfo=timezone.utc
        )

    def test_json_runtime_settings(self):
        config = self._get_config_file_object(
            config_filename="config_file_json/arbitrary_config_file_runtime_settings.json"
        )
        expected = RuntimeSettings(time_resolution_seconds=5, concurrency_level=6)
        actual = config.configuration.runtime_settings
        assert expected == actual

    @pytest.fixture(
        params=["config_file_json/arbitrary_config_file_no_observation_target.json"]
    )
    def config_no_observation_target(self, request):
        yield self._get_config_file_object(config_filename=request.param)

    def test_observation_target_is_optional(self, config_no_observation_target):
        assert config_no_observation_target.configuration.observation_target is None
        assert config_no_observation_target.configuration.reservation is not None
        assert (
            config_no_observation_target.configuration.static_antenna_position
            is not None
        )

    @pytest.fixture(
        params=[
            "config_file_json/arbitrary_config_file_partial_observation_target.json"
        ]
    )
    def config_partial_observation_target(self, request):
        yield self._get_config_file_object(config_filename=request.param)

    def test_error_is_returned_if_partial_observation_target(
        self, config_partial_observation_target
    ):
        with pytest.raises(ValueError):
            _ = config_partial_observation_target.configuration

    @pytest.fixture(
        params=[
            "config_file_json/arbitrary_config_file_no_static_antenna_position.json"
        ]
    )
    def config_no_static_antenna_position(self, request):
        yield self._get_config_file_object(config_filename=request.param)

    def test_static_antenna_position_is_optional(
        self, config_no_static_antenna_position
    ):
        assert (
            config_no_static_antenna_position.configuration.static_antenna_position
            is None
        )
        assert (
            config_no_static_antenna_position.configuration.observation_target
            is not None
        )
        assert config_no_static_antenna_position.configuration.reservation is not None

    @pytest.fixture(
        params=[
            "config_file_json/arbitrary_config_file_partial_static_antenna_position.json"
        ]
    )
    def config_partial_static_antenna_position(self, request):
        yield self._get_config_file_object(config_filename=request.param)

    def test_error_is_returned_if_partial_static_antenna_position(
        self, config_partial_static_antenna_position
    ):
        with pytest.raises(ValueError):
            _ = config_partial_static_antenna_position.configuration

    @staticmethod
    def _get_config_file_object(config_filename: str) -> ConfigFileLoaderBase:
        return get_config_file_object(
            config_filepath=Path(get_script_directory(__file__), config_filename)
        )
