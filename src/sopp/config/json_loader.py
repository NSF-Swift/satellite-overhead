import json
from functools import cached_property
from pathlib import Path

import numpy as np

from sopp.config.base import ConfigFileLoaderBase
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
from sopp.utils.helpers import read_datetime_string_as_utc


class ConfigFileLoaderJson(ConfigFileLoaderBase):
    @cached_property
    def configuration(self) -> ConfigurationFile:
        return ConfigurationFile(
            reservation=self._reservation,
            runtime_settings=self._runtime_settings,
            antenna_trajectory=self._antenna_trajectory,
            observation_target=self._observation_target,
            static_antenna_position=self._static_antenna_position,
        )

    def _get_required_section(self, key: str) -> dict:
        data = self._config_object.get(key)
        if data is None:
            raise ValueError(
                f"Invalid Configuration: Missing required section '{key}' in {self._filepath}"
            )
        return data

    def _get_required_field(self, data: dict, key: str, section_name: str):
        val = data.get(key)
        if val is None:
            raise ValueError(
                f"Invalid Configuration: Missing field '{key}' in section '{section_name}'"
            )
        return val

    @cached_property
    def _reservation(self) -> Reservation:
        return Reservation(
            facility=self._facility,
            time=self._reservation_window,
            frequency=self._frequency_range,
        )

    @cached_property
    def _facility(self) -> Facility:
        conf = self._get_required_section("facility")

        lat = self._get_required_field(conf, "latitude", "facility")
        lon = self._get_required_field(conf, "longitude", "facility")
        name = self._get_required_field(conf, "name", "facility")

        return Facility(
            coordinates=Coordinates(latitude=lat, longitude=lon),
            name=name,
            elevation=conf.get("elevation", 0.0),
            beamwidth=conf.get("beamwidth", 3.0),
        )

    @cached_property
    def _reservation_window(self) -> TimeWindow:
        conf = self._get_required_section("reservationWindow")

        start_str = self._get_required_field(conf, "startTimeUtc", "reservationWindow")
        end_str = self._get_required_field(conf, "endTimeUtc", "reservationWindow")

        try:
            start = read_datetime_string_as_utc(start_str)
            end = read_datetime_string_as_utc(end_str)
        except Exception as e:
            raise ValueError(f"Invalid Date Format in reservationWindow: {e}") from e

        return TimeWindow(begin=start, end=end)

    @cached_property
    def _frequency_range(self) -> FrequencyRange:
        conf = self._get_required_section("frequencyRange")

        freq = self._get_required_field(conf, "frequency", "frequencyRange")
        bw = self._get_required_field(conf, "bandwidth", "frequencyRange")

        return FrequencyRange(frequency=freq, bandwidth=bw)

    @cached_property
    def _runtime_settings(self) -> RuntimeSettings:
        conf = self._config_object.get("runtimeSettings", {})

        return RuntimeSettings(
            time_resolution_seconds=float(conf.get("time_resolution_seconds", 1.0)),
            concurrency_level=int(conf.get("concurrency_level", 1)),
            min_altitude=float(conf.get("min_altitude", 0.0)),
        )

    @cached_property
    def _antenna_trajectory(self) -> AntennaTrajectory | None:
        data_list = self._config_object.get("antennaPositionTimes")

        if not data_list:
            return None

        times, azimuths, altitudes = [], [], []

        for i, item in enumerate(data_list):
            try:
                t = read_datetime_string_as_utc(item["time"])
                az = float(item["azimuth"])
                alt = float(item["altitude"])
            except KeyError as e:
                raise ValueError(
                    f"Missing field {e} in antennaPositionTimes item #{i}"
                ) from e
            except ValueError as e:
                raise ValueError(
                    f"Invalid number/date format in antennaPositionTimes item #{i}"
                ) from e

            times.append(t)
            azimuths.append(az)
            altitudes.append(alt)

        return AntennaTrajectory(
            times=np.array(times, dtype=object),
            azimuth=np.array(azimuths, dtype=float),
            altitude=np.array(altitudes, dtype=float),
        )

    @cached_property
    def _observation_target(self) -> ObservationTarget | None:
        conf = self._config_object.get("observationTarget")
        if not conf:
            return None

        dec = self._get_required_field(conf, "declination", "observationTarget")
        ra = self._get_required_field(conf, "rightAscension", "observationTarget")

        return ObservationTarget(declination=dec, right_ascension=ra)

    @cached_property
    def _static_antenna_position(self) -> Position | None:
        conf = self._config_object.get("staticAntennaPosition")
        if not conf:
            return None

        alt = self._get_required_field(conf, "altitude", "staticAntennaPosition")
        az = self._get_required_field(conf, "azimuth", "staticAntennaPosition")

        return Position(altitude=alt, azimuth=az)

    @cached_property
    def _config_object(self) -> dict:
        if not Path(self._filepath).exists():
            raise FileNotFoundError(f"Configuration file not found: {self._filepath}")

        with open(self._filepath) as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in configuration file: {e}") from e

    @classmethod
    def filename_extension(cls) -> str:
        return ".json"
