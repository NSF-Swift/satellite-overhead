import pickle
import re
import types
from pathlib import Path

from skyfield.api import load
from sopp.utils.helpers import get_script_directory

from tests.models.satellite.utilities import (
    expected_international_space_station_tle_as_satellite_cu,
)


class TestSatelliteToRhodesMill:
    def test_satellite_can_translate_to_rhodesmill(self):
        self.given_a_cu_satellite_with_international_space_station_properties()
        self.given_a_rhodesmill_satellite_loaded_from_the_international_space_station_tle()
        self.when_the_cu_satellite_is_converted_into_rhodesmill()
        self.then_the_satellites_should_match()

    def given_a_cu_satellite_with_international_space_station_properties(self) -> None:
        self._cu_satellite = expected_international_space_station_tle_as_satellite_cu()

    def given_a_rhodesmill_satellite_loaded_from_the_international_space_station_tle(
        self,
    ) -> None:
        tle_file = Path(
            get_script_directory(__file__), "international_space_station_tle.tle"
        )
        self._rhodesmill_satellite = load.tle_file(url=str(tle_file))[0]

    def when_the_cu_satellite_is_converted_into_rhodesmill(self) -> None:
        self._converted_satellite = self._cu_satellite.to_rhodesmill()

    def then_the_satellites_should_match(self) -> None:
        assert self._models_match() and self._non_model_properties_match()

    def _models_match(self) -> bool:
        for attribute in dir(self._converted_satellite.model):
            built_satellite_value = getattr(self._converted_satellite.model, attribute)
            should_test_attribute = not re.compile("^_").match(attribute) and type(
                built_satellite_value
            ) not in [types.BuiltinMethodType, types.MethodType]
            if should_test_attribute and built_satellite_value != getattr(
                self._rhodesmill_satellite.model, attribute
            ):
                return False
        return True

    def _non_model_properties_match(self) -> bool:
        built_satellite_model = self._converted_satellite.model
        expected_satellite_model = self._rhodesmill_satellite.model
        self._converted_satellite.model = None
        self._rhodesmill_satellite.model = None

        is_match = pickle.dumps(self._converted_satellite) == pickle.dumps(
            self._rhodesmill_satellite
        )

        self._converted_satellite.model = built_satellite_model
        self._rhodesmill_satellite.model = expected_satellite_model

        return is_match
