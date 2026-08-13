from dataclasses import replace

import numpy as np
import pytest
from skyfield.api import EarthSatellite

from tests.models.satellite.utilities import (
    expected_international_space_station_tle_as_satellite_cu,
)

EXACT_MODEL_ATTRIBUTES = ["satnum", "classification", "intldesg", "revnum"]
FLOAT_MODEL_ATTRIBUTES = [
    "argpo",
    "bstar",
    "ecco",
    "inclo",
    "mo",
    "ndot",
    "nddot",
    "no_kozai",
    "nodeo",
]
SIX_DIGIT_SATELLITE_NUMBER = 100000


class TestSatelliteToSkyfield:
    def test_satellite_can_translate_to_skyfield(self):
        self.given_a_cu_satellite_with_international_space_station_properties()
        self.given_a_skyfield_satellite_loaded_from_the_international_space_station_tle()
        self.when_the_cu_satellite_is_converted_into_skyfield()
        self.then_the_satellites_should_match()

    def given_a_cu_satellite_with_international_space_station_properties(self) -> None:
        self._cu_satellite = expected_international_space_station_tle_as_satellite_cu()

    def given_a_skyfield_satellite_loaded_from_the_international_space_station_tle(
        self,
    ) -> None:
        line1 = "1 25544U 98067A   19343.69339541  .00001764  00000-0  38792-4 0    04"
        line2 = "2 25544  51.6439 211.2001 0007417  17.6667  85.6398 15.50103472202482"
        self._skyfield_satellite = EarthSatellite(line1, line2, "FAKE ISS (ZARYA)")

    def when_the_cu_satellite_is_converted_into_skyfield(self) -> None:
        self._converted_satellite = self._cu_satellite.to_skyfield()

    def then_the_satellites_should_match(self) -> None:
        assert self._converted_satellite.name == self._skyfield_satellite.name
        assert self._models_match()
        assert self._epochs_match()
        assert self._propagated_positions_match()

    def _models_match(self) -> bool:
        built = self._converted_satellite.model
        expected = self._skyfield_satellite.model

        exact_match = all(
            getattr(built, attribute) == getattr(expected, attribute)
            for attribute in EXACT_MODEL_ATTRIBUTES
        )
        float_match = all(
            getattr(built, attribute)
            == pytest.approx(getattr(expected, attribute), rel=1e-12)
            for attribute in FLOAT_MODEL_ATTRIBUTES
        )
        return exact_match and float_match

    def _epochs_match(self) -> bool:
        built = self._converted_satellite.model
        expected = self._skyfield_satellite.model

        built_epoch = built.jdsatepoch + built.jdsatepochF
        expected_epoch = expected.jdsatepoch + expected.jdsatepochF
        return built_epoch == pytest.approx(expected_epoch, abs=1e-8)

    def _propagated_positions_match(self) -> bool:
        epoch = self._skyfield_satellite.epoch
        times = epoch.ts.tt_jd(epoch.tt + np.linspace(0, 1, 5))

        built_positions = self._converted_satellite.at(times).position.km
        expected_positions = self._skyfield_satellite.at(times).position.km
        return np.allclose(built_positions, expected_positions, rtol=0, atol=1e-3)


class TestSatelliteToSkyfieldSixDigitNoradId:
    def test_six_digit_norad_id_can_translate_to_skyfield(self):
        """NORAD IDs above 99999 no longer fit the TLE format; conversion
        to Skyfield must not round-trip through TLE lines."""
        satellite = expected_international_space_station_tle_as_satellite_cu()
        tle_information = satellite.tle_information
        assert tle_information is not None
        satellite = replace(
            satellite,
            tle_information=replace(
                tle_information, satellite_number=SIX_DIGIT_SATELLITE_NUMBER
            ),
        )

        skyfield_satellite = satellite.to_skyfield()

        assert skyfield_satellite.model.satnum == SIX_DIGIT_SATELLITE_NUMBER
        geocentric = skyfield_satellite.at(skyfield_satellite.epoch)
        assert np.isfinite(geocentric.position.km).all()
