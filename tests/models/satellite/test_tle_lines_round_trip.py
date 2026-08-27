from dataclasses import replace

from sopp.models.satellite.tle import TleInformation
from tests.models.satellite.utilities import (
    expected_international_space_station_tle_as_satellite_cu,
)


class TestTleLinesRoundTrip:
    def test_five_digit_satellite_number_round_trips(self):
        tle_information = (
            expected_international_space_station_tle_as_satellite_cu().tle_information
        )
        assert tle_information is not None

        line1, line2 = tle_information.to_tle_lines()

        assert line1[2:7] == "25544"
        assert TleInformation.from_tle_lines(line1, line2) == tle_information

    def test_alpha5_satellite_number_round_trips(self):
        """NORAD IDs 100000 to 339999 are spelled with a leading letter in
        TLE lines (Alpha-5); sgp4 encodes and decodes them transparently."""
        tle_information = (
            expected_international_space_station_tle_as_satellite_cu().tle_information
        )
        assert tle_information is not None
        tle_information = replace(tle_information, satellite_number=100000)

        line1, line2 = tle_information.to_tle_lines()

        assert line1[2:7] == "A0000"
        assert line2[2:7] == "A0000"
        assert TleInformation.from_tle_lines(line1, line2) == tle_information
