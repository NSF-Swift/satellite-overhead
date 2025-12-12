from dataclasses import replace
from pathlib import Path

from sopp.io.get_frequency_data_from_csv import (
    GetFrequencyDataFromCsv,
)
from sopp.models.satellite.satellite import Satellite
from tests.models.satellite.utilities import (
    expected_international_space_station_tle_as_satellite_cu,
)

TEST_DIR = Path(__file__).resolve().parent


class TestTleToSatelliteCu:
    def test_single_satellite(self):
        tle_file = TEST_DIR / "international_space_station_tle.tle"
        frequency_file = TEST_DIR / "fake_ISS_frequency_file.csv"
        satellite = Satellite.from_tle_file(tlefilepath=tle_file)
        frequency_list = GetFrequencyDataFromCsv(filepath=frequency_file).get()
        satellite_list_with_frequencies = [
            replace(
                satellite,
                frequency=frequency_list.get(
                    satellite.tle_information.satellite_number, []
                ),
            )
            for satellite in satellite
        ]
        assert satellite_list_with_frequencies == [self._expected_satellite_first]

    def test_multiple_satellites(self):
        tle_file = TEST_DIR / "international_space_station_tle_multiple.tle"
        frequency_file = TEST_DIR / "fake_ISS_frequency_file_multiple.csv"
        satellite = Satellite.from_tle_file(tlefilepath=tle_file)
        frequency_list = GetFrequencyDataFromCsv(filepath=frequency_file).get()
        satellite_list_with_frequencies = [
            replace(
                satellite,
                frequency=frequency_list.get(
                    satellite.tle_information.satellite_number, []
                ),
            )
            for satellite in satellite
        ]
        assert satellite_list_with_frequencies == [
            self._expected_satellite_first,
            self._expected_satellite_second,
        ]

    @property
    def _expected_satellite_second(self) -> Satellite:
        satellite = self._expected_satellite_first
        satellite.name = "FAKE ISS (ZARYA) 2"
        satellite.tle_information.international_designator.launch_piece = "AB"
        satellite.tle_information.satellite_number = 25545
        satellite.frequency[0].frequency = 200
        satellite.frequency[0].bandwidth = 10
        return satellite

    @property
    def _expected_satellite_first(self) -> Satellite:
        return expected_international_space_station_tle_as_satellite_cu()
