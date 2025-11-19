from pathlib import Path

from sopp.models.frequency_range import FrequencyRange
from sopp.io.get_frequency_data_from_csv import (
    GetFrequencyDataFromCsv,
)
from sopp.utils.helpers import get_script_directory


class TestFromCsv:
    def test_one_frequency(self):
        frequency_file = Path(
            get_script_directory(__file__), "arbitrary_frequency_file.csv"
        )
        frequency_list = GetFrequencyDataFromCsv(filepath=frequency_file).get()
        frequencies = frequency_list[2023]
        assert frequencies == [
            FrequencyRange(frequency=136.65, bandwidth=None, status="active")
        ]

    def test_two_frequencies(self):
        frequency_file = Path(
            get_script_directory(__file__),
            "arbitrary_frequency_file_two_frequencies.csv",
        )
        frequency_list = GetFrequencyDataFromCsv(filepath=frequency_file).get()
        frequencies = frequency_list[2023]
        assert frequencies == [
            FrequencyRange(frequency=136.65, bandwidth=None, status="active"),
            FrequencyRange(frequency=2, bandwidth=None, status="active"),
        ]

    def test_with_bandwidth(self):
        frequency_file = Path(
            get_script_directory(__file__),
            "arbitrary_frequency_file_with_bandwidth.csv",
        )
        frequency_list = GetFrequencyDataFromCsv(filepath=frequency_file).get()
        frequencies = frequency_list[2023]
        assert frequencies == [
            FrequencyRange(frequency=136.65, bandwidth=None, status="active"),
            FrequencyRange(frequency=2, bandwidth=None, status="active"),
            FrequencyRange(frequency=700, bandwidth=10, status="active"),
        ]

    def test_no_sat_id(self):
        frequency_file = Path(
            get_script_directory(__file__), "arbitrary_frequency_file_none.csv"
        )
        frequency_list = GetFrequencyDataFromCsv(filepath=frequency_file).get()
        frequencies = frequency_list[2023]
        assert len(frequency_list) == 1
        assert frequencies == [
            FrequencyRange(frequency=136.65, bandwidth=None, status="active")
        ]

    def test_junk_data(self):
        frequency_file = Path(
            get_script_directory(__file__), "arbitrary_frequency_file_junk.csv"
        )
        frequency_list = GetFrequencyDataFromCsv(filepath=frequency_file).get()
        frequencies = frequency_list[2023]
        assert len(frequency_list) == 1
        assert frequencies == [
            FrequencyRange(frequency=136.65, bandwidth=12, status="active")
        ]


class TestOverlaps:
    def test_if_overlaps_lower_range(self):
        reservation_frequency = FrequencyRange(frequency=135, bandwidth=10)
        satellite_frequency = FrequencyRange(frequency=127, bandwidth=10)
        overlaps = reservation_frequency.overlaps(satellite_frequency)
        assert overlaps == 1

    def test_if_overlaps_higher_range(self):
        reservation_frequency = FrequencyRange(frequency=135, bandwidth=10)
        satellite_frequency = FrequencyRange(frequency=144, bandwidth=10)
        overlaps = reservation_frequency.overlaps(satellite_frequency)
        assert overlaps == 1

    def test_if_overlaps_all(self):
        reservation_frequency = FrequencyRange(frequency=135, bandwidth=10)
        satellite_frequency = FrequencyRange(frequency=135, bandwidth=50)
        overlaps = reservation_frequency.overlaps(satellite_frequency)
        assert overlaps == 1

    def test_if_overlaps_within_res_frequency(self):
        reservation_frequency = FrequencyRange(frequency=135, bandwidth=10)
        satellite_frequency = FrequencyRange(frequency=137, bandwidth=2)
        overlaps = reservation_frequency.overlaps(satellite_frequency)
        assert overlaps == 1
