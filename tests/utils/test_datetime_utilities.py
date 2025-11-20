from datetime import datetime, timezone

import pytest
from sopp.utils.helpers import (
    parse_time_and_convert_to_utc,
    read_datetime_string_as_utc,
)


class TestDatetimeUtilities:
    def test_read_datetime_string_as_utc_with_microseconds(self):
        expected_datetime = datetime(2023, 12, 27, 19, 0, 0, 0, tzinfo=timezone.utc)

        assert read_datetime_string_as_utc("2023-12-27T19:00:00.0") == expected_datetime

    def test_read_datetime_string_as_utc_without_microseconds(self):
        expected_datetime = datetime(2023, 12, 27, 19, 0, 0, 0, tzinfo=timezone.utc)

        assert read_datetime_string_as_utc("2023-12-27T19:00:00") == expected_datetime

    def test_read_datetime_string_as_utc_with_timezone(self):
        expected_datetime = datetime(2023, 12, 27, 12, 0, 0, 0, tzinfo=timezone.utc)

        assert (
            read_datetime_string_as_utc("2023-12-27T5:00:00-07:00") == expected_datetime
        )

    def test_read_datetime_with_invalid_string(self):
        with pytest.raises(ValueError) as _:
            read_datetime_string_as_utc("20223-12-27-T5:00:00")

    def test_read_datetime_with_datetime_raises_type_error(self):
        with pytest.raises(TypeError) as _:
            read_datetime_string_as_utc(
                datetime(2023, 12, 27, 12, 0, 0, 0, tzinfo=timezone.utc)
            )

    def test_parse_time_and_convert_to_utc_with_datetime(self):
        expected_datetime = datetime(2023, 12, 27, 12, 0, 0, 0, tzinfo=timezone.utc)

        assert parse_time_and_convert_to_utc(expected_datetime) == expected_datetime

    def test_parse_time_and_convert_to_utc_with_string(self):
        expected_datetime = datetime(2023, 12, 27, 12, 0, 0, 0, tzinfo=timezone.utc)

        assert (
            parse_time_and_convert_to_utc("2023-12-27T5:00:00-07:00")
            == expected_datetime
        )

    def test_parse_time_and_convert_to_utc_with_invalid_string(self):
        with pytest.raises(ValueError) as _:
            parse_time_and_convert_to_utc("20223-12-27-T5:00:00")
