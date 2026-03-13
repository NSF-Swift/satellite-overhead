"""Date/time and file parsing utilities."""

import json
from datetime import datetime, timezone
from pathlib import Path

from dateutil import parser


def read_json_file(filepath: Path) -> dict:
    """Read and parse a JSON file."""
    with open(filepath) as f:
        return json.load(f)


def convert_datetime_to_utc(localtime: datetime) -> datetime:
    """Ensure a datetime is timezone-aware UTC.

    Naive datetimes are assumed to be UTC. Timezone-aware datetimes
    are converted to UTC.
    """
    if localtime.tzinfo == timezone.utc:
        return localtime
    elif localtime.tzinfo is None:
        return localtime.replace(tzinfo=timezone.utc)
    else:
        return localtime.astimezone(timezone.utc)


def read_datetime_string_as_utc(string_value: str) -> datetime:
    """Parse an ISO 8601 string and return a UTC datetime.

    Raises:
        ValueError: If the string cannot be parsed.
    """
    try:
        time = parser.parse(string_value)
        return convert_datetime_to_utc(time)
    except ValueError:
        raise ValueError(f"Unable to parse datetime string: {string_value}") from None


def parse_time_and_convert_to_utc(time: str | datetime) -> datetime:
    """Accept either a string or datetime and return UTC datetime."""
    try:
        return read_datetime_string_as_utc(time)
    except TypeError:
        return convert_datetime_to_utc(time)
