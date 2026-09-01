"""Satellite file loading (TLE and OMM) and remote fetching."""

import os
import warnings
from dataclasses import replace
from pathlib import Path

import requests

from sopp.io.frequency import GetFrequencyDataFromCsv
from sopp.io.omm import parse_omm_file
from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.tle import TleInformation

NUMBER_OF_LINES_PER_TLE_OBJECT = 3

OMM_FILE_FORMATS = ("csv", "json", "xml")
DOWNLOAD_FILE_FORMATS = ("tle", *OMM_FILE_FORMATS)

# Space-Track spells the TLE format "3le" in its query API.
_SPACETRACK_FORMAT_NAMES = {"tle": "3le"}

_TLE_FORMAT_WARNINGS = {
    "celestrak": (
        "Celestrak TLE output omits all satellites with NORAD IDs above 99999 "
        "(everything cataloged since July 2026); use csv, json, or xml for "
        "full coverage."
    ),
    "spacetrack": (
        "Space-Track TLE output uses Alpha-5 spellings for NORAD IDs 100000 "
        "to 339999 and omits objects above 339999; use csv, json, or xml for "
        "full coverage."
    ),
}


def infer_file_format(file: Path | str) -> str:
    """Returns the satellite file format implied by a path's extension.

    Extensions .csv/.json/.xml map to the OMM formats; anything else is
    treated as TLE.
    """
    suffix = Path(file).suffix.lower().lstrip(".")
    return suffix if suffix in OMM_FILE_FORMATS else "tle"


def load_satellites(
    tle_file: Path | str,
    frequency_file: Path | str | None = None,
    file_format: str = "auto",
) -> list[Satellite]:
    """
    Loads satellites from a TLE or OMM file and optionally attaches
    frequency data.

    file_format may be "tle", "csv", "json", or "xml". The default
    "auto" picks the format by file extension; anything without a
    .csv/.json/.xml extension is treated as TLE.
    """
    tle_path = Path(tle_file)

    if file_format == "auto":
        file_format = infer_file_format(tle_path)

    if file_format == "tle":
        satellites = _parse_tle_file(tle_path)
    else:
        satellites = parse_omm_file(tle_path, file_format)

    if frequency_file:
        satellites = _attach_frequency_data(satellites, Path(frequency_file))

    return satellites


def _attach_frequency_data(
    satellites: list[Satellite], frequency_file: Path
) -> list[Satellite]:
    freq_data = GetFrequencyDataFromCsv(filepath=frequency_file).get()

    satellites_with_data = []
    for sat in satellites:
        number = sat.satellite_number
        frequency = freq_data.get(number, []) if number is not None else []
        satellites_with_data.append(replace(sat, frequency=frequency))

    return satellites_with_data


def _parse_tle_file(tlefilepath: Path) -> list[Satellite]:
    with open(tlefilepath) as f:
        lines = f.readlines()

    name_line_indices = range(0, len(lines), NUMBER_OF_LINES_PER_TLE_OBJECT)

    return [
        Satellite(
            name=lines[idx].strip(),
            tle_information=TleInformation.from_tle_lines(
                line1=lines[idx + 1], line2=lines[idx + 2]
            ),
        )
        for idx in name_line_indices
    ]


def fetch_tles(
    output_path: Path, source: str = "celestrak", file_format: str = "csv"
) -> Path:
    """
    Downloads satellite data from a remote source and saves it to output_path.

    file_format may be "csv", "json", "xml", or "tle". The TLE format cannot
    represent the full catalog anymore and emits a warning.
    """
    if file_format not in DOWNLOAD_FILE_FORMATS:
        raise ValueError(f"Unknown download format: {file_format}")

    if file_format == "tle" and source in _TLE_FORMAT_WARNINGS:
        warnings.warn(_TLE_FORMAT_WARNINGS[source], stacklevel=2)

    if source == "celestrak":
        content = _fetch_celestrak(file_format)
    elif source == "spacetrack":
        content = _fetch_spacetrack(file_format)
    else:
        raise ValueError(f"Unknown TLE source: {source}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(content)

    return output_path


def _fetch_celestrak(file_format: str) -> bytes:
    url = (
        f"https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT={file_format}"
    )
    response = requests.get(url=url, allow_redirects=True)
    response.raise_for_status()
    return response.content


def _fetch_spacetrack(file_format: str) -> bytes:
    from dotenv import load_dotenv

    load_dotenv()

    identity = os.getenv("IDENTITY")
    password = os.getenv("PASSWORD")

    if not identity or not password:
        raise ValueError("IDENTITY and PASSWORD env vars required for SpaceTrack")

    spacetrack_format = _SPACETRACK_FORMAT_NAMES.get(file_format, file_format)
    url = "https://www.space-track.org/ajaxauth/login"
    query = (
        "https://www.space-track.org/basicspacedata/query/class/gp/"
        f"decay_date/null-val/epoch/%3Enow-30/orderby/norad_cat_id/format/{spacetrack_format}"
    )
    data = {"identity": identity, "password": password, "query": query}

    response = requests.post(url=url, data=data)
    response.raise_for_status()
    return response.content
