"""Satellite file loading (TLE and OMM) and remote fetching."""

import os
from dataclasses import replace
from pathlib import Path

import requests

from sopp.io.frequency import GetFrequencyDataFromCsv
from sopp.io.omm import parse_omm_file
from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.tle import TleInformation

NUMBER_OF_LINES_PER_TLE_OBJECT = 3

OMM_FILE_FORMATS = ("csv", "json", "xml")


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
        suffix = tle_path.suffix.lower().lstrip(".")
        file_format = suffix if suffix in OMM_FILE_FORMATS else "tle"

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


def fetch_tles(output_path: Path, source: str = "celestrak") -> Path:
    """
    Downloads TLEs from a remote source and saves them to output_path.
    """
    if source == "celestrak":
        content = _fetch_celestrak()
    elif source == "spacetrack":
        content = _fetch_spacetrack()
    else:
        raise ValueError(f"Unknown TLE source: {source}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(content)

    return output_path


def _fetch_celestrak() -> bytes:
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
    response = requests.get(url=url, allow_redirects=True)
    response.raise_for_status()
    return response.content


def _fetch_spacetrack() -> bytes:
    from dotenv import load_dotenv

    load_dotenv()

    identity = os.getenv("IDENTITY")
    password = os.getenv("PASSWORD")

    if not identity or not password:
        raise ValueError("IDENTITY and PASSWORD env vars required for SpaceTrack")

    url = "https://www.space-track.org/ajaxauth/login"
    query = "https://www.space-track.org/basicspacedata/query/class/gp/decay_date/null-val/epoch/%3Enow-30/orderby/norad_cat_id/format/3le"
    data = {"identity": identity, "password": password, "query": query}

    response = requests.post(url=url, data=data)
    response.raise_for_status()
    return response.content
