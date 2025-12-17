import os
from dataclasses import replace
from pathlib import Path

import requests

from sopp.io.frequency import GetFrequencyDataFromCsv
from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.tle import TleInformation

NUMBER_OF_LINES_PER_TLE_OBJECT = 3


def load_satellites(
    tle_file: Path | str, frequency_file: Path | str | None = None
) -> list[Satellite]:
    """
    Loads TLEs from disk and optionally attaches frequency data.
    """
    tle_path = Path(tle_file)
    freq_path = Path(frequency_file) if frequency_file else None

    satellites = _parse_tle_file(tle_path)

    if freq_path:
        freq_data = GetFrequencyDataFromCsv(filepath=freq_path).get()

        satellites = [
            replace(
                sat, frequency=freq_data.get(sat.tle_information.satellite_number, [])
            )
            for sat in satellites
        ]

    return satellites


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
