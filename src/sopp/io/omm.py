"""OMM (Orbit Mean-Elements Message) file loading.

OMM is the successor to the TLE format, served by Celestrak and
Space-Track as CSV, JSON, and XML. It carries the same SGP4 mean elements
as a TLE but has no fixed-width fields.
"""

import json
from pathlib import Path

from sgp4 import omm as sgp4_omm
from sgp4.api import Satrec

from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.tle import TleInformation

# sgp4's omm.initialize() assumes these fields are present and non-null.
# Analyst objects can lack them, so fill TLE-style defaults before parsing.
OPTIONAL_FIELD_DEFAULTS = {
    "OBJECT_NAME": "",
    "OBJECT_ID": "",
    "CLASSIFICATION_TYPE": "U",
    "EPHEMERIS_TYPE": "0",
    "ELEMENT_SET_NO": "0",
    "REV_AT_EPOCH": "0",
}


def parse_omm_file(omm_file: Path | str, file_format: str) -> list[Satellite]:
    """
    Loads satellites from an OMM file in csv, json, or xml format.
    """
    path = Path(omm_file)

    if file_format == "csv":
        with open(path, newline="") as f:
            return [_to_satellite(fields) for fields in sgp4_omm.parse_csv(f)]
    if file_format == "json":
        with open(path) as f:
            records = json.load(f)
        return [_to_satellite(fields) for fields in records]
    if file_format == "xml":
        with open(path) as f:
            return [_to_satellite(fields) for fields in sgp4_omm.parse_xml(f)]

    raise ValueError(f"Unknown OMM file format: {file_format}")


def _to_satellite(fields: dict) -> Satellite:
    fields = {**fields}
    for key, default in OPTIONAL_FIELD_DEFAULTS.items():
        if fields.get(key) in (None, ""):
            fields[key] = default

    satrec = Satrec()
    sgp4_omm.initialize(satrec, fields)

    return Satellite(
        name=fields["OBJECT_NAME"].strip(),
        tle_information=TleInformation.from_satrec(satrec),
    )
