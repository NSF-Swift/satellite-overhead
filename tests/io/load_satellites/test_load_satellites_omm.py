import csv
import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from sopp.io.omm import _to_satellite
from sopp.io.tle import _attach_frequency_data, load_satellites
from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.tle import InternationalDesignator

TEST_SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
OMM_EXTENSIONS = ["csv", "json", "xml"]


def _fixture(filename: str) -> str:
    return os.path.join(TEST_SCRIPT_DIRECTORY, filename)


class TestOmmLoadersMatchTleLoader:
    """The OMM fixtures are generated from satellites.tle, so every format
    must load to the same satellites the TLE loader produces."""

    def test_all_formats_match_tle(self):
        tle_satellites = load_satellites(_fixture("satellites.tle"))

        for extension in OMM_EXTENSIONS:
            omm_satellites = load_satellites(_fixture(f"satellites.{extension}"))
            assert len(omm_satellites) == 3, extension

            for omm_sat, tle_sat in zip(
                omm_satellites[:2], tle_satellites, strict=True
            ):
                assert omm_sat.name == tle_sat.name, extension
                omm_info = omm_sat.tle_information
                tle_info = tle_sat.tle_information
                assert omm_info is not None and tle_info is not None
                # The OMM epoch round-trips through a timestamp string,
                # which costs float precision below a microsecond.
                assert omm_info.epoch_days == pytest.approx(
                    tle_info.epoch_days, rel=1e-12
                ), extension
                assert replace(omm_info, epoch_days=tle_info.epoch_days) == tle_info, (
                    extension
                )


class TestOmmSixDigitSatellite:
    """Satellites cataloged after July 2026 have 6-digit NORAD IDs and exist
    only in OMM formats; they must load and propagate."""

    def test_six_digit_satellite_loads_and_propagates(self):
        for extension in OMM_EXTENSIONS:
            saramago = load_satellites(_fixture(f"satellites.{extension}"))[-1]

            assert saramago.name == "FAKE SARAMAGO", extension
            assert saramago.satellite_number == 100000, extension
            assert saramago.tle_information is not None
            assert saramago.tle_information.international_designator == (
                InternationalDesignator(year=2026, launch_number=1, launch_piece="A")
            ), extension

            skyfield_satellite = saramago.to_skyfield()
            assert skyfield_satellite.model.satnum == 100000, extension
            position = skyfield_satellite.at(skyfield_satellite.epoch).position.km
            assert np.isfinite(position).all(), extension


class TestOmmFrequencyData:
    def test_frequency_data_attaches_to_omm_satellites(self):
        satellites = load_satellites(
            tle_file=_fixture("satellites.csv"),
            frequency_file=_fixture("satellite_frequencies.csv"),
        )

        assert len(satellites[0].frequency) == 3
        assert len(satellites[1].frequency) == 2
        assert satellites[2].frequency == []

    def test_satellite_without_orbital_data_attaches_no_frequencies(self):
        satellites = _attach_frequency_data(
            [Satellite(name="NO ELEMENTS")],
            Path(_fixture("satellite_frequencies.csv")),
        )

        assert satellites[0].frequency == []


class TestOmmFileFormatOverride:
    def test_explicit_file_format_overrides_extension(self, tmp_path):
        unrecognized = tmp_path / "satellites.omm"
        shutil.copy(_fixture("satellites.json"), unrecognized)

        satellites = load_satellites(unrecognized, file_format="json")

        assert len(satellites) == 3


class TestOmmAnalystObject:
    def test_null_metadata_loads_with_defaults(self):
        with open(_fixture("satellites.csv")) as f:
            fields = dict(next(csv.DictReader(f)))
        fields["OBJECT_NAME"] = None
        fields["OBJECT_ID"] = None
        fields["CLASSIFICATION_TYPE"] = None

        satellite = _to_satellite(fields)

        assert satellite.name == ""
        assert satellite.tle_information is not None
        assert satellite.tle_information.international_designator is None
        assert satellite.tle_information.classification == "U"
