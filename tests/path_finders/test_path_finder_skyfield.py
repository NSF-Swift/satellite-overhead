import pytest
from path_finder_base_test import PathFinderBaseTest

from sopp.analysis.path_finders.skyfield import (
    ObservationPathFinderSkyfield,
)
from sopp.models.observation_target import ObservationTarget


class TestPathFinderSkyfield(PathFinderBaseTest):
    PathFinderClass = ObservationPathFinderSkyfield

    @pytest.mark.parametrize(
        "declination, right_ascension, expected",
        [
            ("12d15m18s", "12h15m18s", (12, 15, 18)),
            ("12d15m18.5s", "12h15m18.5s", (12, 15, 18.5)),
            ("-38d6m50.8s", "-38h6m50.8s", (-38, 6, 50.8)),
        ],
    )
    def test_ra_dec_to_skyfield(self, declination, right_ascension, expected):
        obs_target = ObservationTarget(
            declination=declination, right_ascension=right_ascension
        )
        actual_ra = ObservationPathFinderSkyfield.right_ascension_to_skyfield(
            obs_target
        )
        actual_dec = ObservationPathFinderSkyfield.declination_to_skyfield(obs_target)

        assert actual_ra == expected
        assert actual_dec == expected
