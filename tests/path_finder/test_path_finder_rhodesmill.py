import pytest
from path_finder_base_test import PathFinderBaseTest
from sopp.models.observation_target import ObservationTarget
from sopp.path_finder.observation_path_finder_rhodesmill import (
    ObservationPathFinderRhodesmill,
)


class TestPathFinderRhodesMill(PathFinderBaseTest):
    PathFinderClass = ObservationPathFinderRhodesmill

    @pytest.mark.parametrize(
        "declination, right_ascension, expected",
        [
            ("12d15m18s", "12h15m18s", (12, 15, 18)),
            ("12d15m18.5s", "12h15m18.5s", (12, 15, 18.5)),
            ("-38d6m50.8s", "-38h6m50.8s", (-38, 6, 50.8)),
        ],
    )
    def test_ra_dec_to_rhodesmill(self, declination, right_ascension, expected):
        obs_target = ObservationTarget(
            declination=declination, right_ascension=right_ascension
        )
        actual_ra = ObservationPathFinderRhodesmill.right_ascension_to_rhodesmill(
            obs_target
        )
        actual_dec = ObservationPathFinderRhodesmill.declination_to_rhodesmill(
            obs_target
        )

        assert actual_ra == expected
        assert actual_dec == expected
