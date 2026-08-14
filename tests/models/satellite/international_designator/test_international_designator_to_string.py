from sopp.models.satellite.tle import (
    InternationalDesignator,
)


class TestInternationalDesignatorToString:
    def test_international_designator_year_is_two_digit_and_padded(self):
        designator = InternationalDesignator(
            year=2002, launch_number=0, launch_piece=""
        )
        assert designator.to_tle_string()[:2] == "02"

    def test_international_designator_1900s_year_is_two_digit(self):
        designator = InternationalDesignator(
            year=1998, launch_number=0, launch_piece=""
        )
        assert designator.to_tle_string()[:2] == "98"

    def test_international_designator_launch_number_is_padded(self):
        arbitrary_launch_number_less_than_three_digits = 2
        designator = InternationalDesignator(
            year=2000,
            launch_number=arbitrary_launch_number_less_than_three_digits,
            launch_piece="",
        )
        assert designator.to_tle_string()[2:5] == "002"

    def test_international_designator_piece_is_included(self):
        arbitrary_piece_less_than_three_characters = "B"
        designator = InternationalDesignator(
            year=2000,
            launch_number=0,
            launch_piece=arbitrary_piece_less_than_three_characters,
        )
        assert designator.to_tle_string()[5:] == "B"

    def test_international_designator_round_trips_through_tle_string(self):
        designator = InternationalDesignator(
            year=1998, launch_number=67, launch_piece="A"
        )
        assert (
            InternationalDesignator.from_tle_string(designator.to_tle_string())
            == designator
        )
