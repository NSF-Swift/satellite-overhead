from sopp.custom_dataclasses.satellite.international_designator import (
    InternationalDesignator,
)


class TestInternationalDesignatorToString:
    def test_international_designator_year_is_padded(self):
        arbitrary_single_digit_year = 2
        designator = InternationalDesignator(
            year=arbitrary_single_digit_year, launch_number=0, launch_piece=""
        )
        assert designator.to_tle_string()[:2] == "02"

    def test_international_designator_launch_number_is_padded(self):
        arbitrary_launch_number_less_than_three_digits = 2
        designator = InternationalDesignator(
            year=0,
            launch_number=arbitrary_launch_number_less_than_three_digits,
            launch_piece="",
        )
        assert designator.to_tle_string()[2:5] == "002"

    def test_international_designator_piece_is_included(self):
        arbitrary_piece_less_than_three_characters = "B"
        designator = InternationalDesignator(
            year=0,
            launch_number=0,
            launch_piece=arbitrary_piece_less_than_three_characters,
        )
        assert designator.to_tle_string()[5:] == "B"
