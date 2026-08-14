from sopp.models.satellite.tle import (
    InternationalDesignator,
)


class TestInternationalDesignatorFromString:
    def test_international_designator_year_before_pivot_is_2000s(self):
        designator = InternationalDesignator.from_tle_string(tle_string="02111A  ")
        assert designator.year == 2002

    def test_international_designator_year_at_pivot_is_1900s(self):
        designator = InternationalDesignator.from_tle_string(tle_string="57001A  ")
        assert designator.year == 1957

    def test_international_designator_year_after_pivot_is_1900s(self):
        designator = InternationalDesignator.from_tle_string(tle_string="98067A  ")
        assert designator.year == 1998

    def test_international_designator_launch_number_is_padded(self):
        arbitrary_launch_number_less_than_three_digits = 2
        tle_string_with_arbitrary_other_values = (
            f"0100{arbitrary_launch_number_less_than_three_digits}A  "
        )
        designator = InternationalDesignator.from_tle_string(
            tle_string=tle_string_with_arbitrary_other_values
        )
        assert (
            designator.launch_number == arbitrary_launch_number_less_than_three_digits
        )

    def test_international_designator_piece_is_stripped_of_additional_whitespace(self):
        arbitrary_piece_of_size_one_character = "B"
        tle_string_with_arbitrary_other_values = (
            f"01111{arbitrary_piece_of_size_one_character}  "
        )
        designator = InternationalDesignator.from_tle_string(
            tle_string=tle_string_with_arbitrary_other_values
        )
        assert designator.launch_piece == arbitrary_piece_of_size_one_character
