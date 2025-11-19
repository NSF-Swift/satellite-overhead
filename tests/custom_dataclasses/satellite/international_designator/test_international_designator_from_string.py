from sopp.custom_dataclasses.satellite.international_designator import (
    InternationalDesignator,
)


class TestInternationalDesignatorFromString:
    def test_international_designator_year_is_padded(self):
        arbitrary_single_digit_year = 2
        tle_string_with_arbitrary_other_values = f"0{arbitrary_single_digit_year}111A  "
        designator = InternationalDesignator.from_tle_string(
            tle_string=tle_string_with_arbitrary_other_values
        )
        assert designator.year == arbitrary_single_digit_year

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
