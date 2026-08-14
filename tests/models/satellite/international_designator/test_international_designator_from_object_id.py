from sopp.models.satellite.tle import (
    InternationalDesignator,
)


class TestInternationalDesignatorFromObjectId:
    def test_object_id_parses_losslessly(self):
        designator = InternationalDesignator.from_object_id("1998-067A")
        assert designator == InternationalDesignator(
            year=1998, launch_number=67, launch_piece="A"
        )

    def test_object_id_with_multi_letter_piece(self):
        designator = InternationalDesignator.from_object_id("2023-054AL")
        assert designator == InternationalDesignator(
            year=2023, launch_number=54, launch_piece="AL"
        )

    def test_object_id_matches_windowed_tle_string(self):
        from_object_id = InternationalDesignator.from_object_id("1998-067A")
        from_tle = InternationalDesignator.from_tle_string("98067A")
        assert from_object_id == from_tle
