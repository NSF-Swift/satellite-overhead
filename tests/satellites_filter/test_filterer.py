from sopp.satellite_selection.filterer import Filterer


class TestFilterer:
    def test_add_filter(self):
        filterer = Filterer().add_filter(lambda x: x % 2)

        assert len(filterer._filters) == 1

    def test_add_none_filter(self):
        values = [1, 2]
        actual = Filterer().add_filter(None).apply_filters(values)

        assert actual == [1, 2]

    def test_no_filters(self):
        values = [1, 2]
        actual = Filterer().apply_filters(values)

        assert actual == [1, 2]

    def test_mix_filters(self):
        values = [1, 12, 16, 40, 42, 43, 44, 45, 47, 48, 50]
        actual = (
            Filterer()
            .add_filter(None)
            .add_filter(lambda x: x % 2 == 0)
            .add_filter(lambda x: x > 40)
            .add_filter(None)
            .add_filter(lambda x: x < 47)
            .apply_filters(values)
        )

        assert actual == [42, 44]

    def test_apply_filter(self):
        values = [1, 12, 16, 40, 42, 43, 44, 45, 47, 48, 50]
        actual = (
            Filterer()
            .add_filter(lambda x: x % 2 == 0)
            .add_filter(lambda x: x > 40)
            .add_filter(lambda x: x < 47)
            .apply_filters(values)
        )

        assert actual == [42, 44]
