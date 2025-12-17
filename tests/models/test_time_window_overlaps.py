from datetime import datetime

from sopp.models.core import TimeWindow


class TestTimeWindowOverlaps:
    def test_second_time_begins_as_first_ends(self):
        time_window1 = TimeWindow(
            begin=datetime(year=2022, month=11, day=20),
            end=datetime(year=2022, month=11, day=21),
        )
        time_window2 = TimeWindow(
            begin=datetime(year=2022, month=11, day=21),
            end=datetime(year=2022, month=11, day=21, hour=1),
        )
        assert not time_window1.overlaps(time_window=time_window2)

    def test_second_time_ends_as_first_begins(self):
        time_window1 = TimeWindow(
            begin=datetime(year=2022, month=11, day=20),
            end=datetime(year=2022, month=11, day=21),
        )
        time_window2 = TimeWindow(
            begin=datetime(year=2022, month=11, day=19),
            end=datetime(year=2022, month=11, day=20),
        )
        assert not time_window1.overlaps(time_window=time_window2)

    def test_time_completely_within_window(self):
        time_window1 = TimeWindow(
            begin=datetime(year=2022, month=11, day=20),
            end=datetime(year=2022, month=11, day=21),
        )
        time_window2 = TimeWindow(
            begin=datetime(year=2022, month=11, day=20, hour=1),
            end=datetime(year=2022, month=11, day=20, hour=2),
        )
        assert time_window1.overlaps(time_window=time_window2)

    def test_time_completely_encases_window(self):
        time_window1 = TimeWindow(
            begin=datetime(year=2022, month=11, day=20),
            end=datetime(year=2022, month=11, day=21),
        )
        time_window2 = TimeWindow(
            begin=datetime(year=2022, month=11, day=19),
            end=datetime(year=2022, month=11, day=22),
        )
        assert time_window1.overlaps(time_window=time_window2)

    def test_time_partially_overlaps_beginning(self):
        time_window1 = TimeWindow(
            begin=datetime(year=2022, month=11, day=20),
            end=datetime(year=2022, month=11, day=21),
        )
        time_window2 = TimeWindow(
            begin=datetime(year=2022, month=11, day=19),
            end=datetime(year=2022, month=11, day=20, hour=2),
        )
        assert time_window1.overlaps(time_window=time_window2)

    def test_time_partially_overlaps_end(self):
        time_window1 = TimeWindow(
            begin=datetime(year=2022, month=11, day=20),
            end=datetime(year=2022, month=11, day=21),
        )
        time_window2 = TimeWindow(
            begin=datetime(year=2022, month=11, day=20, hour=1),
            end=datetime(year=2022, month=11, day=21, hour=1),
        )
        assert time_window1.overlaps(time_window=time_window2)
