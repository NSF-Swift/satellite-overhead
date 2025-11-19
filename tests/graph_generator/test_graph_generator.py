from datetime import datetime

import pytest
from numpy import asarray
from numpy.ma import allequal
from sopp.custom_dataclasses.overhead_window import OverheadWindow
from sopp.custom_dataclasses.position import Position
from sopp.custom_dataclasses.position_time import PositionTime
from sopp.graph_generator import graph_generator
from sopp.graph_generator.graph_generator import GraphGenerator


class TestGraphGenerator:
    @pytest.fixture(autouse=True)
    def _set_mock_pyplot(self, mocker):
        self._mock_pyplot = mocker.patch.object(graph_generator, "plt")

    def test_generate_graph(self):
        GraphGenerator(
            interference_windows=self._arbitrary_overhead_window_two,
            satellites_above_horizon=self._arbitrary_overhead_window,
            search_window_start=self._arbitrary_search_window_start,
            search_window_end=self._arbitrary_search_window_end,
        ).generate_graph()
        self._assert_correct_satellites_above_horizon_calls()
        self._assert_correct_satellites_in_main_beam_calls()
        self._assert_correct_label_calls()
        self._mock_pyplot.show.assert_called()

    @property
    def _arbitrary_overhead_window_two(self) -> list[OverheadWindow]:
        return [
            OverheadWindow(
                satellite=None,
                positions=[
                    PositionTime(
                        Position(0, 0),
                        time=datetime(year=2023, month=4, day=19, hour=1),
                    ),
                    PositionTime(
                        Position(0, 0),
                        time=datetime(year=2023, month=4, day=19, hour=2),
                    ),
                ],
            )
        ]

    @property
    def _arbitrary_overhead_window(self) -> list[OverheadWindow]:
        return [
            OverheadWindow(
                satellite=None,
                positions=[
                    PositionTime(
                        Position(0, 0),
                        time=datetime(year=2023, month=4, day=19, hour=1),
                    ),
                    PositionTime(
                        Position(0, 0),
                        time=datetime(year=2023, month=4, day=19, hour=2),
                    ),
                ],
            )
        ]

    @property
    def _arbitrary_search_window_start(self) -> datetime:
        return datetime(year=2023, month=4, day=18, hour=23)

    @property
    def _arbitrary_search_window_end(self) -> datetime:
        return datetime(year=2023, month=4, day=19, hour=7)

    def _assert_correct_satellites_above_horizon_calls(self) -> None:
        expected_satellites_above_horizon_bar_positions = (
            asarray([0, 1, 2, 3, 4, 5, 6, 7]) - 0.2
        )
        expected_bar_width = 0.4
        assert allequal(
            self._mock_pyplot.bar.call_args_list[0].args[0],
            expected_satellites_above_horizon_bar_positions,
        )
        assert allequal(
            self._mock_pyplot.bar.call_args_list[0].args[1],
            self._expected_number_of_satellites_above_horizon,
        )
        assert self._mock_pyplot.bar.call_args_list[0].args[2] == expected_bar_width
        assert self._mock_pyplot.bar.call_args_list[0].kwargs == {
            "label": "Satellites above the horizon"
        }

    @property
    def _expected_number_of_satellites_above_horizon(self) -> list[int]:
        return [0, 0, 1, 0, 0, 0, 0, 0]

    def _assert_correct_satellites_in_main_beam_calls(self) -> None:
        expected_satellites_in_main_beam_bar_positions = (
            asarray([0, 1, 2, 3, 4, 5, 6, 7]) + 0.2
        )
        expected_bar_width = 0.4
        assert allequal(
            self._mock_pyplot.bar.call_args_list[1].args[0],
            expected_satellites_in_main_beam_bar_positions,
        )
        assert allequal(
            self._mock_pyplot.bar.call_args_list[1].args[1],
            self._expected_number_of_satellites_in_main_beam,
        )
        assert self._mock_pyplot.bar.call_args_list[1].args[2] == expected_bar_width
        assert self._mock_pyplot.bar.call_args_list[1].kwargs == {
            "label": "Satellites crossing main observation beam"
        }

    @property
    def _expected_number_of_satellites_in_main_beam(self) -> list[int]:
        return [0, 0, 1, 0, 0, 0, 0, 0]

    def _assert_correct_label_calls(self) -> None:
        self._mock_pyplot.xlabel.assert_called_with("Hour (UTC)")
        self._mock_pyplot.ylabel.assert_called_with("Number of satellites")
        self._mock_pyplot.title.assert_called_with(
            "Overhead Satellites at HCRO starting at 2023-04-18 23:00:00"
        )
        self._mock_pyplot.legend.assert_called()
        self._assert_correct_xtick_calls()

    def _assert_correct_xtick_calls(self) -> None:
        assert allequal(
            self._mock_pyplot.xticks.call_args.args[0], [0, 1, 2, 3, 4, 5, 6, 7]
        )
        assert allequal(
            self._mock_pyplot.xticks.call_args.args[1],
            ["23", "0", "1", "2", "3", "4", "5", "6"],
        )
