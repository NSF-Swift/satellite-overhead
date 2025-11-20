from datetime import timedelta

from sopp.models.position import Position
from sopp.models.position_time import PositionTime
from sopp.analysis.event_finders.interference import (
    AntennaPosition,
    SatellitesInterferenceFilter,
    SatellitesWithinMainBeamFilter,
)

from tests.analysis.event_finder.definitions import (
    ARBITRARY_ANTENNA_POSITION,
    ARBITRARY_FACILITY,
    assert_windows_eq,
    create_expected_windows,
)


class TestSatellitesWithinMainBeamMultipleAntennas:
    def test_multiple_antenna_positions(self):
        self._run_multiple_positions(
            antenna_positions=self._antenna_positions_sorted_by_time_ascending
        )

    def test_unsorted_antenna_positions(self):
        self._run_multiple_positions(
            antenna_positions=list(
                reversed(self._antenna_positions_sorted_by_time_ascending)
            )
        )

    def _run_multiple_positions(self, antenna_positions: list[PositionTime]):
        cutoff_time = self._antenna_positions_sorted_by_time_ascending[
            -1
        ].time + timedelta(minutes=1)
        slew = SatellitesInterferenceFilter(
            facility=ARBITRARY_FACILITY,
            antenna_positions=[
                AntennaPosition(
                    satellite_positions=[antenna_position],
                    antenna_direction=antenna_position,
                )
                for antenna_position in antenna_positions
            ],
            cutoff_time=cutoff_time,
            filter_strategy=SatellitesWithinMainBeamFilter,
        )
        windows = slew.run()
        expected_positions = [self._antenna_positions_sorted_by_time_ascending[::]]
        expected_windows = create_expected_windows(expected_positions)

        assert len(windows) == 1
        assert_windows_eq(windows, expected_windows)

    @property
    def _antenna_positions_sorted_by_time_ascending(self) -> list[PositionTime]:
        arbitrary_antenna_position2 = PositionTime(
            position=Position(altitude=200, azimuth=200),
            time=ARBITRARY_ANTENNA_POSITION.time + timedelta(minutes=1),
        )
        return [ARBITRARY_ANTENNA_POSITION, arbitrary_antenna_position2]
