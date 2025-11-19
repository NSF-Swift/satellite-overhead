from dataclasses import replace
from datetime import timedelta
from functools import cached_property

from sopp.models.position_time import PositionTime
from sopp.analysis.event_finders.interference import (
    AntennaPosition,
    SatellitesInterferenceFilter,
    SatellitesWithinMainBeamFilter,
)

from tests.definitions import SMALL_EPSILON
from tests.event_finder.event_finder_rhodesmill.definitions import (
    ARBITRARY_ANTENNA_POSITION,
    ARBITRARY_FACILITY,
    assert_windows_eq,
    create_expected_windows,
)


class TestSatellitesWithinMainBeamOneAntennaPositionMultipleSatellitePositions:
    def test_one_satellite_with_a_few_overhead_windows(self):
        self._run_multiple_positions(
            satellite_positions=self._satellite_positions_by_time_ascending
        )

    def test_unsorted_satellite_positions(self):
        self._run_multiple_positions(
            satellite_positions=list(
                reversed(self._satellite_positions_by_time_ascending)
            )
        )

    def _run_multiple_positions(self, satellite_positions: list[PositionTime]):
        cutoff_time = ARBITRARY_ANTENNA_POSITION.time + timedelta(
            minutes=len(self._satellite_positions_by_time_ascending)
        )
        slew = SatellitesInterferenceFilter(
            facility=ARBITRARY_FACILITY,
            antenna_positions=[
                AntennaPosition(
                    satellite_positions=satellite_positions,
                    antenna_direction=ARBITRARY_ANTENNA_POSITION,
                )
            ],
            cutoff_time=cutoff_time,
            filter_strategy=SatellitesWithinMainBeamFilter,
        )
        windows = slew.run()
        expected_positions = [
            self._satellite_positions_by_time_ascending[0],
            self._satellite_positions_by_time_ascending[2],
            self._satellite_positions_by_time_ascending[4],
        ]
        expected_windows = create_expected_windows(expected_positions)

        assert len(windows) == 3
        assert_windows_eq(windows, expected_windows)

    @cached_property
    def _satellite_positions_by_time_ascending(self) -> list[PositionTime]:
        value_slightly_larger_than_half_beam_width = (
            ARBITRARY_FACILITY.half_beamwidth + SMALL_EPSILON
        )
        out_of_altitude = (
            ARBITRARY_ANTENNA_POSITION.position.altitude
            - value_slightly_larger_than_half_beam_width
        )
        return [
            replace(
                ARBITRARY_ANTENNA_POSITION,
                position=replace(
                    ARBITRARY_ANTENNA_POSITION.position,
                    altitude=out_of_altitude
                    if i % 2
                    else ARBITRARY_ANTENNA_POSITION.position.altitude,
                ),
                time=ARBITRARY_ANTENNA_POSITION.time + timedelta(minutes=i),
            )
            for i in range(5)
        ]
