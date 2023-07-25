from dataclasses import replace
from datetime import datetime
from functools import cached_property
from typing import List

import pytz

from satellite_determination.custom_dataclasses.time_window import TimeWindow
from satellite_determination.event_finder.event_finder_rhodesmill.support.satellites_within_main_beam_filter import SatellitesWithinMainBeamFilter, \
    AntennaPosition
from tests.definitions import SMALL_EPSILON
from tests.event_finder.event_finder_rhodesmill.definitions import ARBITRARY_ANTENNA_POSITION, ARBITRARY_FACILITY


class TestSatellitesWithinMainBeamAltitude:
    def test_one_satellite_position_below_beamwidth_altitude(self):
        self._run_test(altitude=ARBITRARY_ANTENNA_POSITION.position.altitude - self._value_slightly_larger_than_half_beamwidth,
                       expected_windows=[])

    def test_one_satellite_position_above_beamwidth_altitude(self):
        self._run_test(altitude=ARBITRARY_ANTENNA_POSITION.position.altitude + self._value_slightly_larger_than_half_beamwidth,
                       expected_windows=[TimeWindow(begin=ARBITRARY_ANTENNA_POSITION.time, end=self._arbitrary_cutoff_time)])

    def _run_test(self, altitude: float, expected_windows: List[TimeWindow]) -> None:
        satellite_positions = [
            replace(ARBITRARY_ANTENNA_POSITION,
                    position=replace(ARBITRARY_ANTENNA_POSITION.position, altitude=altitude))
        ]
        slew = SatellitesWithinMainBeamFilter(facility=ARBITRARY_FACILITY,
                                              antenna_positions=[
                                                  AntennaPosition(satellite_positions=satellite_positions,
                                                                  antenna_direction=ARBITRARY_ANTENNA_POSITION)],
                                              cutoff_time=self._arbitrary_cutoff_time)
        windows = slew.run()
        assert windows == expected_windows

    @property
    def _value_slightly_larger_than_half_beamwidth(self) -> float:
        return ARBITRARY_FACILITY.half_beamwidth + SMALL_EPSILON

    @cached_property
    def _arbitrary_cutoff_time(self) -> datetime:
        return datetime.now(tz=pytz.UTC)