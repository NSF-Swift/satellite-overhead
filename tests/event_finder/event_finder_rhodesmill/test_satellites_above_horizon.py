from dataclasses import replace
from datetime import timedelta, datetime

import pytz

from sopp.custom_dataclasses.time_window import TimeWindow
from sopp.custom_dataclasses.position import Position
from sopp.custom_dataclasses.position_time import PositionTime
from sopp.event_finder.event_finder_rhodesmill.support.satellites_interference_filter import SatellitesAboveHorizonFilter, \
    AntennaPosition, SatellitesInterferenceFilter, SatellitesWithinMainBeamFilter
from tests.event_finder.event_finder_rhodesmill.definitions import ARBITRARY_ANTENNA_POSITION, ARBITRARY_FACILITY, create_expected_windows, assert_windows_eq, ARBITRARY_SATELLITE_POSITION


class TestSatellitesAboveHorizon:
    def test_satellite_is_above_horizon(self):
        satellite_above_horizon = self.satellite_above_horizon()
        satellite_positions = [
            satellite_above_horizon
        ]

        cutoff_time = satellite_above_horizon.time + timedelta(minutes=1)

        slew = SatellitesInterferenceFilter(
            facility=ARBITRARY_FACILITY,
            antenna_positions= [
                    AntennaPosition(satellite_positions=satellite_positions, antenna_direction=ARBITRARY_ANTENNA_POSITION)
                ],
                cutoff_time=cutoff_time,
                filter_strategy=SatellitesAboveHorizonFilter
        )

        windows = slew.run()
        expected_positions = satellite_positions
        expected_windows = create_expected_windows(expected_positions)

        assert len(windows) == 1
        assert_windows_eq(windows, expected_windows)

    def test_satellite_is_below_horizon(self):
        satellite_below_horizon = self.satellite_below_horizon()

        satellite_positions = [
            satellite_below_horizon
        ]

        cutoff_time = satellite_below_horizon.time + timedelta(minutes=1)

        slew = SatellitesInterferenceFilter(
            facility=ARBITRARY_FACILITY,
            antenna_positions= [
                    AntennaPosition(satellite_positions=satellite_positions, antenna_direction=ARBITRARY_ANTENNA_POSITION)
                ],
                cutoff_time=cutoff_time,
                filter_strategy=SatellitesAboveHorizonFilter
        )

        windows = slew.run()
        expected_positions = []
        expected_windows = create_expected_windows(expected_positions)

        assert len(windows) == 0
        assert_windows_eq(windows, expected_windows)

    def test_satellite_is_below_horizon_then_above(self):
        satellite_below_horizon = self.satellite_below_horizon()
        satellite_above_horizon = self.satellite_above_horizon(time_offset=1)
        satellite_positions = [
            satellite_below_horizon,
            satellite_above_horizon,
        ]

        cutoff_time = satellite_above_horizon.time + timedelta(minutes=1)

        slew = SatellitesInterferenceFilter(
            facility=ARBITRARY_FACILITY,
            antenna_positions= [
                    AntennaPosition(satellite_positions=satellite_positions, antenna_direction=ARBITRARY_ANTENNA_POSITION)
                ],
                cutoff_time=cutoff_time,
                filter_strategy=SatellitesAboveHorizonFilter
        )

        windows = slew.run()
        expected_positions = [satellite_above_horizon]
        expected_windows = create_expected_windows(expected_positions)

        assert len(windows) == 1
        assert_windows_eq(windows, expected_windows)

    def test_satellite_is_above_horizon_then_below(self):
        satellite_above_horizon = self.satellite_above_horizon()
        satellite_below_horizon = self.satellite_below_horizon(time_offset=1)
        satellite_positions = [
            satellite_above_horizon,
            satellite_below_horizon,
        ]

        cutoff_time = satellite_below_horizon.time + timedelta(minutes=1)

        slew = SatellitesInterferenceFilter(
            facility=ARBITRARY_FACILITY,
            antenna_positions= [
                    AntennaPosition(satellite_positions=satellite_positions, antenna_direction=ARBITRARY_ANTENNA_POSITION)
                ],
                cutoff_time=cutoff_time,
                filter_strategy=SatellitesAboveHorizonFilter
        )

        windows = slew.run()
        expected_positions = [satellite_above_horizon]
        expected_windows = create_expected_windows(expected_positions)

        assert len(windows) == 1
        assert_windows_eq(windows, expected_windows)


    def satellite_above_horizon(self, time_offset=0):
        time_offset = timedelta(minutes=time_offset)
        return PositionTime(position=Position(altitude=100, azimuth=100), time=datetime.now(tz=pytz.UTC) + time_offset)

    def satellite_below_horizon(self, time_offset=0):
        time_offset = timedelta(minutes=time_offset)
        return PositionTime(position=Position(altitude=-100, azimuth=100), time=datetime.now(tz=pytz.UTC) + time_offset)


    #def test_satellite_is_below_horizon(self):
    #    self._run_close_azimuth_due_to_modulus(antenna_azimuth=self._HIGH_DEGREES, satellite_azimuth=self._LOW_DEGREES)

    #@staticmethod
    #def _run_close_azimuth_due_to_modulus(antenna_azimuth: float, satellite_azimuth: float):
    #    antenna_position_at_horizon = replace(ARBITRARY_ANTENNA_POSITION,
    #                                          position=replace(ARBITRARY_ANTENNA_POSITION.position,
    #                                                           azimuth=antenna_azimuth))
    #    satellite_positions = [
    #        replace(antenna_position_at_horizon, position=replace(antenna_position_at_horizon.position, azimuth=satellite_azimuth))
    #    ]
    #    cutoff_time = ARBITRARY_ANTENNA_POSITION.time + timedelta(minutes=1)
    #    slew = SatellitesInterferenceFilter(facility=ARBITRARY_FACILITY,
    #                                          antenna_positions=[
    #                                              AntennaPosition(satellite_positions=satellite_positions,
    #                                                              antenna_direction=None)],
    #                                          cutoff_time=cutoff_time,
    #                                          filter_strategy=SatellitesWithinMainBeamFilter)
    #    windows = slew.run()
    #    expected_positions = satellite_positions
    #    expected_windows = create_expected_windows(expected_positions)

    #    assert len(windows) == 1
    #    assert_windows_eq(windows, expected_windows)
