from datetime import datetime, timedelta, timezone

from sopp.custom_dataclasses.position import Position
from sopp.custom_dataclasses.position_time import PositionTime
from sopp.custom_dataclasses.runtime_settings import RuntimeSettings
from sopp.event_finder.event_finder_rhodesmill.support.satellites_interference_filter import (
    AntennaPosition,
    SatellitesAboveHorizonFilter,
    SatellitesInterferenceFilter,
)

from tests.event_finder.event_finder_rhodesmill.definitions import (
    ARBITRARY_ANTENNA_POSITION,
    ARBITRARY_FACILITY,
    assert_windows_eq,
    create_expected_windows,
)


class TestSatellitesAboveHorizon:
    def test_satellite_is_above_horizon(self):
        satellite_above_horizon = self.satellite_above_horizon()
        satellite_positions = [satellite_above_horizon]

        cutoff_time = satellite_above_horizon.time + timedelta(minutes=1)

        slew = SatellitesInterferenceFilter(
            facility=ARBITRARY_FACILITY,
            antenna_positions=[
                AntennaPosition(
                    satellite_positions=satellite_positions,
                    antenna_direction=ARBITRARY_ANTENNA_POSITION,
                )
            ],
            cutoff_time=cutoff_time,
            filter_strategy=SatellitesAboveHorizonFilter,
        )

        windows = slew.run()
        expected_positions = satellite_positions
        expected_windows = create_expected_windows(expected_positions)

        assert len(windows) == 1
        assert_windows_eq(windows, expected_windows)

    def test_satellite_is_below_horizon(self):
        satellite_below_horizon = self.satellite_below_horizon()

        satellite_positions = [satellite_below_horizon]

        cutoff_time = satellite_below_horizon.time + timedelta(minutes=1)

        slew = SatellitesInterferenceFilter(
            facility=ARBITRARY_FACILITY,
            antenna_positions=[
                AntennaPosition(
                    satellite_positions=satellite_positions,
                    antenna_direction=ARBITRARY_ANTENNA_POSITION,
                )
            ],
            cutoff_time=cutoff_time,
            filter_strategy=SatellitesAboveHorizonFilter,
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
            antenna_positions=[
                AntennaPosition(
                    satellite_positions=satellite_positions,
                    antenna_direction=ARBITRARY_ANTENNA_POSITION,
                )
            ],
            cutoff_time=cutoff_time,
            filter_strategy=SatellitesAboveHorizonFilter,
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
            antenna_positions=[
                AntennaPosition(
                    satellite_positions=satellite_positions,
                    antenna_direction=ARBITRARY_ANTENNA_POSITION,
                )
            ],
            cutoff_time=cutoff_time,
            filter_strategy=SatellitesAboveHorizonFilter,
        )

        windows = slew.run()
        expected_positions = [satellite_above_horizon]
        expected_windows = create_expected_windows(expected_positions)

        assert len(windows) == 1
        assert_windows_eq(windows, expected_windows)

    def test_satellite_is_above_horizon_below_min_alt(self):
        satellite_above_horizon = self.satellite_above_horizon()

        satellite_positions = [satellite_above_horizon]

        cutoff_time = satellite_above_horizon.time + timedelta(minutes=1)

        slew = SatellitesInterferenceFilter(
            facility=ARBITRARY_FACILITY,
            antenna_positions=[
                AntennaPosition(
                    satellite_positions=satellite_positions,
                    antenna_direction=ARBITRARY_ANTENNA_POSITION,
                )
            ],
            cutoff_time=cutoff_time,
            filter_strategy=SatellitesAboveHorizonFilter,
            runtime_settings=RuntimeSettings(min_altitude=105.0),
        )

        windows = slew.run()
        expected_positions = []
        expected_windows = create_expected_windows(expected_positions)

        assert len(windows) == 0
        assert_windows_eq(windows, expected_windows)

    def satellite_above_horizon(self, time_offset=0):
        time_offset = timedelta(minutes=time_offset)
        return PositionTime(
            position=Position(altitude=100, azimuth=100),
            time=datetime.now(tz=timezone.utc) + time_offset,
        )

    def satellite_below_horizon(self, time_offset=0):
        time_offset = timedelta(minutes=time_offset)
        return PositionTime(
            position=Position(altitude=-100, azimuth=100),
            time=datetime.now(tz=timezone.utc) + time_offset,
        )
