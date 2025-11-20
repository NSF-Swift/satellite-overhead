from dataclasses import replace
from datetime import timedelta

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


class TestSatellitesWithinMainBeamModulo360:
    _LOW_DEGREES = 0.5
    _HIGH_DEGREES = 359.5

    def test_satellite_is_high_antenna_is_low(self):
        self._run_close_azimuth_due_to_modulus(
            antenna_azimuth=self._LOW_DEGREES, satellite_azimuth=self._HIGH_DEGREES
        )

    def test_satellite_is_low_antenna_is_high(self):
        self._run_close_azimuth_due_to_modulus(
            antenna_azimuth=self._HIGH_DEGREES, satellite_azimuth=self._LOW_DEGREES
        )

    @staticmethod
    def _run_close_azimuth_due_to_modulus(
        antenna_azimuth: float, satellite_azimuth: float
    ):
        antenna_position_at_horizon = replace(
            ARBITRARY_ANTENNA_POSITION,
            position=replace(
                ARBITRARY_ANTENNA_POSITION.position, azimuth=antenna_azimuth
            ),
        )
        satellite_positions = [
            replace(
                antenna_position_at_horizon,
                position=replace(
                    antenna_position_at_horizon.position, azimuth=satellite_azimuth
                ),
            )
        ]
        cutoff_time = ARBITRARY_ANTENNA_POSITION.time + timedelta(minutes=1)
        slew = SatellitesInterferenceFilter(
            facility=ARBITRARY_FACILITY,
            antenna_positions=[
                AntennaPosition(
                    satellite_positions=satellite_positions,
                    antenna_direction=antenna_position_at_horizon,
                )
            ],
            cutoff_time=cutoff_time,
            filter_strategy=SatellitesWithinMainBeamFilter,
        )
        windows = slew.run()
        expected_positions = satellite_positions
        expected_windows = create_expected_windows(expected_positions)

        assert len(windows) == 1
        assert_windows_eq(windows, expected_windows)
