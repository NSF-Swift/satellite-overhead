from dataclasses import replace
from datetime import datetime, timedelta, timezone

from sopp.analysis.event_finders.interference import (
    AntennaPosition,
    SatellitesInterferenceFilter,
    SatellitesWithinMainBeamFilter,
)
from sopp.models.position_time import PositionTime

from tests.analysis.event_finder.definitions import (
    ARBITRARY_ANTENNA_POSITION,
    ARBITRARY_FACILITY,
    assert_windows_eq,
    create_expected_windows,
)
from tests.definitions import SMALL_EPSILON


class TestSatellitesWithinMainBeam:
    def test_no_satellite_positions(self):
        slew = SatellitesInterferenceFilter(
            facility=ARBITRARY_FACILITY,
            antenna_positions=[
                AntennaPosition(
                    satellite_positions=[], antenna_direction=ARBITRARY_ANTENNA_POSITION
                )
            ],
            cutoff_time=self._arbitrary_cutoff_time,
            filter_strategy=SatellitesWithinMainBeamFilter,
        )
        windows = slew.run()
        assert windows == []

    def test_one_satellite_position_exactly_at_antenna_position(self):
        satellite_positions = [ARBITRARY_ANTENNA_POSITION]
        cutoff_time = ARBITRARY_ANTENNA_POSITION.time + timedelta(minutes=1)
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
        expected_positions = [ARBITRARY_ANTENNA_POSITION]
        expected_windows = create_expected_windows(expected_positions)

        assert len(windows) == 1
        assert_windows_eq(windows, expected_windows)

    def test_one_satellite_position_outside_beamwidth_azimuth(self):
        satellite_positions = [
            self._replace_antenna_position(
                antenna_position=ARBITRARY_ANTENNA_POSITION,
                azimuth=ARBITRARY_ANTENNA_POSITION.position.azimuth
                - self._value_slightly_larger_than_half_beamwidth,
            )
        ]
        slew = SatellitesInterferenceFilter(
            facility=ARBITRARY_FACILITY,
            antenna_positions=[
                AntennaPosition(
                    satellite_positions=satellite_positions,
                    antenna_direction=ARBITRARY_ANTENNA_POSITION,
                )
            ],
            cutoff_time=self._arbitrary_cutoff_time,
            filter_strategy=SatellitesWithinMainBeamFilter,
        )
        windows = slew.run()
        assert windows == []

    def test_one_satellite_with_multiple_sequential_positions_in_view(self):
        out_of_altitude = (
            ARBITRARY_ANTENNA_POSITION.position.altitude
            - self._value_slightly_larger_than_half_beamwidth
        )
        satellite_positions = [
            self._replace_antenna_position(
                antenna_position=ARBITRARY_ANTENNA_POSITION,
                altitude=out_of_altitude
                if i == 2
                else ARBITRARY_ANTENNA_POSITION.position.altitude,
                time=ARBITRARY_ANTENNA_POSITION.time + timedelta(minutes=i),
            )
            for i in range(3)
        ]
        cutoff_time = ARBITRARY_ANTENNA_POSITION.time + timedelta(
            minutes=len(satellite_positions)
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
        expected_positions = [[satellite_positions[0], satellite_positions[1]]]
        expected_windows = create_expected_windows(expected_positions)

        assert len(windows) == 1
        assert_windows_eq(windows, expected_windows)

    def test_one_satellite_with_multiple_sequential_positions_out_of_view(self):
        out_of_altitude = (
            ARBITRARY_ANTENNA_POSITION.position.altitude
            - self._value_slightly_larger_than_half_beamwidth
        )
        satellite_positions = [
            self._replace_antenna_position(
                antenna_position=ARBITRARY_ANTENNA_POSITION,
                altitude=out_of_altitude
                if 0 < i < 3
                else ARBITRARY_ANTENNA_POSITION.position.altitude,
                time=ARBITRARY_ANTENNA_POSITION.time + timedelta(minutes=i),
            )
            for i in range(6)
        ]
        cutoff_time = ARBITRARY_ANTENNA_POSITION.time + timedelta(
            minutes=len(satellite_positions)
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
            satellite_positions[0],
            [satellite_positions[3], satellite_positions[4], satellite_positions[5]],
        ]
        expected_windows = create_expected_windows(expected_positions)

        assert len(windows) == 2
        assert_windows_eq(windows, expected_windows)

    @property
    def _value_slightly_larger_than_half_beamwidth(self) -> float:
        return ARBITRARY_FACILITY.half_beamwidth + SMALL_EPSILON

    def test_one_satellite_below_horizon_but_within_beamwidth(self):
        antenna_position_at_horizon = self._replace_antenna_position(
            antenna_position=ARBITRARY_ANTENNA_POSITION, altitude=0
        )
        satellite_positions = [
            self._replace_antenna_position(
                antenna_position=antenna_position_at_horizon, altitude=-SMALL_EPSILON
            )
        ]
        slew = SatellitesInterferenceFilter(
            facility=ARBITRARY_FACILITY,
            antenna_positions=[
                AntennaPosition(
                    satellite_positions=satellite_positions,
                    antenna_direction=antenna_position_at_horizon,
                )
            ],
            cutoff_time=self._arbitrary_cutoff_time,
            filter_strategy=SatellitesWithinMainBeamFilter,
        )
        windows = slew.run()
        assert windows == []

    def test_no_satellites_past_cutoff_time(self):
        azimuth_outside_main_beam = (
            ARBITRARY_ANTENNA_POSITION.position.azimuth
            + ARBITRARY_FACILITY.half_beamwidth
            + SMALL_EPSILON
        )
        satellite_position_outside_main_beam = self._replace_antenna_position(
            antenna_position=ARBITRARY_ANTENNA_POSITION,
            azimuth=azimuth_outside_main_beam,
        )
        satellite_position_inside_main_beam_but_past_cutoff_time = (
            self._replace_antenna_position(
                antenna_position=ARBITRARY_ANTENNA_POSITION,
                time=ARBITRARY_ANTENNA_POSITION.time + timedelta(seconds=2),
            )
        )
        satellite_positions = [
            satellite_position_outside_main_beam,
            satellite_position_inside_main_beam_but_past_cutoff_time,
        ]
        cutoff_time = ARBITRARY_ANTENNA_POSITION.time + timedelta(seconds=1)
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
        assert windows == []

    def test_satellite_reenters_horizon(self):
        facility_with_beam_width_that_sees_entire_sky = replace(
            ARBITRARY_FACILITY, beamwidth=360
        )
        satellite_positions_above_horizon = self._replace_antenna_position(
            ARBITRARY_ANTENNA_POSITION, altitude=1
        )
        satellite_positions_below_horizon = self._replace_antenna_position(
            ARBITRARY_ANTENNA_POSITION, altitude=-1
        )
        satellite_positions_without_time = [
            satellite_positions_above_horizon,
            satellite_positions_below_horizon,
            satellite_positions_above_horizon,
        ]
        satellite_positions = [
            replace(
                position, time=ARBITRARY_ANTENNA_POSITION.time + timedelta(seconds=i)
            )
            for i, position in enumerate(satellite_positions_without_time)
        ]
        cutoff_time = ARBITRARY_ANTENNA_POSITION.time + timedelta(
            seconds=len(satellite_positions)
        )
        slew = SatellitesInterferenceFilter(
            facility=facility_with_beam_width_that_sees_entire_sky,
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
        expected_positions = [satellite_positions[0], satellite_positions[2]]
        expected_windows = create_expected_windows(expected_positions)

        assert len(windows) == 2
        assert_windows_eq(windows, expected_windows)

    @property
    def _arbitrary_cutoff_time(self) -> datetime:
        return datetime.now(tz=timezone.utc)

    @staticmethod
    def _replace_antenna_position(
        antenna_position: PositionTime,
        altitude: float | None = None,
        azimuth: float | None = None,
        time: datetime | None = None,
    ) -> PositionTime:
        return replace(
            antenna_position,
            position=replace(
                antenna_position.position,
                altitude=antenna_position.position.altitude
                if altitude is None
                else altitude,
                azimuth=antenna_position.position.azimuth
                if azimuth is None
                else azimuth,
            ),
            time=antenna_position.time if time is None else time,
        )
