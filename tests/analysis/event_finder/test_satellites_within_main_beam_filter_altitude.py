from dataclasses import replace
from datetime import datetime, timezone
from functools import cached_property

from sopp.analysis.event_finders.interference import (
    AntennaPosition,
    SatellitesInterferenceFilter,
    SatellitesWithinMainBeamFilter,
)
from sopp.models.position import Position
from sopp.models.position_time import PositionTime

from tests.analysis.event_finder.definitions import (
    ARBITRARY_ANTENNA_POSITION,
    ARBITRARY_FACILITY,
)
from tests.definitions import SMALL_EPSILON


class TestSatellitesWithinMainBeamAltitude:
    def test_one_satellite_position_below_beamwidth_altitude(self):
        self._run_test(
            altitude=ARBITRARY_ANTENNA_POSITION.position.altitude
            - self._value_slightly_larger_than_half_beamwidth,
            expected_windows=[],
        )

    def test_one_satellite_position_above_beamwidth_altitude(self):
        altitude = (
            ARBITRARY_ANTENNA_POSITION.position.altitude
            + self._value_slightly_larger_than_half_beamwidth
        )

        self._run_test(
            altitude=altitude,
            expected_windows=[
                [
                    PositionTime(
                        position=Position(altitude=altitude, azimuth=100),
                        time=ARBITRARY_ANTENNA_POSITION.time,
                    )
                ]
            ],
        )

    def _run_test(self, altitude: float, expected_windows: list[PositionTime]) -> None:
        satellite_positions = [
            replace(
                ARBITRARY_ANTENNA_POSITION,
                position=replace(
                    ARBITRARY_ANTENNA_POSITION.position, altitude=altitude
                ),
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
        assert windows == expected_windows

    @property
    def _value_slightly_larger_than_half_beamwidth(self) -> float:
        return ARBITRARY_FACILITY.half_beamwidth + SMALL_EPSILON

    @cached_property
    def _arbitrary_cutoff_time(self) -> datetime:
        return datetime.now(tz=timezone.utc)
