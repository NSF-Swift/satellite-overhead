import multiprocessing

from sopp.analysis.event_finders.base import EventFinder
from sopp.analysis.event_finders.interference import (
    AntennaPosition,
    SatellitesAboveHorizonFilter,
    SatellitesInterferenceFilter,
    SatellitesWithinMainBeamFilter,
)
from sopp.ephemeris.base import EphemerisCalculator
from sopp.models import (
    OverheadWindow,
    PositionTime,
    Reservation,
    RuntimeSettings,
    Satellite,
    TimeWindow,
)


class EventFinderSkyfield(EventFinder):
    def __init__(
        self,
        antenna_direction_path: list[PositionTime],
        list_of_satellites: list[Satellite],
        reservation: Reservation,
        ephemeris_calculator: EphemerisCalculator,
        runtime_settings: RuntimeSettings | None = None,
    ):
        super().__init__(
            antenna_direction_path=antenna_direction_path,
            list_of_satellites=list_of_satellites,
            reservation=reservation,
            runtime_settings=runtime_settings,
        )

        self.ephemeris_calculator = ephemeris_calculator
        self._filter_strategy = None

    def _get_rise_set_windows(self, satellite: Satellite) -> list[TimeWindow]:
        time_windows = self.ephemeris_calculator.find_events(
            satellite,
            self.runtime_settings.min_altitude,
            self.reservation.time.begin,
            self.reservation.time.end,
        )

        return time_windows

    def get_satellites_above_horizon(self) -> list[OverheadWindow]:
        overhead_windows = []
        for satellite in self.list_of_satellites:
            time_windows = self._get_rise_set_windows(satellite)

            satellite_overhead_windows = []
            for tw in time_windows:
                positions = self.ephemeris_calculator.get_positions_window(
                    satellite, tw.begin, tw.end
                )
                satellite_overhead_windows.append(
                    OverheadWindow(satellite=satellite, positions=positions),
                )

            overhead_windows.extend(satellite_overhead_windows)

        return overhead_windows

    def get_satellites_crossing_main_beam(self) -> list[OverheadWindow]:
        self._filter_strategy = SatellitesWithinMainBeamFilter
        return self._get_satellites_interference()

    def _get_satellites_interference(self) -> list[OverheadWindow]:
        processes = (
            int(self.runtime_settings.concurrency_level)
            if self.runtime_settings.concurrency_level > 1
            else 1
        )
        pool = multiprocessing.Pool(processes=processes)
        results = pool.map(
            self._get_satellite_overhead_windows, self.list_of_satellites
        )
        pool.close()
        pool.join()

        return [overhead_window for result in results for overhead_window in result]

    def _get_satellite_overhead_windows(
        self, satellite: Satellite
    ) -> list[OverheadWindow]:
        antenna_direction_end_times = [
            antenna_direction.time
            for antenna_direction in self.antenna_direction_path[1:]
        ] + [self.reservation.time.end]
        satellite_positions = self._get_satellite_positions_within_reservation(
            satellite
        )
        antenna_positions = [
            AntennaPosition(
                satellite_positions=self._filter_satellite_positions_within_time_window(
                    satellite_positions,
                    time_window=TimeWindow(
                        begin=max(self.reservation.time.begin, antenna_direction.time),
                        end=end_time,
                    ),
                ),
                antenna_direction=antenna_direction,
            )
            for antenna_direction, end_time in zip(
                self.antenna_direction_path, antenna_direction_end_times, strict=False
            )
            if end_time > self.reservation.time.begin
        ]
        time_windows = SatellitesInterferenceFilter(
            facility=self.reservation.facility,
            antenna_positions=antenna_positions,
            cutoff_time=self.reservation.time.end,
            filter_strategy=self._filter_strategy,
            runtime_settings=self.runtime_settings,
        ).run()

        return [
            OverheadWindow(satellite=satellite, positions=positions)
            for positions in time_windows
        ]

    def _get_satellite_positions_within_reservation(
        self, satellite: Satellite
    ) -> list[PositionTime]:
        return self.ephemeris_calculator.get_positions_window(
            satellite=satellite,
            start=self.reservation.time.begin,
            end=self.reservation.time.end,
        )

    @staticmethod
    def _filter_satellite_positions_within_time_window(
        satellite_positions: list[PositionTime], time_window: TimeWindow
    ) -> list[PositionTime]:
        return [
            positions
            for positions in satellite_positions
            if time_window.begin <= positions.time < time_window.end
        ]
