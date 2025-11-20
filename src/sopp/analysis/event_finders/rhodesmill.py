import multiprocessing

from sopp.analysis.event_finders.base import EventFinder
from sopp.analysis.event_finders.interference import (
    AntennaPosition,
    SatellitesAboveHorizonFilter,
    SatellitesInterferenceFilter,
    SatellitesWithinMainBeamFilter,
)
from sopp.models.overhead_window import OverheadWindow
from sopp.models.position_time import PositionTime
from sopp.models.reservation import Reservation
from sopp.models.runtime_settings import RuntimeSettings
from sopp.models.satellite.satellite import Satellite
from sopp.models.time_window import TimeWindow
from sopp.positioning.rhodesmill import (
    SatellitePositionsWithRespectToFacilityRetriever,
    SatellitePositionsWithRespectToFacilityRetrieverRhodesmill,
)
from sopp.utils.time import (
    EvenlySpacedTimeIntervalsCalculator,
)


class EventFinderRhodesmill(EventFinder):
    def __init__(
        self,
        antenna_direction_path: list[PositionTime],
        list_of_satellites: list[Satellite],
        reservation: Reservation,
        satellite_positions_with_respect_to_facility_retriever_class: type[
            SatellitePositionsWithRespectToFacilityRetriever
        ] = SatellitePositionsWithRespectToFacilityRetrieverRhodesmill,
        runtime_settings: RuntimeSettings = RuntimeSettings(),
    ):
        super().__init__(
            antenna_direction_path=antenna_direction_path,
            list_of_satellites=list_of_satellites,
            reservation=reservation,
            satellite_positions_with_respect_to_facility_retriever_class=satellite_positions_with_respect_to_facility_retriever_class,
            runtime_settings=runtime_settings,
        )

        datetimes = EvenlySpacedTimeIntervalsCalculator(
            time_window=reservation.time,
            resolution=runtime_settings.time_continuity_resolution,
        ).run()

        self._satellite_positions_retriever = (
            satellite_positions_with_respect_to_facility_retriever_class(
                facility=reservation.facility, datetimes=datetimes
            )
        )

        self._filter_strategy = None

    def get_satellites_above_horizon(self):
        self._filter_strategy = SatellitesAboveHorizonFilter
        return self._get_satellites_interference()

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
        return self._satellite_positions_retriever.run(satellite)

    @staticmethod
    def _filter_satellite_positions_within_time_window(
        satellite_positions: list[PositionTime], time_window: TimeWindow
    ) -> list[PositionTime]:
        return [
            positions
            for positions in satellite_positions
            if time_window.begin <= positions.time < time_window.end
        ]
