from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sopp.models.frequency_range import FrequencyRange
from sopp.io.get_frequency_data_from_csv import (
    GetFrequencyDataFromCsv,
)
from sopp.models.overhead_window import OverheadWindow
from sopp.models.reservation import Reservation
from sopp.models.satellite.international_designator import (
    InternationalDesignator,
)
from sopp.models.satellite.mean_motion import MeanMotion
from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.tle_information import TleInformation
from sopp.models.time_window import TimeWindow
from sopp.utils.helpers import get_script_directory
from sopp.scheduling.observation_window_finder import SuggestedReservation, WindowFinder

from tests.scheduling.definitions import ARBITRARY_FACILITY

_ARBITRARY_FREQUENCY_RANGE = FrequencyRange(frequency=2.0, bandwidth=1.0)


@pytest.mark.skip("This feature is deprecated due to users not wanting it.")
class TestSortedByLeastNumberOfSatellites:
    def test_search(self):
        satellites = Satellite.from_tle_file(
            tlefilepath=Path(
                get_script_directory(__file__), "international_space_station_tle.tle"
            )
        )
        frequency_file_path = Path(
            get_script_directory(__file__), "fake_ISS_frequency_file.csv"
        )
        frequency_list = GetFrequencyDataFromCsv(filepath=frequency_file_path).get()
        satellite_list_with_frequency = [
            replace(
                satellite,
                frequency=frequency_list.get(
                    satellite.tle_information.satellite_number, []
                ),
            )
            for satellite in satellites
        ]
        suggestions = WindowFinder(
            ideal_reservation=self._ideal_reservation,
            satellites=satellite_list_with_frequency,
            start_time_increments=timedelta(hours=4),
            search_window=timedelta(days=1),
        ).search()
        assert suggestions == self._expected_suggestions_search

    @property
    def _overhead_windows(self) -> list[OverheadWindow]:
        return (
            self._two_overhead_windows_on_ideal_reservation
            + self._one_overhead_window_on_second_closest_reservation
        )

    @property
    def _expected_suggestions_search(self) -> list[SuggestedReservation]:
        return [
            SuggestedReservation(
                suggested_start_time=datetime(
                    year=2022, month=11, day=20, hour=1, tzinfo=timezone.utc
                ),
                ideal_reservation=self._ideal_reservation,
                overhead_satellites=[],
            ),
            SuggestedReservation(
                suggested_start_time=datetime(
                    year=2022, month=11, day=20, hour=5, tzinfo=timezone.utc
                ),
                ideal_reservation=self._ideal_reservation,
                overhead_satellites=[],
            ),
            SuggestedReservation(
                suggested_start_time=datetime(
                    year=2022, month=11, day=19, hour=17, tzinfo=timezone.utc
                ),
                ideal_reservation=self._ideal_reservation,
                overhead_satellites=[],
            ),
            SuggestedReservation(
                suggested_start_time=datetime(
                    year=2022, month=11, day=20, hour=13, tzinfo=timezone.utc
                ),
                ideal_reservation=self._ideal_reservation,
                overhead_satellites=[],
            ),
            SuggestedReservation(
                suggested_start_time=datetime(
                    year=2022, month=11, day=19, hour=21, tzinfo=timezone.utc
                ),
                ideal_reservation=self._ideal_reservation,
                overhead_satellites=[
                    OverheadWindow(
                        satellite=Satellite(
                            name="FAKE ISS (ZARYA)",
                            tle_information=TleInformation(
                                argument_of_perigee=0.3083420829620822,
                                drag_coefficient=3.8792e-05,
                                eccentricity=0.0007417,
                                epoch_days=25545.69339541,
                                inclination=0.9013560935706996,
                                international_designator=InternationalDesignator(
                                    year=98, launch_number=67, launch_piece="A"
                                ),
                                mean_anomaly=1.4946964807494398,
                                mean_motion=MeanMotion(
                                    first_derivative=5.3450708342326346e-11,
                                    second_derivative=0.0,
                                    value=0.06763602333248933,
                                ),
                                revolution_number=20248,
                                right_ascension_of_ascending_node=3.686137125541276,
                                satellite_number=25544,
                            ),
                            frequency=[
                                FrequencyRange(
                                    frequency=136.65, bandwidth=None, status="active"
                                )
                            ],
                        ),
                        overhead_time=TimeWindow(
                            begin=datetime(
                                year=2022,
                                month=11,
                                day=19,
                                hour=23,
                                minute=45,
                                second=7,
                                microsecond=540745,
                                tzinfo=timezone.utc,
                            ),
                            end=datetime(
                                year=2022,
                                month=11,
                                day=19,
                                hour=23,
                                minute=45,
                                second=21,
                                microsecond=540745,
                                tzinfo=timezone.utc,
                            ),
                        ),
                    )
                ],
            ),
            SuggestedReservation(
                suggested_start_time=datetime(
                    year=2022, month=11, day=20, hour=9, tzinfo=timezone.utc
                ),
                ideal_reservation=self._ideal_reservation,
                overhead_satellites=[
                    OverheadWindow(
                        satellite=Satellite(
                            name="FAKE ISS (ZARYA)",
                            tle_information=TleInformation(
                                argument_of_perigee=0.3083420829620822,
                                drag_coefficient=3.8792e-05,
                                eccentricity=0.0007417,
                                epoch_days=25545.69339541,
                                inclination=0.9013560935706996,
                                international_designator=InternationalDesignator(
                                    year=98, launch_number=67, launch_piece="A"
                                ),
                                mean_anomaly=1.4946964807494398,
                                mean_motion=MeanMotion(
                                    first_derivative=5.3450708342326346e-11,
                                    second_derivative=0.0,
                                    value=0.06763602333248933,
                                ),
                                revolution_number=20248,
                                right_ascension_of_ascending_node=3.686137125541276,
                                satellite_number=25544,
                            ),
                            frequency=[
                                FrequencyRange(
                                    frequency=136.65, bandwidth=None, status="active"
                                )
                            ],
                        ),
                        overhead_time=TimeWindow(
                            begin=datetime(
                                year=2022,
                                month=11,
                                day=20,
                                hour=12,
                                minute=57,
                                second=59,
                                microsecond=107439,
                                tzinfo=timezone.utc,
                            ),
                            end=datetime(
                                year=2022,
                                month=11,
                                day=20,
                                hour=12,
                                minute=58,
                                second=47,
                                microsecond=107439,
                                tzinfo=timezone.utc,
                            ),
                        ),
                    )
                ],
            ),
        ]

    @property
    def _expected_suggestions(self) -> list[SuggestedReservation]:
        ideal_reservation_index = len(self._expected_suggestion_start_times) - 1
        second_closest_reservation_index = ideal_reservation_index - 1
        return [
            SuggestedReservation(
                suggested_start_time=start_time,
                ideal_reservation=self._ideal_reservation,
                overhead_satellites=self._two_overhead_windows_on_ideal_reservation
                if index == ideal_reservation_index
                else self._one_overhead_window_on_second_closest_reservation
                if index == second_closest_reservation_index
                else [],
            )
            for index, start_time in enumerate(self._expected_suggestion_start_times)
        ]

    @property
    def _ideal_reservation(self) -> Reservation:
        return Reservation(
            facility=ARBITRARY_FACILITY,
            time=TimeWindow(
                begin=datetime(
                    year=2022, month=11, day=20, hour=1, tzinfo=timezone.utc
                ),
                end=datetime(year=2022, month=11, day=20, hour=5, tzinfo=timezone.utc),
            ),
            frequency=FrequencyRange(frequency=None, bandwidth=None),
        )

    @property
    def _two_overhead_windows_on_ideal_reservation(self) -> list[OverheadWindow]:
        return [
            OverheadWindow(
                satellite=Satellite(frequency=_ARBITRARY_FREQUENCY_RANGE, name="name1"),
                overhead_time=TimeWindow(
                    begin=datetime(year=2022, month=11, day=20),
                    end=datetime(year=2022, month=11, day=20, hour=1),
                ),
            ),
            OverheadWindow(
                satellite=Satellite(frequency=_ARBITRARY_FREQUENCY_RANGE, name="name2"),
                overhead_time=TimeWindow(
                    begin=datetime(year=2022, month=11, day=20),
                    end=datetime(year=2022, month=11, day=20, hour=1),
                ),
            ),
        ]

    @property
    def _one_overhead_window_on_second_closest_reservation(
        self,
    ) -> list[OverheadWindow]:
        return [
            OverheadWindow(
                satellite=Satellite(frequency=_ARBITRARY_FREQUENCY_RANGE, name="name3"),
                overhead_time=TimeWindow(
                    begin=datetime(year=2022, month=11, day=21),
                    end=datetime(year=2022, month=11, day=21, hour=1),
                ),
            ),
        ]

    @property
    def _expected_suggestion_start_times(self) -> list[datetime]:
        return [
            datetime(year=2022, month=11, day=19),
            datetime(year=2022, month=11, day=22),
            datetime(year=2022, month=11, day=18),
            datetime(year=2022, month=11, day=23),
            datetime(year=2022, month=11, day=17),
            datetime(year=2022, month=11, day=21),
            datetime(year=2022, month=11, day=20),
        ]
