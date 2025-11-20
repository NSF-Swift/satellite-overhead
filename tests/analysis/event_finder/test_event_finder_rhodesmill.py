from datetime import datetime, timedelta, timezone

from sopp.analysis.event_finders.rhodesmill import (
    EventFinderRhodesmill,
)
from sopp.models.coordinates import Coordinates
from sopp.models.facility import Facility
from sopp.models.position import Position
from sopp.models.position_time import PositionTime
from sopp.models.reservation import Reservation
from sopp.models.satellite.satellite import Satellite
from sopp.models.time_window import TimeWindow

from tests.analysis.event_finder.definitions import (
    create_overhead_window,
)
from tests.definitions import SMALL_EPSILON

ARBITRARY_SATELLITE_ALTITUDE = 0
ARBITRARY_SATELLITE_AZIMUTH = 0


class SatellitePositionsWithRespectToFacilityRetrieverStub:
    def __init__(self, facility, datetimes):
        self._datetimes = datetimes

    def run(self, satellite: Satellite) -> list[PositionTime]:
        return [
            PositionTime(
                position=Position(
                    altitude=ARBITRARY_SATELLITE_ALTITUDE,
                    azimuth=ARBITRARY_SATELLITE_AZIMUTH,
                ),
                time=time,
            )
            for time in self._datetimes
        ]


class TestEventFinderRhodesmill:
    def test_single_satellite(self):
        arbitrary_satellite = Satellite(name="arbitrary")
        arbitrary_datetime = datetime.now(tz=timezone.utc)
        arbitrary_time_window = TimeWindow(
            begin=arbitrary_datetime, end=arbitrary_datetime + timedelta(seconds=2)
        )
        arbitrary_reservation = Reservation(
            facility=Facility(coordinates=Coordinates(latitude=0, longitude=0)),
            time=arbitrary_time_window,
        )
        event_finder = EventFinderRhodesmill(
            list_of_satellites=[arbitrary_satellite],
            reservation=arbitrary_reservation,
            antenna_direction_path=[
                PositionTime(
                    position=Position(
                        altitude=ARBITRARY_SATELLITE_ALTITUDE,
                        azimuth=ARBITRARY_SATELLITE_AZIMUTH,
                    ),
                    time=arbitrary_datetime,
                )
            ],
            satellite_positions_with_respect_to_facility_retriever_class=SatellitePositionsWithRespectToFacilityRetrieverStub,
        )
        windows = event_finder.get_satellites_crossing_main_beam()
        expected_windows = [
            create_overhead_window(
                arbitrary_satellite, 0, 0, arbitrary_time_window.begin, 2
            )
        ]

        assert len(windows) == 1
        assert windows == expected_windows

    def test_multiple_satellites(self):
        arbitrary_satellites = [
            Satellite(name="arbitrary"),
            Satellite(name="arbitrary2"),
        ]
        arbitrary_datetime = datetime.now(tz=timezone.utc)
        arbitrary_time_window = TimeWindow(
            begin=arbitrary_datetime, end=arbitrary_datetime + timedelta(seconds=2)
        )
        arbitrary_reservation = Reservation(
            facility=Facility(coordinates=Coordinates(latitude=0, longitude=0)),
            time=arbitrary_time_window,
        )
        event_finder = EventFinderRhodesmill(
            list_of_satellites=arbitrary_satellites,
            reservation=arbitrary_reservation,
            antenna_direction_path=[
                PositionTime(
                    position=Position(
                        altitude=ARBITRARY_SATELLITE_ALTITUDE,
                        azimuth=ARBITRARY_SATELLITE_AZIMUTH,
                    ),
                    time=arbitrary_datetime,
                )
            ],
            satellite_positions_with_respect_to_facility_retriever_class=SatellitePositionsWithRespectToFacilityRetrieverStub,
        )
        windows = event_finder.get_satellites_crossing_main_beam()
        expected_windows = [
            create_overhead_window(
                arbitrary_satellites[0], 0, 0, arbitrary_time_window.begin, 2
            ),
            create_overhead_window(
                arbitrary_satellites[1], 0, 0, arbitrary_time_window.begin, 2
            ),
        ]

        assert len(windows) == 2
        assert windows == expected_windows

    def test_multiple_antenna_positions_with_azimuth_filtering(self):
        arbitrary_satellite = Satellite(name="arbitrary")
        arbitrary_datetime = datetime.now(tz=timezone.utc)

        arbitrary_time_window = TimeWindow(
            begin=arbitrary_datetime, end=arbitrary_datetime + timedelta(seconds=5)
        )

        arbitrary_reservation = Reservation(
            facility=Facility(coordinates=Coordinates(latitude=0, longitude=0)),
            time=arbitrary_time_window,
        )

        altitude_outside_beamwidth = (
            ARBITRARY_SATELLITE_ALTITUDE
            + arbitrary_reservation.facility.half_beamwidth
            + SMALL_EPSILON
        )
        event_finder = EventFinderRhodesmill(
            list_of_satellites=[arbitrary_satellite],
            reservation=arbitrary_reservation,
            antenna_direction_path=[
                PositionTime(
                    position=Position(
                        altitude=ARBITRARY_SATELLITE_ALTITUDE,
                        azimuth=ARBITRARY_SATELLITE_AZIMUTH,
                    ),
                    time=arbitrary_datetime,
                ),
                PositionTime(
                    position=Position(
                        altitude=altitude_outside_beamwidth,
                        azimuth=ARBITRARY_SATELLITE_AZIMUTH,
                    ),
                    time=arbitrary_datetime + timedelta(seconds=1),
                ),
                PositionTime(
                    position=Position(
                        altitude=ARBITRARY_SATELLITE_ALTITUDE,
                        azimuth=ARBITRARY_SATELLITE_AZIMUTH,
                    ),
                    time=arbitrary_datetime + timedelta(seconds=2),
                ),
            ],
            satellite_positions_with_respect_to_facility_retriever_class=SatellitePositionsWithRespectToFacilityRetrieverStub,
        )

        windows = event_finder.get_satellites_crossing_main_beam()
        expected_windows = [
            create_overhead_window(
                arbitrary_satellite,
                ARBITRARY_SATELLITE_ALTITUDE,
                ARBITRARY_SATELLITE_AZIMUTH,
                arbitrary_time_window.begin,
                1,
            ),
            create_overhead_window(
                arbitrary_satellite,
                ARBITRARY_SATELLITE_ALTITUDE,
                ARBITRARY_SATELLITE_AZIMUTH,
                arbitrary_time_window.begin + timedelta(seconds=2),
                3,
            ),
        ]

        assert len(windows) == 2
        assert windows == expected_windows
