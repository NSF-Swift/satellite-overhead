from datetime import timedelta
from sopp.models.position import Position
from sopp.models.position_time import PositionTime
from tests.analysis.event_finder.definitions import create_overhead_window
from tests.definitions import SMALL_EPSILON

from tests.conftest import ARBITRARY_ALTITUDE, ARBITRARY_AZIMUTH


class TestEventFinderReservationStartTime:
    def test_reservation_begins_part_way_through_antenna_position_time(
        self, arbitrary_datetime, satellite, make_reservation, make_event_finder
    ):
        """
        Scenario: Antenna is set at T-1s. Reservation starts at T=0.
        Expected: The EventFinder should detect the overlap starting at T=0.
        """
        # Setup
        reservation = make_reservation(
            start_time=arbitrary_datetime, duration_seconds=2
        )

        # Antenna starts 1 second BEFORE reservation
        antenna_time = arbitrary_datetime - timedelta(seconds=1)
        antenna_path = [
            PositionTime(
                position=Position(
                    altitude=ARBITRARY_ALTITUDE, azimuth=ARBITRARY_AZIMUTH
                ),
                time=antenna_time,
            )
        ]

        event_finder = make_event_finder(
            reservation=reservation, satellites=[satellite], antenna_path=antenna_path
        )

        # Execute
        windows = event_finder.get_satellites_crossing_main_beam()

        # Verify
        # We expect a window starting at the Reservation Start (arbitrary_datetime), NOT the antenna time
        expected_window = create_overhead_window(
            satellite,
            ARBITRARY_ALTITUDE,
            ARBITRARY_AZIMUTH,
            reservation.time.begin,
            2,  # duration
        )

        assert len(windows) == 1
        # Check equality based on time/satellite logic
        assert windows[0].overhead_time.begin == expected_window.overhead_time.begin
        assert windows[0].satellite.name == expected_window.satellite.name

    def test_antenna_positions_that_end_before_reservation_starts_are_not_included(
        self,
        arbitrary_datetime,
        satellite,
        make_reservation,
        make_event_finder,
        facility,
    ):
        """
        Scenario: Antenna points at T-2 and T-1. Reservation starts at T=0.
        Expected: No windows found (Antenna path is in the past).
        """
        # Setup
        reservation = make_reservation(
            start_time=arbitrary_datetime, duration_seconds=2
        )

        # Define Altitudes
        alt_in_beam = ARBITRARY_ALTITUDE
        alt_out_beam = ARBITRARY_ALTITUDE + facility.half_beamwidth + SMALL_EPSILON

        # Times before reservation
        t_minus_2 = arbitrary_datetime - timedelta(seconds=2)
        t_minus_1 = arbitrary_datetime - timedelta(seconds=1)

        antenna_path = [
            # 1. Ends before reservation begins (T-2)
            PositionTime(
                position=Position(altitude=alt_in_beam, azimuth=ARBITRARY_AZIMUTH),
                time=t_minus_2,
            ),
            # 2. Ends exactly at reservation start (T-1)
            PositionTime(
                position=Position(altitude=alt_out_beam, azimuth=ARBITRARY_AZIMUTH),
                time=t_minus_1,
            ),
        ]

        event_finder = make_event_finder(
            reservation=reservation, satellites=[satellite], antenna_path=antenna_path
        )

        # Execute
        windows = event_finder.get_satellites_crossing_main_beam()

        # Verify
        assert windows == []
