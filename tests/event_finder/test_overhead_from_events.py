from datetime import datetime, timedelta

from satellite_determination.custom_dataclasses.coordinates import Coordinates
from satellite_determination.custom_dataclasses.facility import Facility
from satellite_determination.custom_dataclasses.overhead_window import OverheadWindow
from satellite_determination.custom_dataclasses.reservation import Reservation
from satellite_determination.custom_dataclasses.satellite.satellite import Satellite
from satellite_determination.custom_dataclasses.time_window import TimeWindow
from satellite_determination.event_finder.event_finder_rhodesmill.support.overhead_window_from_events import \
    EventTypesRhodesmill, EventRhodesmill, OverheadWindowFromEvents


class TestOverheadWindowFromEvents:

    def test_one_satellite_only_leaves(self):
        event = EventRhodesmill(
            event_type=EventTypesRhodesmill.EXITS,
            satellite=self._arbitrary_satellite,
            timestamp=self._arbitrary_date
        )
        overhead_windows = OverheadWindowFromEvents(events=[event], reservation=self._arbitrary_reservation_with_nonzero_timewindow).get()
        assert overhead_windows == [
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=self._arbitrary_reservation_with_nonzero_timewindow.time.begin,
                    end=event.timestamp
                )
            )
        ]

    def test_one_satellite_only_enters(self):
        event = EventRhodesmill(
            event_type=EventTypesRhodesmill.ENTERS,
            satellite=self._arbitrary_satellite,
            timestamp=self._arbitrary_date
        )
        overhead_windows = OverheadWindowFromEvents(events=[event], reservation=self._arbitrary_reservation_with_nonzero_timewindow).get()
        assert overhead_windows == [
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=event.timestamp,
                    end=self._arbitrary_reservation_with_nonzero_timewindow.time.end
                )
            )
        ]

    def test_one_satellite_stays_in_view(self):
        events = [
            EventRhodesmill(
                event_type=EventTypesRhodesmill.CULMINATES,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date)
        ]
        overhead_windows = OverheadWindowFromEvents(events=events, reservation=self._arbitrary_reservation_with_nonzero_timewindow).get()
        assert overhead_windows == [
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=self._arbitrary_reservation_with_nonzero_timewindow.time.begin,
                    end=self._arbitrary_reservation_with_nonzero_timewindow.time.end
                )
            )
        ]

    def test_one_satellite_enters_and_leaves(self):
        events = {
            EventTypesRhodesmill.ENTERS: EventRhodesmill(
                event_type=EventTypesRhodesmill.ENTERS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date
            ),
            EventTypesRhodesmill.EXITS: EventRhodesmill(
                event_type=EventTypesRhodesmill.EXITS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date + timedelta(hours=1)
            ),
        }
        overhead_windows = OverheadWindowFromEvents(events=list(events.values()), reservation=self._arbitrary_reservation_with_nonzero_timewindow).get()
        assert overhead_windows == [
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=events[EventTypesRhodesmill.ENTERS].timestamp,
                    end=events[EventTypesRhodesmill.EXITS].timestamp
                )
            )
        ]

    def test_one_satellite_enters_and_leaves_twice(self):
        events = [
            EventRhodesmill(
                event_type=EventTypesRhodesmill.ENTERS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date
            ),
            EventRhodesmill(
                event_type=EventTypesRhodesmill.EXITS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date + timedelta(hours=1)
            ),
            EventRhodesmill(
                event_type=EventTypesRhodesmill.ENTERS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date_two
            ),
            EventRhodesmill(
                event_type=EventTypesRhodesmill.EXITS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date_two + timedelta(hours=1)
            ),
        ]
        print(events)
        overhead_windows = OverheadWindowFromEvents(events=events, reservation=self._arbitrary_reservation_with_nonzero_timewindow).get()
        assert overhead_windows == [
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=events[0].timestamp,
                    end=events[1].timestamp
                )
            ),
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=events[2].timestamp,
                    end=events[3].timestamp
                )
            )
        ]

    def test_one_satellite_leaves_then_enters_twice(self):
        events = [
            EventRhodesmill(
                event_type=EventTypesRhodesmill.EXITS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date
            ),
            EventRhodesmill(
                event_type=EventTypesRhodesmill.ENTERS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date + timedelta(hours=1)
            ),
            EventRhodesmill(
                event_type=EventTypesRhodesmill.EXITS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date_two
            ),
            EventRhodesmill(
                event_type=EventTypesRhodesmill.ENTERS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date_two + timedelta(hours=1)
            ),
        ]
        overhead_windows = OverheadWindowFromEvents(events=events, reservation=self._arbitrary_reservation_with_nonzero_timewindow).get()
        assert overhead_windows == [
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=self._arbitrary_reservation_with_nonzero_timewindow.time.begin,
                    end=events[0].timestamp
                )
            ),
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=events[1].timestamp,
                    end=events[2].timestamp
                )
            ),
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=events[3].timestamp,
                    end=self._arbitrary_reservation_with_nonzero_timewindow.time.end
                )
            )
        ]

    def test_one_satellite_enters_leaves_enters(self):
        events = [
            EventRhodesmill(
                event_type=EventTypesRhodesmill.ENTERS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date
            ),
            EventRhodesmill(
                event_type=EventTypesRhodesmill.EXITS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date + timedelta(hours=1)
            ),
            EventRhodesmill(
                event_type=EventTypesRhodesmill.ENTERS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date_two
            ),
        ]
        overhead_windows = OverheadWindowFromEvents(events=events, reservation=self._arbitrary_reservation_with_nonzero_timewindow).get()
        assert overhead_windows == [
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=events[0].timestamp,
                    end=events[1].timestamp
                )
            ),
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=events[2].timestamp,
                    end=self._arbitrary_reservation_with_nonzero_timewindow.time.end
                )
            )
        ]

    def test_one_satellite_leaves_enters_leaves(self):
        events = [
            EventRhodesmill(
                event_type=EventTypesRhodesmill.EXITS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date
            ),
            EventRhodesmill(
                event_type=EventTypesRhodesmill.ENTERS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date + timedelta(hours=1)
            ),
            EventRhodesmill(
                event_type=EventTypesRhodesmill.EXITS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date + timedelta(hours=2)
            ),
        ]
        overhead_windows = OverheadWindowFromEvents(events=events, reservation=self._arbitrary_reservation_with_nonzero_timewindow).get()
        assert overhead_windows == [
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=self._arbitrary_reservation_with_nonzero_timewindow.time.begin,
                    end=events[0].timestamp
                )
            ),
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=events[1].timestamp,
                    end=events[2].timestamp
                )
            )
        ]

    def test_satellite_1_2_0_1_2(self):
        events = [
            EventRhodesmill(
                event_type=EventTypesRhodesmill.CULMINATES,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date
            ),
            EventRhodesmill(
                event_type=EventTypesRhodesmill.EXITS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date + timedelta(hours=1)
            ),
            EventRhodesmill(
                event_type=EventTypesRhodesmill.ENTERS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date + timedelta(hours=2)
            ),
            EventRhodesmill(
                event_type=EventTypesRhodesmill.CULMINATES,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date
            ),
            EventRhodesmill(
                event_type=EventTypesRhodesmill.EXITS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date + timedelta(hours=3)
            ),
        ]
        overhead_windows = OverheadWindowFromEvents(events=events, reservation=self._arbitrary_reservation_with_nonzero_timewindow).get()
        assert overhead_windows == [
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=self._arbitrary_reservation_with_nonzero_timewindow.time.begin,
                    end=events[1].timestamp
                )
            ),
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=events[2].timestamp,
                    end=events[4].timestamp
                )
            )
        ]

    def test_satellite_1_2_0_1(self):
        events = [
            EventRhodesmill(
                event_type=EventTypesRhodesmill.CULMINATES,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date
            ),
            EventRhodesmill(
                event_type=EventTypesRhodesmill.EXITS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date + timedelta(hours=1)
            ),
            EventRhodesmill(
                event_type=EventTypesRhodesmill.ENTERS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date + timedelta(hours=2)
            ),
            EventRhodesmill(
                event_type=EventTypesRhodesmill.CULMINATES,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date + timedelta(hours=3)
            ),
        ]
        overhead_windows = OverheadWindowFromEvents(events=events, reservation=self._arbitrary_reservation_with_nonzero_timewindow).get()
        assert overhead_windows == [
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=self._arbitrary_reservation_with_nonzero_timewindow.time.begin,
                    end=events[1].timestamp
                )
            ),
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=events[2].timestamp,
                    end=self._arbitrary_reservation_with_nonzero_timewindow.time.end
                )
            )
        ]

    def test_satellite_0_1_2_0(self):
        events = [
            EventRhodesmill(
                event_type=EventTypesRhodesmill.ENTERS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date
            ),
            EventRhodesmill(
                event_type=EventTypesRhodesmill.CULMINATES,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date + timedelta(hours=1)
            ),
            EventRhodesmill(
                event_type=EventTypesRhodesmill.EXITS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date + timedelta(hours=2)
            ),
            EventRhodesmill(
                event_type=EventTypesRhodesmill.ENTERS,
                satellite=self._arbitrary_satellite,
                timestamp=self._arbitrary_date + timedelta(hours=3)
            ),
        ]
        overhead_windows = OverheadWindowFromEvents(events=events, reservation=self._arbitrary_reservation_with_nonzero_timewindow).get()
        assert overhead_windows == [
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=events[0].timestamp,
                    end=events[2].timestamp
                )
            ),
            OverheadWindow(
                satellite=self._arbitrary_satellite,
                overhead_time=TimeWindow(
                    begin=events[3].timestamp,
                    end=self._arbitrary_reservation_with_nonzero_timewindow.time.end
                )
            )
        ]

#Could a sat be in view but not trigger any of the events?
    @property
    def _arbitrary_satellite(self) -> Satellite:
        return Satellite(name='name')

    @property
    def _arbitrary_date(self) -> datetime:
        return datetime(year=2000, month=1, day=1, hour=1)

    @property
    def _arbitrary_date_two(self) -> datetime:
        return datetime(year=2000, month=3, day=3, hour=3)

    @property
    def _arbitrary_reservation_with_nonzero_timewindow(self) -> Reservation:
        return Reservation(facility=Facility(angle_of_visibility_cone=0,
                                             point_coordinates=Coordinates(latitude=0, longitude=0),
                                             name='name', azimuth=30),
                           time=TimeWindow(begin=datetime(year=2001, month=2, day=1, hour=1), end=datetime(year=2001, month=2, day=1, hour=6)))


#TestOverheadWindowFromEvents.test_one_satellite_enters_and_leaves_twice(TestOverheadWindowFromEvents)
