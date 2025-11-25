# tests/conftest.py
import pytest
from datetime import datetime, timedelta, timezone
from sopp.models.coordinates import Coordinates
from sopp.models.facility import Facility
from sopp.models.position import Position
from sopp.models.position_time import PositionTime
from sopp.models.reservation import Reservation
from sopp.models.satellite.satellite import Satellite
from sopp.models.time_window import TimeWindow
from sopp.models.runtime_settings import RuntimeSettings
from sopp.ephemeris.base import EphemerisCalculator
from sopp.analysis.event_finders.skyfield import EventFinderSkyfield

# Constants
ARBITRARY_ALTITUDE = 0
ARBITRARY_AZIMUTH = 0


# --- STUB ---
class EphemerisCalculatorStub(EphemerisCalculator):
    """
    Simulates a satellite that is always visible and at (0,0) coordinates.
    """

    def find_events(self, satellite, min_altitude, start_time, end_time):
        return [TimeWindow(start_time, end_time)]

    def get_positions_window(self, satellite, start, end):
        # Generate 1-second positions for the whole window
        positions = []
        current = start
        perfect_pos = Position(
            altitude=ARBITRARY_ALTITUDE, azimuth=ARBITRARY_AZIMUTH, distance_km=500.0
        )
        while current <= end:
            positions.append(PositionTime(perfect_pos, current))
            current += timedelta(seconds=1)
        return positions

    def get_position_at(self, satellite, time):
        return PositionTime(
            Position(
                altitude=ARBITRARY_ALTITUDE,
                azimuth=ARBITRARY_AZIMUTH,
                distance_km=500.0,
            ),
            time=time,
        )


# --- FIXTURES ---


@pytest.fixture
def arbitrary_datetime():
    return datetime.now(tz=timezone.utc)


@pytest.fixture
def facility():
    return Facility(coordinates=Coordinates(latitude=0, longitude=0))


@pytest.fixture
def satellite():
    return Satellite(name="arbitrary")


@pytest.fixture
def ephemeris_stub():
    return EphemerisCalculatorStub()


@pytest.fixture
def start_time():
    return datetime.now(tz=timezone.utc)


@pytest.fixture
def time_window_duration():
    # Default duration, tests can override if they construct their own
    return timedelta(seconds=5)


@pytest.fixture
def reservation(start_time, time_window_duration):
    return Reservation(
        facility=Facility(coordinates=Coordinates(latitude=0, longitude=0)),
        time=TimeWindow(begin=start_time, end=start_time + time_window_duration),
    )


@pytest.fixture
def make_reservation(facility):
    """Factory to create reservations with custom start/duration."""

    def _make(start_time: datetime, duration_seconds: float = 2.0):
        window = TimeWindow(
            begin=start_time, end=start_time + timedelta(seconds=duration_seconds)
        )
        return Reservation(facility=facility, time=window)

    return _make


@pytest.fixture
def make_event_finder(ephemeris_stub):
    """
    Factory to assemble the EventFinder with specific reservation/satellites/antenna.
    """

    def _make(reservation, satellites, antenna_path):
        return EventFinderSkyfield(
            list_of_satellites=satellites,
            reservation=reservation,
            antenna_direction_path=antenna_path,
            ephemeris_calculator=ephemeris_stub,
            runtime_settings=RuntimeSettings(),
        )

    return _make
