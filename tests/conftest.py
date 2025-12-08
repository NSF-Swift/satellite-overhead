import pytest
import numpy as np
from datetime import datetime, timedelta, timezone

from sopp.models import (
    Coordinates,
    Facility,
    Position,
    Reservation,
    Satellite,
    TimeWindow,
    RuntimeSettings,
    SatelliteTrajectory,
    antenna_trajectory,
)
from sopp.ephemeris.base import EphemerisCalculator
from sopp.event_finders.skyfield import EventFinderSkyfield
from sopp.utils.time import generate_time_grid

# Constants
ARBITRARY_ALTITUDE = 0
ARBITRARY_AZIMUTH = 0


class EphemerisCalculatorStub(EphemerisCalculator):
    """
    Simulates a satellite that is always visible and always located at
    (ARBITRARY_AZIMUTH, ARBITRARY_ALTITUDE).
    """

    def calculate_visibility_windows(
        self, satellite, min_altitude, start_time, end_time
    ) -> list[TimeWindow]:
        # Always visible for the entire duration requested
        return [TimeWindow(start_time, end_time)]

    def calculate_trajectories(self, satellite, windows) -> list[SatelliteTrajectory]:
        # Simple implementation: just loop and call the single method
        return [self.calculate_trajectory(satellite, w.begin, w.end) for w in windows]

    def calculate_trajectory(self, satellite, start, end) -> SatelliteTrajectory:
        # 1. Generate grid (1 second resolution for tests)
        times = generate_time_grid(start, end, resolution_seconds=1.0)
        n = len(times)

        # 2. Create Constant Arrays (Always at the arbitrary position)
        azimuth = np.full(n, ARBITRARY_AZIMUTH, dtype=float)
        altitude = np.full(n, ARBITRARY_ALTITUDE, dtype=float)
        distance = np.full(n, 500.0, dtype=float)

        return SatelliteTrajectory(
            satellite=satellite,
            times=times,
            azimuth=azimuth,
            altitude=altitude,
            distance_km=distance,
        )

    def calculate_position(self, satellite, time) -> Position:
        # Scalar return Position
        return Position(
            altitude=ARBITRARY_ALTITUDE,
            azimuth=ARBITRARY_AZIMUTH,
            distance_km=500.0,
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

    def _make(reservation, satellites, antenna_trajectory=None):
        return EventFinderSkyfield(
            list_of_satellites=satellites,
            reservation=reservation,
            antenna_trajectory=antenna_trajectory,
            ephemeris_calculator=ephemeris_stub,
            runtime_settings=RuntimeSettings(),
        )

    return _make
