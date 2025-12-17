from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sopp.ephemeris.base import EphemerisCalculator
from sopp.models.configuration import Configuration, RuntimeSettings
from sopp.models.core import Coordinates, FrequencyRange, Position, TimeWindow
from sopp.models.ground.config import CustomTrajectoryConfig
from sopp.models.ground.facility import Facility
from sopp.models.ground.trajectory import AntennaTrajectory
from sopp.models.reservation import Reservation
from sopp.models.satellite import InternationalDesignator, MeanMotion, TleInformation
from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.trajectory import SatelliteTrajectory
from sopp.utils.time import generate_time_grid

# Constants
ARBITRARY_ALTITUDE = 0
ARBITRARY_AZIMUTH = 0


class EphemerisCalculatorStub(EphemerisCalculator):
    """
    Simulates a satellite that is always visible and always located at
    (ARBITRARY_AZIMUTH, ARBITRARY_ALTITUDE).
    """

    def __init__(*args, **kwargs):
        pass

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
def ephemeris_stub():
    return EphemerisCalculatorStub


@pytest.fixture
def start_time():
    return datetime.now(tz=timezone.utc)


@pytest.fixture
def time_window_duration():
    return timedelta(seconds=5)


@pytest.fixture
def antenna_trajectory():
    return AntennaTrajectory(
        np.array([datetime.now()]),
        np.array([ARBITRARY_AZIMUTH]),
        np.array([ARBITRARY_ALTITUDE]),
    )


@pytest.fixture
def antenna_config(antenna_trajectory):
    return CustomTrajectoryConfig(antenna_trajectory)


@pytest.fixture
def frequency_range():
    return FrequencyRange(frequency=10, bandwidth=10, status="active")


@pytest.fixture
def reservation(start_time, time_window_duration, frequency_range):
    return Reservation(
        facility=Facility(coordinates=Coordinates(latitude=0, longitude=0)),
        time=TimeWindow(begin=start_time, end=start_time + time_window_duration),
        frequency=frequency_range,
    )


@pytest.fixture
def configuration(
    antenna_trajectory,
    reservation,
    satellite,
):
    runtime_settings = RuntimeSettings()

    return Configuration(
        satellites=[satellite],
        antenna_config=CustomTrajectoryConfig(antenna_trajectory),
        reservation=reservation,
        runtime_settings=runtime_settings,
    )


@pytest.fixture
def satellite(tle_information) -> Satellite:
    """COSMOS 1932 DEB"""
    return Satellite(
        name="ARBITRARY SATELLITE",
        tle_information=tle_information,
        frequency=[],
    )


@pytest.fixture
def tle_information() -> TleInformation:
    return TleInformation(
        argument_of_perigee=5.153187590939126,
        drag_coefficient=0.00015211,
        eccentricity=0.0057116,
        epoch_days=26633.28893622,
        inclination=1.1352005427406557,
        international_designator=InternationalDesignator(
            year=88, launch_number=19, launch_piece="F"
        ),
        mean_anomaly=4.188343400497881,
        mean_motion=MeanMotion(
            first_derivative=2.363466695408988e-12,
            second_derivative=0.0,
            value=0.060298700041442894,
        ),
        revolution_number=95238,
        right_ascension_of_ascending_node=2.907844197528697,
        satellite_number=28275,
        classification="U",
    )


@pytest.fixture
def make_reservation(facility):
    """Factory to create reservations with custom start/duration."""

    def _make(start_time: datetime, duration_seconds: float = 2.0):
        window = TimeWindow(
            begin=start_time, end=start_time + timedelta(seconds=duration_seconds)
        )
        return Reservation(
            facility=facility,
            time=window,
            frequency=frequency_range,
        )

    return _make
