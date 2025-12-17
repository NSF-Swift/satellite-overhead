import pytest

from sopp.models.configuration import Configuration, RuntimeSettings
from sopp.models.core import Coordinates, FrequencyRange, TimeWindow
from sopp.models.ground.config import CustomTrajectoryConfig
from sopp.models.ground.facility import Facility
from sopp.models.reservation import Reservation


def test_time_window(time_window):
    expected = (
        "TimeWindow:\n"
        "  Begin:              2024-02-08 10:00:00\n"
        "  End:                2024-02-08 12:00:00"
    )
    assert str(time_window) == expected


def test_frequency_range(frequency_range):
    expected = (
        "FrequencyRange:\n  Frequency:          10 MHz\n  Bandwidth:          10 MHz"
    )
    assert str(frequency_range) == expected


def test_runtime_settings(runtime_settings):
    expected = (
        "RuntimeSettings:\n"
        "  Time Interval:      1\n"
        "  Concurrency:        10"
        "  Min. Altitude:      0.0"
    )
    assert str(runtime_settings) == expected


def test_facility(facility):
    expected = (
        "Facility:\n"
        "  Name:               TestFacility\n"
        "  Latitude:           10\n"
        "  Longitude:          10\n"
        "  Elevation:          1000 meters\n"
        "  Beamwidth:          3 degrees"
    )
    assert str(facility) == expected


def test_reservation(reservation):
    expected = f"Reservation:\n{reservation.facility}\n{reservation.time}\n{reservation.frequency}"
    assert str(reservation) == expected


def test_configuration(configuration):
    expected = (
        f"Configuration:\n"
        f"{configuration.reservation}\n"
        f"{configuration.runtime_settings}\n"
        f"Satellites:           1 total"
    )
    assert str(configuration) == expected


@pytest.fixture
def time_window():
    return TimeWindow(begin="2024-02-08 10:00:00", end="2024-02-08 12:00:00")


@pytest.fixture
def frequency_range():
    return FrequencyRange(10, 10)


@pytest.fixture
def runtime_settings():
    return RuntimeSettings(1, 10)


@pytest.fixture
def facility():
    return Facility(Coordinates(10, 10), 3, 1000, "TestFacility")


@pytest.fixture
def reservation(facility, time_window, frequency_range):
    return Reservation(facility, time_window, frequency_range)


@pytest.fixture
def test_satellite(satellite):
    satellite.name = "Test"
    return satellite


@pytest.fixture
def configuration(reservation, antenna_trajectory, runtime_settings, test_satellite):
    return Configuration(
        reservation,
        [test_satellite],
        CustomTrajectoryConfig(antenna_trajectory),
        runtime_settings,
    )
