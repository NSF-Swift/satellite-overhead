import pytest

from sopp.models import Configuration, RuntimeSettings
from sopp.models.ground.receiver import Receiver


def test_validate_empty_satellites_list(antenna_config, reservation):
    with pytest.raises(ValueError) as _:
        _ = Configuration(
            satellites=[],
            antenna_config=antenna_config,
            reservation=reservation,
        )


def test_validate_runtime_settings_time_resolution(
    satellite, reservation, antenna_config
):
    runtime_settings = RuntimeSettings(time_resolution_seconds=-1)

    with pytest.raises(ValueError) as _:
        _ = Configuration(
            satellites=[satellite],
            antenna_config=antenna_config,
            reservation=reservation,
            runtime_settings=runtime_settings,
        )


def test_validate_runtime_settings_concurrency(satellite, antenna_config, reservation):
    runtime_settings = RuntimeSettings(concurrency_level=0)
    with pytest.raises(ValueError) as _:
        _ = Configuration(
            satellites=[satellite],
            antenna_config=antenna_config,
            reservation=reservation,
            runtime_settings=runtime_settings,
        )


def test_validate_minimum_altitude(satellite, antenna_config, reservation):
    runtime_settings = RuntimeSettings(min_altitude=-1)
    with pytest.raises(ValueError) as _:
        _ = Configuration(
            satellites=[satellite],
            antenna_config=antenna_config,
            reservation=reservation,
            runtime_settings=runtime_settings,
        )


def test_validate_reservation_time_window(satellite, antenna_config, reservation):
    reservation.time.begin = reservation.time.end
    with pytest.raises(ValueError) as _:
        _ = Configuration(
            satellites=[satellite],
            antenna_config=antenna_config,
            reservation=reservation,
            runtime_settings=RuntimeSettings(),
        )


def test_validate_reservation_beamwidth():
    """Receiver validates beamwidth > 0 at construction time."""
    with pytest.raises(ValueError, match="Beamwidth must be > 0"):
        Receiver(beamwidth=0)


def test_validate_invalid_antenna_config(satellite, reservation):
    antenna_config = None
    with pytest.raises(ValueError) as _:
        _ = Configuration(
            satellites=[satellite],
            antenna_config=antenna_config,
            reservation=reservation,
            runtime_settings=RuntimeSettings(),
        )
