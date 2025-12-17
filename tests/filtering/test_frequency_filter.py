import pytest
from sopp.models.core import FrequencyRange
from sopp.models.satellite.satellite import Satellite

from sopp.filtering.presets import filter_frequency


@pytest.fixture
def observation_frequency():
    """The frequency range of the ground station / observation."""
    return FrequencyRange(frequency=135.5, bandwidth=10)


@pytest.fixture
def sat_in_band(tle_information):
    """A satellite strictly inside the observation frequency."""
    return Satellite(
        name="name",
        tle_information=tle_information,
        frequency=[FrequencyRange(frequency=136, bandwidth=10)],
    )


@pytest.fixture
def sat_out_of_band(tle_information):
    """A satellite far outside the observation frequency."""
    return Satellite(
        name="name",
        tle_information=tle_information,
        frequency=[FrequencyRange(frequency=200, bandwidth=10)],
    )


@pytest.fixture
def sat_with_bandwidth(tle_information):
    """A satellite that overlaps via its bandwidth (128 +/- 5)."""
    return Satellite(
        name="name",
        tle_information=tle_information,
        frequency=[FrequencyRange(frequency=128, bandwidth=10)],
    )


@pytest.fixture
def sat_inactive(tle_information):
    """A satellite that overlaps but is marked inactive."""
    return Satellite(
        name="name",
        tle_information=tle_information,
        frequency=[FrequencyRange(frequency=130, bandwidth=10, status="inactive")],
    )


@pytest.fixture
def sat_freq_is_none(tle_information):
    """A satellite with explicit None for frequency fields."""
    return Satellite(
        name="name",
        tle_information=tle_information,
        frequency=[FrequencyRange(frequency=None, bandwidth=None)],
    )


@pytest.fixture
def sat_no_freq_data(tle_information):
    """A satellite with an empty frequency list."""
    return Satellite(name="name", tle_information=tle_information, frequency=[])


# --- Tests ---


def test_single_sat_no_bandwidth(observation_frequency, sat_in_band):
    """Verifies a simple frequency match is preserved."""
    filtered = list(filter(filter_frequency(observation_frequency), [sat_in_band]))
    assert filtered == [sat_in_band]


def test_two_sats_one_out_of_band(observation_frequency, sat_in_band, sat_out_of_band):
    """Verifies that out-of-band satellites are removed."""
    filtered = list(
        filter(filter_frequency(observation_frequency), [sat_in_band, sat_out_of_band])
    )
    assert filtered == [sat_in_band]


def test_single_sat_with_bandwidth(observation_frequency, sat_with_bandwidth):
    """Verifies overlap detection logic works with bandwidth."""
    filtered = list(
        filter(filter_frequency(observation_frequency), [sat_with_bandwidth])
    )
    assert filtered == [sat_with_bandwidth]


def test_inactive_sat(observation_frequency, sat_inactive):
    """Verifies that inactive satellites are filtered out even if they overlap."""
    filtered = list(filter(filter_frequency(observation_frequency), [sat_inactive]))
    assert filtered == []


def test_active_and_inactive_sat(
    observation_frequency, sat_with_bandwidth, sat_inactive
):
    """Verifies mixed active/inactive lists are filtered correctly."""
    filtered = list(
        filter(
            filter_frequency(observation_frequency), [sat_with_bandwidth, sat_inactive]
        )
    )
    assert filtered == [sat_with_bandwidth]


def test_no_frequency_data_sat(observation_frequency, sat_no_freq_data):
    """
    Verifies behavior when satellite has NO frequency data.
    Current logic: Keep it (Assume conservative interference).
    """
    filtered = list(filter(filter_frequency(observation_frequency), [sat_no_freq_data]))
    assert filtered == [sat_no_freq_data]


def test_frequency_data_none(observation_frequency, sat_freq_is_none):
    """
    Verifies behavior when frequency values are explicitly None.
    Current logic: Keep it.
    """
    filtered = list(filter(filter_frequency(observation_frequency), [sat_freq_is_none]))
    assert filtered == [sat_freq_is_none]


def test_observation_frequency_is_none(sat_freq_is_none):
    """
    Verifies behavior when the Observation Filter itself is None.
    Current logic: Return everything (No filter applied).
    """
    filtered = list(filter(filter_frequency(None), [sat_freq_is_none]))
    assert filtered == [sat_freq_is_none]
