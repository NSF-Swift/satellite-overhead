from datetime import timedelta

import numpy as np
import pytest
from tests.conftest import ARBITRARY_ALTITUDE, ARBITRARY_AZIMUTH

from sopp.analysis.strategies import (
    GeometricStrategy,
    InterferenceResult,
    PatternLinkBudgetStrategy,
    SimpleLinkBudgetStrategy,
)
from sopp.models.antenna import AntennaPattern
from sopp.models.core import FrequencyRange
from sopp.models.ground.trajectory import AntennaTrajectory
from sopp.models.satellite.trajectory import SatelliteTrajectory
from sopp.models.satellite.transmitter import Transmitter
from sopp.utils.time import generate_time_grid


def test_geometric_strategy_detects_interference(
    arbitrary_datetime, satellite, facility
):
    """
    Scenario: Pre-computed trajectory at exact antenna position.
    Expected: GeometricStrategy returns an InterferenceResult.
    """
    times = generate_time_grid(arbitrary_datetime, arbitrary_datetime, 1)
    frequency = FrequencyRange(frequency=10, bandwidth=10)

    sat_traj = SatelliteTrajectory(
        satellite=satellite,
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
        distance_km=np.full(len(times), 500.0),
    )

    ant_traj = AntennaTrajectory(
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
    )

    strategy = GeometricStrategy()
    result = strategy.calculate(sat_traj, ant_traj, facility, frequency)

    assert result is not None
    assert isinstance(result, InterferenceResult)
    assert result.trajectory.satellite.name == satellite.name
    assert len(result.trajectory) == len(times)


def test_geometric_strategy_returns_none_outside_beam(
    arbitrary_datetime, satellite, facility
):
    """
    Scenario: Pre-computed trajectory far from antenna pointing.
    Expected: GeometricStrategy returns None.
    """
    times = generate_time_grid(arbitrary_datetime, arbitrary_datetime, 1)
    frequency = FrequencyRange(frequency=10, bandwidth=10)

    sat_traj = SatelliteTrajectory(
        satellite=satellite,
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
        distance_km=np.full(len(times), 500.0),
    )

    offset = facility.beamwidth + 1.0
    ant_traj = AntennaTrajectory(
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH + offset),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
    )

    strategy = GeometricStrategy()
    result = strategy.calculate(sat_traj, ant_traj, facility, frequency)

    assert result is None


def test_geometric_strategy_masks_partial_crossing(
    arbitrary_datetime, satellite, facility
):
    """
    Scenario:
        T=0: Satellite outside beam
        T=1: Satellite inside beam
        T=2: Satellite outside beam
    Expected: Result contains only T=1.
    """
    t_end = arbitrary_datetime + timedelta(seconds=2)
    times = generate_time_grid(arbitrary_datetime, t_end, resolution_seconds=1)
    frequency = FrequencyRange(frequency=10, bandwidth=10)

    sat_traj = SatelliteTrajectory(
        satellite=satellite,
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
        distance_km=np.full(len(times), 500.0),
    )

    # Antenna: far at T=0, aligned at T=1, far at T=2
    offset = facility.beamwidth + 1.0
    ant_traj = AntennaTrajectory(
        times=times,
        azimuth=np.array(
            [
                ARBITRARY_AZIMUTH + offset,
                ARBITRARY_AZIMUTH,
                ARBITRARY_AZIMUTH + offset,
            ]
        ),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
    )

    strategy = GeometricStrategy()
    result = strategy.calculate(sat_traj, ant_traj, facility, frequency)

    assert result is not None
    assert len(result.trajectory) == 1
    assert result.trajectory.times[0] == arbitrary_datetime + timedelta(seconds=1)


def test_interference_result_wraps_trajectory(arbitrary_datetime, satellite, facility):
    """
    Verify InterferenceResult.trajectory is a valid SatelliteTrajectory
    with all expected fields.
    """
    times = generate_time_grid(arbitrary_datetime, arbitrary_datetime, 1)
    frequency = FrequencyRange(frequency=10, bandwidth=10)

    sat_traj = SatelliteTrajectory(
        satellite=satellite,
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
        distance_km=np.full(len(times), 500.0),
    )

    ant_traj = AntennaTrajectory(
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
    )

    strategy = GeometricStrategy()
    result = strategy.calculate(sat_traj, ant_traj, facility, frequency)

    assert result is not None

    traj = result.trajectory
    assert isinstance(traj, SatelliteTrajectory)
    assert traj.satellite is satellite
    np.testing.assert_array_equal(traj.times, times)
    np.testing.assert_allclose(traj.azimuth, np.full(len(times), ARBITRARY_AZIMUTH))
    np.testing.assert_allclose(traj.altitude, np.full(len(times), ARBITRARY_ALTITUDE))
    np.testing.assert_allclose(traj.distance_km, np.full(len(times), 500.0))

    # GeometricStrategy produces no quantitative level
    assert result.interference_level is None
    assert result.level_units is None


# --- SimpleLinkBudgetStrategy Tests ---


def test_simple_link_budget_returns_none_without_eirp(
    arbitrary_datetime, satellite, facility
):
    """Strategy returns None when satellite has no transmitter and no default."""
    times = generate_time_grid(arbitrary_datetime, arbitrary_datetime, 1)
    frequency = FrequencyRange(frequency=10000, bandwidth=100)  # 10 GHz

    # Facility has gain but satellite has no transmitter
    facility.peak_gain_dbi = 60.0

    sat_traj = SatelliteTrajectory(
        satellite=satellite,
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
        distance_km=np.full(len(times), 500.0),
    )

    ant_traj = AntennaTrajectory(
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
    )

    strategy = SimpleLinkBudgetStrategy()
    result = strategy.calculate(sat_traj, ant_traj, facility, frequency)

    assert result is None


def test_simple_link_budget_raises_without_gain(
    arbitrary_datetime, satellite, facility
):
    """Strategy raises ValueError when facility has no peak_gain_dbi."""
    times = generate_time_grid(arbitrary_datetime, arbitrary_datetime, 1)
    frequency = FrequencyRange(frequency=10000, bandwidth=100)

    # Satellite has transmitter but facility has no gain
    satellite.transmitter = Transmitter(eirp_dbw=35.0)

    sat_traj = SatelliteTrajectory(
        satellite=satellite,
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
        distance_km=np.full(len(times), 500.0),
    )

    ant_traj = AntennaTrajectory(
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
    )

    strategy = SimpleLinkBudgetStrategy()
    with pytest.raises(ValueError, match="peak_gain_dbi"):
        strategy.calculate(sat_traj, ant_traj, facility, frequency)


def test_simple_link_budget_calculates_power(arbitrary_datetime, satellite, facility):
    """Strategy calculates received power when EIRP and gain are available."""
    times = generate_time_grid(arbitrary_datetime, arbitrary_datetime, 1)
    frequency = FrequencyRange(frequency=10000, bandwidth=100)  # 10 GHz

    satellite.transmitter = Transmitter(eirp_dbw=35.0)
    facility.peak_gain_dbi = 60.0

    sat_traj = SatelliteTrajectory(
        satellite=satellite,
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
        distance_km=np.full(len(times), 500.0),
    )

    ant_traj = AntennaTrajectory(
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
    )

    strategy = SimpleLinkBudgetStrategy()
    result = strategy.calculate(sat_traj, ant_traj, facility, frequency)

    assert result is not None
    assert result.interference_level is not None
    assert result.level_units == "dBW"
    assert len(result.interference_level) == len(times)
    assert result.metadata["eirp_dbw"] == 35.0
    assert result.metadata["gain_dbi"] == 60.0


def test_simple_link_budget_uses_default_eirp(arbitrary_datetime, satellite, facility):
    """Strategy uses default EIRP when satellite has no transmitter."""
    times = generate_time_grid(arbitrary_datetime, arbitrary_datetime, 1)
    frequency = FrequencyRange(frequency=10000, bandwidth=100)

    # Satellite has no transmitter, but facility has gain set
    facility.peak_gain_dbi = 55.0

    sat_traj = SatelliteTrajectory(
        satellite=satellite,
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
        distance_km=np.full(len(times), 500.0),
    )

    ant_traj = AntennaTrajectory(
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
    )

    strategy = SimpleLinkBudgetStrategy(default_eirp_dbw=30.0)
    result = strategy.calculate(sat_traj, ant_traj, facility, frequency)

    assert result is not None
    assert result.metadata["eirp_dbw"] == 30.0
    assert result.metadata["gain_dbi"] == 55.0


def test_simple_link_budget_power_varies_with_distance(
    arbitrary_datetime, satellite, facility
):
    """Power should decrease as distance increases."""
    times = generate_time_grid(
        arbitrary_datetime, arbitrary_datetime + timedelta(seconds=2), 1
    )
    frequency = FrequencyRange(frequency=10000, bandwidth=100)

    satellite.transmitter = Transmitter(eirp_dbw=35.0)
    facility.peak_gain_dbi = 60.0

    # Distance increases over time: 500, 600, 700 km
    distances = np.array([500.0, 600.0, 700.0])

    sat_traj = SatelliteTrajectory(
        satellite=satellite,
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
        distance_km=distances,
    )

    ant_traj = AntennaTrajectory(
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
    )

    strategy = SimpleLinkBudgetStrategy()
    result = strategy.calculate(sat_traj, ant_traj, facility, frequency)

    assert result is not None
    power = result.interference_level
    # Power should decrease with distance
    assert power[0] > power[1] > power[2]


# --- PatternLinkBudgetStrategy Tests (Tier 1.5) ---


def test_pattern_link_budget_returns_none_without_eirp(
    arbitrary_datetime, satellite, facility
):
    """Strategy returns None when satellite has no transmitter and no default."""
    times = generate_time_grid(arbitrary_datetime, arbitrary_datetime, 1)
    frequency = FrequencyRange(frequency=10000, bandwidth=100)

    # Facility has antenna pattern but satellite has no transmitter
    facility.antenna_pattern = AntennaPattern(
        angles_deg=np.array([0.0, 1.0, 5.0, 10.0, 90.0]),
        gains_dbi=np.array([60.0, 50.0, 30.0, 10.0, -10.0]),
    )

    sat_traj = SatelliteTrajectory(
        satellite=satellite,
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
        distance_km=np.full(len(times), 500.0),
    )

    ant_traj = AntennaTrajectory(
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
    )

    strategy = PatternLinkBudgetStrategy()
    result = strategy.calculate(sat_traj, ant_traj, facility, frequency)

    assert result is None


def test_pattern_link_budget_raises_without_pattern(
    arbitrary_datetime, satellite, facility
):
    """Strategy raises ValueError when facility has no antenna pattern."""
    times = generate_time_grid(arbitrary_datetime, arbitrary_datetime, 1)
    frequency = FrequencyRange(frequency=10000, bandwidth=100)

    # Satellite has transmitter but facility has no antenna pattern
    satellite.transmitter = Transmitter(eirp_dbw=35.0)

    sat_traj = SatelliteTrajectory(
        satellite=satellite,
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
        distance_km=np.full(len(times), 500.0),
    )

    ant_traj = AntennaTrajectory(
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
    )

    strategy = PatternLinkBudgetStrategy()
    with pytest.raises(ValueError, match="antenna_pattern"):
        strategy.calculate(sat_traj, ant_traj, facility, frequency)


def test_pattern_link_budget_calculates_power_on_axis(
    arbitrary_datetime, satellite, facility
):
    """When satellite is at boresight, should use peak gain."""
    times = generate_time_grid(arbitrary_datetime, arbitrary_datetime, 1)
    frequency = FrequencyRange(frequency=10000, bandwidth=100)

    satellite.transmitter = Transmitter(eirp_dbw=35.0)
    facility.antenna_pattern = AntennaPattern(
        angles_deg=np.array([0.0, 1.0, 5.0, 10.0, 90.0]),
        gains_dbi=np.array([60.0, 50.0, 30.0, 10.0, -10.0]),
    )

    # Satellite at same position as antenna pointing (on-axis)
    sat_traj = SatelliteTrajectory(
        satellite=satellite,
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
        distance_km=np.full(len(times), 500.0),
    )

    ant_traj = AntennaTrajectory(
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
    )

    strategy = PatternLinkBudgetStrategy()
    result = strategy.calculate(sat_traj, ant_traj, facility, frequency)

    assert result is not None
    assert result.interference_level is not None
    assert result.level_units == "dBW"
    assert result.metadata["eirp_dbw"] == 35.0
    # On-axis, gain should be peak gain (60 dBi)
    np.testing.assert_allclose(result.metadata["gain_dbi"], 60.0)
    np.testing.assert_allclose(result.metadata["off_axis_deg"], 0.0)


def test_pattern_link_budget_gain_decreases_off_axis(
    arbitrary_datetime, satellite, facility
):
    """Gain should decrease as satellite moves off-axis."""
    times = generate_time_grid(
        arbitrary_datetime, arbitrary_datetime + timedelta(seconds=2), 1
    )
    frequency = FrequencyRange(frequency=10000, bandwidth=100)

    satellite.transmitter = Transmitter(eirp_dbw=35.0)
    facility.antenna_pattern = AntennaPattern(
        angles_deg=np.array([0.0, 1.0, 5.0, 10.0, 90.0]),
        gains_dbi=np.array([60.0, 50.0, 30.0, 10.0, -10.0]),
    )

    # Satellite moves from on-axis to off-axis
    # T=0: on-axis, T=1: 1 degree off, T=2: 2 degrees off
    sat_az = np.array(
        [ARBITRARY_AZIMUTH, ARBITRARY_AZIMUTH + 1.0, ARBITRARY_AZIMUTH + 2.0]
    )

    sat_traj = SatelliteTrajectory(
        satellite=satellite,
        times=times,
        azimuth=sat_az,
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
        distance_km=np.full(len(times), 500.0),  # Same distance for all
    )

    # Antenna stays fixed
    ant_traj = AntennaTrajectory(
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
    )

    strategy = PatternLinkBudgetStrategy()
    result = strategy.calculate(sat_traj, ant_traj, facility, frequency)

    assert result is not None
    power = result.interference_level
    gain = result.metadata["gain_dbi"]
    off_axis = result.metadata["off_axis_deg"]

    # Off-axis angle should increase
    assert off_axis[0] < off_axis[1] < off_axis[2]

    # Gain should decrease as we go off-axis
    assert gain[0] > gain[1] > gain[2]

    # Power should decrease (same distance, lower gain)
    assert power[0] > power[1] > power[2]


def test_pattern_link_budget_uses_default_eirp(arbitrary_datetime, satellite, facility):
    """Strategy uses default EIRP when satellite has no transmitter."""
    times = generate_time_grid(arbitrary_datetime, arbitrary_datetime, 1)
    frequency = FrequencyRange(frequency=10000, bandwidth=100)

    # Satellite has no transmitter
    facility.antenna_pattern = AntennaPattern(
        angles_deg=np.array([0.0, 1.0, 5.0, 10.0, 90.0]),
        gains_dbi=np.array([60.0, 50.0, 30.0, 10.0, -10.0]),
    )

    sat_traj = SatelliteTrajectory(
        satellite=satellite,
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
        distance_km=np.full(len(times), 500.0),
    )

    ant_traj = AntennaTrajectory(
        times=times,
        azimuth=np.full(len(times), ARBITRARY_AZIMUTH),
        altitude=np.full(len(times), ARBITRARY_ALTITUDE),
    )

    strategy = PatternLinkBudgetStrategy(default_eirp_dbw=30.0)
    result = strategy.calculate(sat_traj, ant_traj, facility, frequency)

    assert result is not None
    assert result.metadata["eirp_dbw"] == 30.0
