"""Tests for SatelliteTrajectory pass characterization properties."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.trajectory import SatelliteTrajectory


def _make_satellite(name="TEST SAT"):
    return Satellite(name=name, tle_information=None, frequency=[])


def _make_trajectory(altitude_profile, dt_seconds=10.0, satellite=None):
    """Helper to build a trajectory from just an elevation profile.

    Args:
        altitude_profile: List of elevation values in degrees.
        dt_seconds: Time between samples.
        satellite: Optional Satellite object.
    """
    n = len(altitude_profile)
    start = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    times = np.array([start + timedelta(seconds=i * dt_seconds) for i in range(n)])
    return SatelliteTrajectory(
        satellite=satellite or _make_satellite(),
        times=times,
        azimuth=np.linspace(100, 260, n),
        altitude=np.array(altitude_profile, dtype=np.float64),
        distance_km=np.full(n, 550.0),
    )


class TestPeakProperties:
    def test_peak_index(self):
        traj = _make_trajectory([10, 30, 60, 45, 15])
        assert traj.peak_index == 2

    def test_peak_elevation(self):
        traj = _make_trajectory([10, 30, 60, 45, 15])
        assert traj.peak_elevation == 60.0

    def test_peak_time(self):
        traj = _make_trajectory([10, 30, 60, 45, 15], dt_seconds=10.0)
        expected = datetime(2026, 3, 15, 12, 0, 20, tzinfo=timezone.utc)
        assert traj.peak_time == expected

    def test_peak_with_empty_trajectory(self):
        traj = SatelliteTrajectory(
            satellite=_make_satellite(),
            times=np.array([]),
            azimuth=np.array([]),
            altitude=np.array([]),
            distance_km=np.array([]),
        )
        assert traj.peak_index == 0
        assert traj.peak_elevation == 0.0
        assert traj.peak_time is None


class TestDuration:
    def test_duration(self):
        traj = _make_trajectory([10, 30, 60, 45, 15], dt_seconds=10.0)
        assert traj.duration_seconds == 40.0

    def test_duration_single_point(self):
        traj = _make_trajectory([45])
        assert traj.duration_seconds == 0.0

    def test_duration_empty(self):
        traj = SatelliteTrajectory(
            satellite=_make_satellite(),
            times=np.array([]),
            azimuth=np.array([]),
            altitude=np.array([]),
            distance_km=np.array([]),
        )
        assert traj.duration_seconds == 0.0


class TestRates:
    def test_azimuth_rate_length(self):
        traj = _make_trajectory([10, 30, 60, 45, 15])
        assert len(traj.azimuth_rate) == 4  # n-1

    def test_altitude_rate_length(self):
        traj = _make_trajectory([10, 30, 60, 45, 15])
        assert len(traj.altitude_rate) == 4

    def test_altitude_rate_values(self):
        # Elevation: 0, 10, 20 at 10s intervals → rate = 1.0 deg/sec
        traj = _make_trajectory([0, 10, 20], dt_seconds=10.0)
        np.testing.assert_allclose(traj.altitude_rate, [1.0, 1.0])

    def test_azimuth_rate_values(self):
        # Azimuth goes from 100 to 260 over 5 points at 10s intervals
        # That's 160 deg over 4 intervals of 10s = 4.0 deg/sec each
        traj = _make_trajectory([10, 30, 60, 45, 15], dt_seconds=10.0)
        np.testing.assert_allclose(traj.azimuth_rate, [4.0, 4.0, 4.0, 4.0])

    def test_rates_empty(self):
        traj = SatelliteTrajectory(
            satellite=_make_satellite(),
            times=np.array([]),
            azimuth=np.array([]),
            altitude=np.array([]),
            distance_km=np.array([]),
        )
        assert len(traj.azimuth_rate) == 0
        assert len(traj.altitude_rate) == 0

    def test_rates_single_point(self):
        traj = _make_trajectory([45])
        assert len(traj.azimuth_rate) == 0
        assert len(traj.altitude_rate) == 0


class TestIsComplete:
    def test_complete_pass(self):
        # Symmetric rise-peak-set: peak in the middle
        traj = _make_trajectory([5, 15, 30, 55, 70, 55, 30, 15, 5])
        assert traj.is_complete is True

    def test_incomplete_rising(self):
        # Peak at the end — we caught the rising part only
        traj = _make_trajectory([5, 15, 30, 45, 60])
        assert traj.is_complete is False

    def test_incomplete_setting(self):
        # Peak at the start — we caught the setting part only
        traj = _make_trajectory([60, 45, 30, 15, 5])
        assert traj.is_complete is False

    def test_too_short(self):
        traj = _make_trajectory([30, 45])
        assert traj.is_complete is False

    def test_empty(self):
        traj = SatelliteTrajectory(
            satellite=_make_satellite(),
            times=np.array([]),
            azimuth=np.array([]),
            altitude=np.array([]),
            distance_km=np.array([]),
        )
        assert traj.is_complete is False
