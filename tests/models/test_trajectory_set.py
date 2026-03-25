"""Tests for TrajectorySet filtering and selection."""

from datetime import datetime, timedelta, timezone

import numpy as np

from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.trajectory import SatelliteTrajectory
from sopp.models.satellite.trajectory_set import TrajectorySet


def _make_trajectory(
    name="TEST SAT",
    peak_el=50.0,
    peak_time_offset_min=0,
    duration_points=21,
    dt_seconds=10.0,
):
    """Build a symmetric pass trajectory centered on peak_el."""
    n = duration_points
    mid = n // 2
    base = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc) + timedelta(
        minutes=peak_time_offset_min
    )
    times = np.array([base + timedelta(seconds=i * dt_seconds) for i in range(n)])

    el = np.zeros(n)
    for i in range(n):
        el[i] = max(5.0, peak_el - abs(i - mid) * (peak_el - 5.0) / mid)

    return SatelliteTrajectory(
        satellite=Satellite(name=name, tle_information=None, frequency=[]),
        times=times,
        azimuth=np.linspace(100, 260, n),
        altitude=el,
        distance_km=np.full(n, 550.0),
    )


class TestTrajectorySetBasics:
    def test_len(self):
        ts = TrajectorySet([_make_trajectory(), _make_trajectory(name="B")])
        assert len(ts) == 2

    def test_iter(self):
        ts = TrajectorySet([_make_trajectory()])
        items = list(ts)
        assert len(items) == 1

    def test_getitem(self):
        ts = TrajectorySet([_make_trajectory(name="A"), _make_trajectory(name="B")])
        assert ts[0].satellite.name == "A"

    def test_slice_returns_trajectory_set(self):
        ts = TrajectorySet([
            _make_trajectory(name="A"),
            _make_trajectory(name="B"),
            _make_trajectory(name="C"),
        ])
        sliced = ts[0:2]
        assert isinstance(sliced, TrajectorySet)
        assert len(sliced) == 2

    def test_sorted_by_peak_time(self):
        ts = TrajectorySet([
            _make_trajectory(name="LATER", peak_time_offset_min=20),
            _make_trajectory(name="EARLIER", peak_time_offset_min=0),
        ])
        assert ts[0].satellite.name == "EARLIER"

    def test_to_list(self):
        ts = TrajectorySet([_make_trajectory()])
        result = ts.to_list()
        assert isinstance(result, list)
        assert len(result) == 1

    def test_str(self):
        ts = TrajectorySet([_make_trajectory(name="STARLINK-1234 [DTC]")])
        output = str(ts)
        assert "STARLINK-1234" in output

    def test_empty(self):
        ts = TrajectorySet([])
        assert len(ts) == 0
        assert str(ts).count("\n") == 1  # header + separator only


class TestTrajectorySetFilter:
    def test_filter_min_el(self):
        ts = TrajectorySet([
            _make_trajectory(peak_el=30),
            _make_trajectory(peak_el=60),
        ])
        result = ts.filter(min_el=50)
        assert len(result) == 1

    def test_filter_max_el(self):
        ts = TrajectorySet([
            _make_trajectory(peak_el=30),
            _make_trajectory(peak_el=60),
        ])
        result = ts.filter(max_el=50)
        assert len(result) == 1

    def test_filter_complete_only(self):
        complete = _make_trajectory(peak_el=50, duration_points=21)
        incomplete = SatelliteTrajectory(
            satellite=Satellite(name="PARTIAL", tle_information=None, frequency=[]),
            times=np.array([
                datetime(2026, 3, 15, 12, 0, i * 10, tzinfo=timezone.utc)
                for i in range(5)
            ]),
            azimuth=np.array([100, 120, 140, 160, 180]),
            altitude=np.array([40, 50, 60, 70, 80]),
            distance_km=np.full(5, 550.0),
        )
        ts = TrajectorySet([complete, incomplete])
        result = ts.filter(complete_only=True)
        assert len(result) == 1

    def test_filter_by_name(self):
        ts = TrajectorySet([
            _make_trajectory(name="STARLINK-1234 [DTC]"),
            _make_trajectory(name="ONEWEB-100"),
        ])
        result = ts.filter(name="starlink")
        assert len(result) == 1

    def test_filter_chaining(self):
        ts = TrajectorySet([
            _make_trajectory(name="A", peak_el=30),
            _make_trajectory(name="B", peak_el=60),
            _make_trajectory(name="C", peak_el=80),
        ])
        result = ts.filter(min_el=25, max_el=70)
        assert len(result) == 2

    def test_filter_returns_trajectory_set(self):
        ts = TrajectorySet([_make_trajectory()])
        result = ts.filter(min_el=10)
        assert isinstance(result, TrajectorySet)


class TestTrajectorySetSelect:
    def test_selects_with_separation(self):
        ts = TrajectorySet([
            _make_trajectory(name="A", peak_time_offset_min=0),
            _make_trajectory(name="B", peak_time_offset_min=5),
            _make_trajectory(name="C", peak_time_offset_min=20),
            _make_trajectory(name="D", peak_time_offset_min=25),
            _make_trajectory(name="E", peak_time_offset_min=40),
        ])
        selected = ts.select(min_separation_min=14)
        names = [t.satellite.name for t in selected]
        assert "A" in names
        assert "B" not in names
        assert "C" in names
        assert "D" not in names
        assert "E" in names

    def test_select_returns_trajectory_set(self):
        ts = TrajectorySet([_make_trajectory()])
        result = ts.select()
        assert isinstance(result, TrajectorySet)

    def test_filter_then_select(self):
        ts = TrajectorySet([
            _make_trajectory(peak_el=20, peak_time_offset_min=0),
            _make_trajectory(peak_el=50, peak_time_offset_min=20),
        ])
        selected = ts.filter(min_el=30).select()
        assert len(selected) == 1

    def test_empty_select(self):
        ts = TrajectorySet([])
        selected = ts.select()
        assert len(selected) == 0
