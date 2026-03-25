"""A filterable, sortable collection of satellite trajectories."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from sopp.models.satellite.trajectory import SatelliteTrajectory


class TrajectorySet:
    """An iterable, filterable collection of satellite trajectories.

    Returned by ``Sopp.get_satellites_above_horizon()`` and related methods.
    Supports filtering, observation scheduling, and plotting.

    Construct from a list of trajectories::

        ts = TrajectorySet(trajectories)

    Filter, select, and chain::

        selected = ts.filter(min_el=25, complete_only=True).select(min_separation_min=14)
    """

    def __init__(self, trajectories: list[SatelliteTrajectory]):
        self._trajectories = sorted(
            trajectories, key=lambda t: t.peak_time or datetime.min
        )

    def __len__(self):
        return len(self._trajectories)

    def __iter__(self):
        return iter(self._trajectories)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return TrajectorySet(self._trajectories[idx])
        return self._trajectories[idx]

    def __str__(self):
        lines = [
            f"{'Satellite':35s}  {'Peak':>6s}  {'Time':>8s}  {'Az':>6s}  {'Dur':>5s}"
        ]
        lines.append("-" * 70)
        for t in self._trajectories:
            if t.peak_time is None:
                continue
            lines.append(
                f"{t.satellite.name:35s}  "
                f"{t.peak_elevation:5.1f}\u00b0  "
                f"{t.peak_time:%H:%M:%S}  "
                f"{t.azimuth[t.peak_index]:5.1f}\u00b0  "
                f"{t.duration_seconds / 60:4.0f}m"
            )
        return "\n".join(lines)

    def to_list(self) -> list[SatelliteTrajectory]:
        """Return the underlying list of trajectories."""
        return list(self._trajectories)

    def filter(
        self,
        min_el: float | None = None,
        max_el: float | None = None,
        complete_only: bool = False,
        name: str | None = None,
        max_az_rate: float | None = None,
        max_el_rate: float | None = None,
    ) -> TrajectorySet:
        """Return a filtered copy.

        Args:
            min_el: Minimum peak elevation in degrees.
            max_el: Maximum peak elevation in degrees.
            complete_only: Only include complete rise-peak-set passes.
            name: Only include satellites whose name contains this string (case-insensitive).
            max_az_rate: Maximum azimuth rate in deg/sec (antenna slew limit).
            max_el_rate: Maximum elevation rate in deg/sec (antenna slew limit).
        """
        result = self._trajectories
        if min_el is not None:
            result = [t for t in result if t.peak_elevation >= min_el]
        if max_el is not None:
            result = [t for t in result if t.peak_elevation <= max_el]
        if complete_only:
            result = [t for t in result if t.is_complete]
        if name is not None:
            result = [t for t in result if name.lower() in t.satellite.name.lower()]
        if max_az_rate is not None:
            result = [
                t for t in result
                if len(t.azimuth_rate) > 0
                and np.max(np.abs(t.azimuth_rate)) < max_az_rate
            ]
        if max_el_rate is not None:
            result = [
                t for t in result
                if len(t.altitude_rate) > 0
                and np.max(np.abs(t.altitude_rate)) < max_el_rate
            ]
        return TrajectorySet(result)

    def select(self, min_separation_min: float = 14) -> TrajectorySet:
        """Select non-overlapping passes with minimum time separation.

        Walks through passes in time order and picks each one that is
        far enough from the last selected pass. Apply ``filter()`` first
        to control which passes are candidates.

        Args:
            min_separation_min: Minimum minutes between selected pass peaks.
        """
        selected = []
        last_time = None
        sep = timedelta(minutes=min_separation_min)

        for t in self._trajectories:
            if t.peak_time is None:
                continue
            if last_time is not None and (t.peak_time - last_time) < sep:
                continue
            selected.append(t)
            last_time = t.peak_time

        return TrajectorySet(selected)

    def plot(self):
        """Plot elevation vs time for all trajectories.

        Requires matplotlib to be installed.
        """
        from sopp.plotting.passes import plot_trajectories

        plot_trajectories(self)
