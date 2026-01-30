from __future__ import annotations

from typing import TYPE_CHECKING

from sopp.analysis.strategies import InterferenceResult
from sopp.models.satellite.trajectory import SatelliteTrajectory

if TYPE_CHECKING:
    from sopp.analysis.strategies import InterferenceStrategy
    from sopp.models.core import FrequencyRange
    from sopp.models.ground.facility import Facility
    from sopp.models.ground.trajectory import AntennaTrajectory


def analyze_interference(
    trajectories: list[SatelliteTrajectory],
    antenna_trajectory: AntennaTrajectory,
    strategy: InterferenceStrategy,
    facility: Facility,
    frequency: FrequencyRange,
) -> list[InterferenceResult]:
    """Apply an interference strategy to pre-computed trajectories.

    This is the primary entry point for strategy-based analysis. It accepts
    trajectories from any source (computed or loaded from disk) and applies
    the given strategy to each one.

    Args:
        trajectories: Pre-computed satellite trajectories.
        antenna_trajectory: Where the antenna is pointing over time.
        strategy: The interference detection strategy to apply.
        facility: Telescope location and parameters.
        frequency: Observation frequency.

    Returns:
        List of InterferenceResult for trajectories where interference
        was detected.
    """
    results = []
    for traj in trajectories:
        result = strategy.calculate(traj, antenna_trajectory, facility, frequency)
        if result is not None:
            results.append(result)
    return results
