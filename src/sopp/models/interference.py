"""Interference analysis result model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sopp.models.satellite.trajectory import SatelliteTrajectory


@dataclass
class InterferenceResult:
    """The output of any interference strategy.

    Wraps a SatelliteTrajectory (the interfering segment) with optional
    quantitative interference data. All strategies produce this same
    structure, making them interchangeable.
    """

    trajectory: SatelliteTrajectory

    # Quantitative interference level (strategy-dependent, may be None).
    # For geometric: None (binary detection only).
    # For gain-based: antenna gain in dB.
    # For link budget: received power in dBW.
    interference_level: np.ndarray | None = None
    level_units: str | None = None
    metadata: dict | None = None
