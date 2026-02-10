from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AntennaPattern:
    """Antenna gain as a function of off-axis angle.

    Represents a rotationally symmetric antenna pattern where gain depends
    only on the angular distance from boresight (the pointing direction).

    Attributes:
        angles_deg: Off-axis angles in degrees, must start at 0 (boresight).
            Should be monotonically increasing.
        gains_dbi: Corresponding gain values in dBi. First value is peak gain.
    """

    angles_deg: np.ndarray
    gains_dbi: np.ndarray

    def __post_init__(self):
        """Validate pattern data."""
        self.angles_deg = np.asarray(self.angles_deg, dtype=float)
        self.gains_dbi = np.asarray(self.gains_dbi, dtype=float)

        if len(self.angles_deg) != len(self.gains_dbi):
            raise ValueError("angles_deg and gains_dbi must have same length")
        if len(self.angles_deg) < 2:
            raise ValueError("Pattern must have at least 2 points")
        if self.angles_deg[0] != 0.0:
            raise ValueError("First angle must be 0 (boresight)")

    @property
    def peak_gain_dbi(self) -> float:
        """Peak gain at boresight (0 degrees off-axis)."""
        return float(self.gains_dbi[0])

    def get_gain(self, off_axis_deg: float | np.ndarray) -> float | np.ndarray:
        """Interpolate gain at given off-axis angle(s).

        Args:
            off_axis_deg: Off-axis angle(s) in degrees. Can be scalar or array.

        Returns:
            Gain in dBi at the specified angle(s). Same shape as input.
        """
        return np.interp(off_axis_deg, self.angles_deg, self.gains_dbi)
