from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sopp.models.antenna import AntennaPattern


@dataclass
class Transmitter:
    """RF transmission characteristics of a satellite.

    Supports two modes of operation:

    Tier 1 (simple): Set only `eirp_dbw` for a worst-case constant EIRP.
        This assumes main beam alignment and is useful for quick screening.

    Tier 2 (detailed): Set `power_dbw` and `antenna_pattern` separately.
        This allows calculating angle-dependent EIRP based on where the
        receiver is relative to the satellite's antenna boresight.

    Attributes:
        eirp_dbw: Effective Isotropic Radiated Power in dBW (Tier 1).
            Combined P_t * G_t as a single scalar. Use this for simple
            worst-case estimates.
        power_dbw: Transmitter power in dBW, before antenna gain (Tier 2).
            Use with antenna_pattern for angle-dependent calculations.
        antenna_pattern: Satellite antenna gain pattern (Tier 2).
            If provided with power_dbw, enables angle-dependent EIRP.
    """

    eirp_dbw: float | None = None
    power_dbw: float | None = None
    antenna_pattern: AntennaPattern | None = None

    def __post_init__(self):
        """Validate that at least one mode is configured."""
        has_simple = self.eirp_dbw is not None
        has_detailed = self.power_dbw is not None and self.antenna_pattern is not None

        if not has_simple and not has_detailed:
            raise ValueError(
                "Transmitter requires either eirp_dbw (Tier 1) or "
                "both power_dbw and antenna_pattern (Tier 2)"
            )

    def get_eirp_dbw(
        self, off_axis_deg: float | np.ndarray = 0.0
    ) -> float | np.ndarray:
        """Get EIRP at a given off-axis angle from the satellite's boresight.

        For Tier 1 (eirp_dbw set): Returns constant EIRP regardless of angle.
        For Tier 2 (power_dbw + pattern): Returns P_t + G_t(angle).

        Args:
            off_axis_deg: Angle from satellite antenna boresight in degrees.
                Can be scalar or array. Ignored for Tier 1.

        Returns:
            EIRP in dBW. Scalar if input is scalar, array if input is array.
        """
        if self.eirp_dbw is not None:
            return self.eirp_dbw

        # Tier 2: angle-dependent EIRP (fields validated in __post_init__)
        g_t = self.antenna_pattern.get_gain(off_axis_deg)  # type: ignore[union-attr]
        return self.power_dbw + g_t  # type: ignore[operator]

    @property
    def peak_eirp_dbw(self) -> float:
        """Peak EIRP (at satellite boresight)."""
        if self.eirp_dbw is not None:
            return self.eirp_dbw
        # Tier 2 fields validated in __post_init__
        return self.power_dbw + self.antenna_pattern.peak_gain_dbi  # type: ignore[operator, union-attr]
