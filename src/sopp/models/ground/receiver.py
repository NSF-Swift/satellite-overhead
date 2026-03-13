"""Receiver antenna characteristics for a ground facility."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sopp.models.antenna import AntennaPattern


@dataclass
class Receiver:
    """Receive-side antenna characteristics of a facility.

    Supports three tiers of fidelity:

    Tier 0 (geometric): Set only `beamwidth` for binary in/out-of-beam detection.
    Tier 1 (simple link budget): Set `peak_gain_dbi` for worst-case constant gain.
    Tier 1.5+ (detailed): Set `antenna_pattern` for angle-dependent gain lookup.

    Attributes:
        beamwidth: Beamwidth of the telescope in degrees.
        peak_gain_dbi: Peak antenna gain in dBi (Tier 1). Defaults to None.
        antenna_pattern: Full antenna gain pattern (Tier 1.5+). Defaults to None.
    """

    beamwidth: float = 3.0
    peak_gain_dbi: float | None = None
    antenna_pattern: AntennaPattern | None = None

    def __post_init__(self):
        if self.beamwidth <= 0:
            raise ValueError(f"Beamwidth must be > 0, got: {self.beamwidth}")

    @cached_property
    def beam_radius(self) -> float:
        return self.beamwidth / 2.0

    def __str__(self):
        lines = [
            f"  Beamwidth:          {self.beamwidth} degrees",
        ]
        if self.antenna_pattern is not None:
            lines.append(
                f"  Peak Gain:          {self.antenna_pattern.peak_gain_dbi} dBi (from pattern)"
            )
        elif self.peak_gain_dbi is not None:
            lines.append(f"  Peak Gain:          {self.peak_gain_dbi} dBi")
        return "\n".join(lines)
