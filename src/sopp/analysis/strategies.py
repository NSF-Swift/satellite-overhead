from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sopp.analysis.geometry import (
    calculate_angular_separation,
    calculate_angular_separation_sq,
)
from sopp.analysis.link_budget import free_space_path_loss_db, received_power_dbw
from sopp.models.satellite.trajectory import SatelliteTrajectory

if TYPE_CHECKING:
    from sopp.models.core import FrequencyRange
    from sopp.models.ground.facility import Facility
    from sopp.models.ground.trajectory import AntennaTrajectory


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


class InterferenceStrategy(ABC):
    """Abstract base class for interference detection strategies.

    Each strategy implements a different method of determining whether
    a satellite causes interference and optionally how much.

    The ``calculate`` method receives data that is common to every
    trajectory in a simulation run: the antenna pointing, telescope
    facility, and observation frequency.  Model-specific configuration
    (transmitter characteristics, atmospheric profiles, antenna gain
    patterns, link-budget functions, etc.) should be provided via the
    strategy's ``__init__`` and stored on ``self``.
    """

    @abstractmethod
    def calculate(
        self,
        satellite_trajectory: SatelliteTrajectory,
        antenna_trajectory: AntennaTrajectory,
        facility: Facility,
        frequency: FrequencyRange,
    ) -> InterferenceResult | None:
        """Analyze a single satellite pass for interference.

        Args:
            satellite_trajectory: The satellite's path (times, az, alt, distance).
            antenna_trajectory: Where the antenna is pointing over time.
            facility: Telescope location and parameters.
            frequency: Observation frequency.

        Returns:
            InterferenceResult if interference detected, None otherwise.
        """


class GeometricStrategy(InterferenceStrategy):
    """Binary in/out-of-beam detection.

    Determines interference by checking whether the angular separation
    between the satellite and antenna boresight is less than the beam
    radius. This is the original SOPP behavior.
    """

    def calculate(
        self,
        satellite_trajectory: SatelliteTrajectory,
        antenna_trajectory: AntennaTrajectory,
        facility: Facility,
        frequency: FrequencyRange,
    ) -> InterferenceResult | None:
        ant_az, ant_alt = antenna_trajectory.get_state_at(satellite_trajectory.times)

        sep_sq = calculate_angular_separation_sq(
            az1=satellite_trajectory.azimuth,
            alt1=satellite_trajectory.altitude,
            az2=ant_az,
            alt2=ant_alt,
        )

        mask = sep_sq <= facility.beam_radius**2

        if not np.any(mask):
            return None

        return InterferenceResult(
            trajectory=SatelliteTrajectory(
                satellite=satellite_trajectory.satellite,
                times=satellite_trajectory.times[mask],
                azimuth=satellite_trajectory.azimuth[mask],
                altitude=satellite_trajectory.altitude[mask],
                distance_km=satellite_trajectory.distance_km[mask],
            ),
        )


class SimpleLinkBudgetStrategy(InterferenceStrategy):
    """Simple link budget calculation using peak gains and FSPL.

    Calculates received power at the telescope using the Friis equation:
        P_rx(dBW) = EIRP(dBW) - FSPL(dB) + G_rx(dBi)

    This is a "Tier 1" worst-case estimate assuming:
    - Peak satellite EIRP (from transmitter)
    - Peak telescope gain (boresight)
    - Free space path loss only (no atmospheric effects)

    Unlike GeometricStrategy, this returns results for all trajectory points,
    not just those within the beam. The caller can filter based on power level
    or use in conjunction with geometric filtering.

    Requires:
        facility.peak_gain_dbi must be set.

    Args:
        default_eirp_dbw: Default EIRP to use for satellites that have no
            transmitter configured. If not provided, satellites without a
            transmitter will be silently skipped (returns None, no
            interference detected).

    Raises:
        ValueError: If facility.peak_gain_dbi is not set.
    """

    def __init__(self, default_eirp_dbw: float | None = None):
        self.default_eirp_dbw = default_eirp_dbw

    def calculate(
        self,
        satellite_trajectory: SatelliteTrajectory,
        antenna_trajectory: AntennaTrajectory,
        facility: Facility,
        frequency: FrequencyRange,
    ) -> InterferenceResult | None:
        # Facility gain is required — this is a configuration error, not
        # a "no interference" case.
        if facility.peak_gain_dbi is None:
            raise ValueError(
                "SimpleLinkBudgetStrategy requires facility.peak_gain_dbi to be set."
            )
        gain_dbi = facility.peak_gain_dbi

        # Satellite EIRP: try transmitter, then default.
        # If neither is available, skip this satellite
        satellite = satellite_trajectory.satellite
        if (
            satellite.transmitter is not None
            and satellite.transmitter.eirp_dbw is not None
        ):
            eirp_dbw = satellite.transmitter.eirp_dbw
        elif self.default_eirp_dbw is not None:
            eirp_dbw = self.default_eirp_dbw
        else:
            return None

        # Convert frequency from MHz to Hz
        frequency_hz = frequency.frequency * 1e6

        # Convert distance from km to m
        distance_m = satellite_trajectory.distance_km * 1000.0

        # Calculate received power for all trajectory points
        power_dbw = received_power_dbw(
            eirp_dbw=eirp_dbw,
            distance_m=distance_m,
            frequency_hz=frequency_hz,
            gain_rx_dbi=gain_dbi,
        )

        return InterferenceResult(
            trajectory=satellite_trajectory,
            interference_level=np.asarray(power_dbw),
            level_units="dBW",
            metadata={
                "eirp_dbw": eirp_dbw,
                "gain_dbi": gain_dbi,
                "frequency_mhz": frequency.frequency,
            },
        )


class PatternLinkBudgetStrategy(InterferenceStrategy):
    """Link budget using telescope antenna pattern for realistic receive gain.

    This is a "Tier 1.5" calculation:
    - Uses the telescope's antenna pattern to look up G_rx at the actual
      off-axis angle to the satellite (realistic receive gain)
    - Uses peak satellite EIRP (worst case for transmit side)

    The off-axis angle is calculated as the angular separation between where
    the antenna is pointing and where the satellite is in the sky.

    P_rx(dBW) = EIRP(dBW) - FSPL(dB) + G_rx(off_axis_angle)

    This gives a more realistic estimate than SimpleLinkBudgetStrategy because
    it accounts for how much the satellite is in the telescope's sidelobes
    vs main beam.

    Requires:
        facility.antenna_pattern must be set.

    Args:
        default_eirp_dbw: Default EIRP to use for satellites that have no
            transmitter configured. If not provided, satellites without a
            transmitter will be silently skipped (returns None, no
            interference detected).

    Raises:
        ValueError: If facility.antenna_pattern is not set.
    """

    def __init__(self, default_eirp_dbw: float | None = None):
        self.default_eirp_dbw = default_eirp_dbw

    def calculate(
        self,
        satellite_trajectory: SatelliteTrajectory,
        antenna_trajectory: AntennaTrajectory,
        facility: Facility,
        frequency: FrequencyRange,
    ) -> InterferenceResult | None:
        # Facility antenna pattern is required
        if facility.antenna_pattern is None:
            raise ValueError(
                "PatternLinkBudgetStrategy requires facility.antenna_pattern to be set."
            )

        # Satellite EIRP: try transmitter, then default.
        # If neither is available, skip this satellite
        satellite = satellite_trajectory.satellite
        if (
            satellite.transmitter is not None
            and satellite.transmitter.eirp_dbw is not None
        ):
            eirp_dbw = satellite.transmitter.eirp_dbw
        elif self.default_eirp_dbw is not None:
            eirp_dbw = self.default_eirp_dbw
        else:
            return None

        # Get antenna pointing at each time the satellite was observed
        ant_az, ant_alt = antenna_trajectory.get_state_at(satellite_trajectory.times)

        # Calculate off-axis angle: how far is satellite from where we're pointing?
        off_axis_deg = calculate_angular_separation(
            az1=satellite_trajectory.azimuth,
            alt1=satellite_trajectory.altitude,
            az2=ant_az,
            alt2=ant_alt,
        )

        # Look up gain at each off-axis angle from the antenna pattern
        gain_dbi = facility.antenna_pattern.get_gain(off_axis_deg)

        # Convert frequency from MHz to Hz
        frequency_hz = frequency.frequency * 1e6

        # Convert distance from km to m
        distance_m = satellite_trajectory.distance_km * 1000.0

        # Calculate received power for all trajectory points
        # P_rx = EIRP - FSPL + G_rx
        fspl_db = free_space_path_loss_db(distance_m, frequency_hz)
        power_dbw = eirp_dbw - fspl_db + gain_dbi

        return InterferenceResult(
            trajectory=satellite_trajectory,
            interference_level=np.asarray(power_dbw),
            level_units="dBW",
            metadata={
                "eirp_dbw": eirp_dbw,
                "frequency_mhz": frequency.frequency,
                "off_axis_deg": off_axis_deg,
                "gain_dbi": gain_dbi,
            },
        )
