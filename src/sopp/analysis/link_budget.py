from __future__ import annotations

import numpy as np


def free_space_path_loss_db(
    distance_m: float | np.ndarray,
    frequency_hz: float,
) -> float | np.ndarray:
    """Calculate free space path loss in dB.

    Uses the standard FSPL formula:
        FSPL(dB) = 20*log10(d) + 20*log10(f) - 147.55

    where:
        d = distance in meters
        f = frequency in Hz
        147.55 = 20*log10(c/(4*pi)) with c = speed of light

    Args:
        distance_m: Distance between transmitter and receiver in meters.
            Can be a scalar or numpy array.
        frequency_hz: Frequency in Hz.

    Returns:
        Path loss in dB (positive value). Same shape as distance_m.
    """
    return 20.0 * np.log10(distance_m) + 20.0 * np.log10(frequency_hz) - 147.55


def received_power_dbw(
    eirp_dbw: float,
    distance_m: float | np.ndarray,
    frequency_hz: float,
    gain_rx_dbi: float,
) -> float | np.ndarray:
    """Calculate received power using the Friis equation.

    Uses the link budget formula:
        P_rx(dBW) = EIRP(dBW) - FSPL(dB) + G_rx(dBi)

    This is a worst-case "Tier 1" estimate assuming:
    - Main beam alignment (peak gains at both ends)
    - No atmospheric losses
    - No polarization mismatch

    Args:
        eirp_dbw: Effective Isotropic Radiated Power in dBW.
        distance_m: Distance between transmitter and receiver in meters.
            Can be a scalar or numpy array.
        frequency_hz: Frequency in Hz.
        gain_rx_dbi: Receiver antenna gain in dBi.

    Returns:
        Received power in dBW. Same shape as distance_m.
    """
    fspl = free_space_path_loss_db(distance_m, frequency_hz)
    return eirp_dbw - fspl + gain_rx_dbi
