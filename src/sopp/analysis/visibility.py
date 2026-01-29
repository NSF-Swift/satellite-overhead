from __future__ import annotations

from typing import TYPE_CHECKING

from sopp.ephemeris.base import EphemerisCalculator
from sopp.models.satellite.trajectory import SatelliteTrajectory

if TYPE_CHECKING:
    from sopp.models.reservation import Reservation
    from sopp.models.satellite.satellite import Satellite


def find_satellites_above_horizon(
    reservation: Reservation,
    satellites: list[Satellite],
    ephemeris_calculator: EphemerisCalculator,
    min_altitude: float = 0.0,
) -> list[SatelliteTrajectory]:
    trajectories = []

    for satellite in satellites:
        time_windows = ephemeris_calculator.calculate_visibility_windows(
            satellite,
            min_altitude,
            reservation.time.begin,
            reservation.time.end,
        )

        if not time_windows:
            continue

        satellite_trajectories = ephemeris_calculator.calculate_trajectories(
            satellite, time_windows
        )

        trajectories.extend(satellite_trajectories)

    return trajectories
