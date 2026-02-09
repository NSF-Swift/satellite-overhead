"""Example: link budget strategies for satellite interference analysis.

Demonstrates the two link budget strategies using synthetic trajectory data:

1. SimpleLinkBudgetStrategy (Tier 1): Worst-case estimate using peak EIRP
   and peak telescope gain. Every trajectory point gets the same gain.

2. PatternLinkBudgetStrategy (Tier 1.5): Uses the telescope's antenna
   pattern to look up realistic gain based on the satellite's off-axis
   angle. Satellites far from boresight get much lower received power.
"""

from datetime import datetime, timedelta

import numpy as np

from sopp.analysis.strategies import PatternLinkBudgetStrategy, SimpleLinkBudgetStrategy
from sopp.models.antenna import AntennaPattern
from sopp.models.core import Coordinates, FrequencyRange
from sopp.models.ground.facility import Facility
from sopp.models.ground.trajectory import AntennaTrajectory
from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.trajectory import SatelliteTrajectory
from sopp.models.satellite.transmitter import Transmitter


def main():
    # --- Setup: a satellite passing through the sky ---
    base_time = datetime(2025, 6, 15, 12, 0, 0)
    n_points = 60  # 60 seconds of observation

    times = np.array(
        [base_time + timedelta(seconds=i) for i in range(n_points)],
        dtype=object,
    )

    # Satellite crosses from az=170 to az=190 (passing through az=180)
    # while the telescope is pointed at az=180, alt=45
    sat_traj = SatelliteTrajectory(
        satellite=Satellite(
            name="EXAMPLE-SAT",
            transmitter=Transmitter(eirp_dbw=35.0),
        ),
        times=times,
        azimuth=np.linspace(170.0, 190.0, n_points),
        altitude=np.full(n_points, 45.0),
        distance_km=np.full(n_points, 550.0),
    )

    # Telescope points at a fixed position: az=180, alt=45
    ant_traj = AntennaTrajectory(
        times=times,
        azimuth=np.full(n_points, 180.0),
        altitude=np.full(n_points, 45.0),
    )

    frequency = FrequencyRange(frequency=10000, bandwidth=100)  # 10 GHz

    # --- Tier 1: SimpleLinkBudgetStrategy ---
    # Uses peak gain everywhere (worst case)
    facility_tier1 = Facility(
        coordinates=Coordinates(latitude=40.8, longitude=-121.5),
        beamwidth=1.0,
        name="Example Observatory",
        peak_gain_dbi=60.0,
    )

    strategy_t1 = SimpleLinkBudgetStrategy()
    result_t1 = strategy_t1.calculate(sat_traj, ant_traj, facility_tier1, frequency)

    print("=== Tier 1: SimpleLinkBudgetStrategy ===")
    print(f"Uses peak gain: {facility_tier1.peak_gain_dbi} dBi for all points\n")

    if result_t1 is not None:
        power = result_t1.interference_level
        print(f"  Points analyzed: {len(power)}")
        print(f"  Power range: {power.min():.1f} to {power.max():.1f} dBW")
        print(f"  Power is constant? {np.allclose(power, power[0])}")
        print("  (Only varies with distance, which is constant here)")

    # --- Tier 1.5: PatternLinkBudgetStrategy ---
    # Uses antenna pattern for realistic off-axis gain
    antenna_pattern = AntennaPattern(
        angles_deg=np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 90.0]),
        gains_dbi=np.array([60.0, 57.0, 48.0, 36.0, 20.0, 5.0, -10.0]),
    )

    facility_tier15 = Facility(
        coordinates=Coordinates(latitude=40.8, longitude=-121.5),
        beamwidth=1.0,
        name="Example Observatory",
        antenna_pattern=antenna_pattern,
    )

    strategy_t15 = PatternLinkBudgetStrategy()
    result_t15 = strategy_t15.calculate(sat_traj, ant_traj, facility_tier15, frequency)

    print("\n\n=== Tier 1.5: PatternLinkBudgetStrategy ===")
    print(f"Uses antenna pattern with peak gain: {antenna_pattern.peak_gain_dbi} dBi\n")

    if result_t15 is not None:
        power = result_t15.interference_level
        off_axis = result_t15.metadata["off_axis_deg"]
        gain = result_t15.metadata["gain_dbi"]

        print(f"  Points analyzed: {len(power)}")
        print(f"  Off-axis range: {off_axis.min():.2f} to {off_axis.max():.2f} deg")
        print(f"  Gain range: {gain.min():.1f} to {gain.max():.1f} dBi")
        print(f"  Power range: {power.min():.1f} to {power.max():.1f} dBW")

        # Show a few key points
        mid = n_points // 2
        print(f"\n  At closest approach (t={mid}s, on-axis):")
        print(f"    Off-axis: {off_axis[mid]:.2f} deg")
        print(f"    Gain:     {gain[mid]:.1f} dBi")
        print(f"    Power:    {power[mid]:.1f} dBW")

        print("\n  At start (t=0s, far off-axis):")
        print(f"    Off-axis: {off_axis[0]:.2f} deg")
        print(f"    Gain:     {gain[0]:.1f} dBi")
        print(f"    Power:    {power[0]:.1f} dBW")

        diff = power[mid] - power[0]
        print(f"\n  Difference: {diff:.1f} dB more power when on-axis")


if __name__ == "__main__":
    main()
