"""Example: link budget strategies for satellite interference analysis.

Demonstrates the three link budget strategies using synthetic trajectory data:

1. SimpleLinkBudgetStrategy (Tier 1): Worst-case estimate using peak EIRP
   and peak telescope gain. Every trajectory point gets the same gain.

2. PatternLinkBudgetStrategy (Tier 1.5): Uses the telescope's antenna
   pattern to look up realistic gain based on the satellite's off-axis
   angle. Satellites far from boresight get much lower received power.

3. NadirLinkBudgetStrategy (Tier 2): Uses both the satellite's transmitter
   antenna pattern and the telescope's receive pattern. EIRP varies with
   the nadir angle at the satellite.
"""

from datetime import datetime, timedelta

import numpy as np

from sopp.analysis.strategies import (
    NadirLinkBudgetStrategy,
    PatternLinkBudgetStrategy,
    SimpleLinkBudgetStrategy,
)
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

    # --- Tier 2: NadirLinkBudgetStrategy ---
    # Uses both satellite transmitter pattern AND telescope receive pattern.
    # EIRP varies with the nadir angle (how far off-nadir the telescope is
    # from the satellite's perspective).
    sat_tx_pattern = AntennaPattern(
        angles_deg=np.array([0.0, 5.0, 10.0, 30.0, 60.0, 90.0]),
        gains_dbi=np.array([30.0, 28.0, 25.0, 15.0, 5.0, -5.0]),
    )

    # Satellite rises from low elevation to overhead and back down.
    # This changes the nadir angle (and distance) along the pass.
    elevations = np.concatenate(
        [
            np.linspace(10.0, 80.0, n_points // 2),
            np.linspace(80.0, 10.0, n_points - n_points // 2),
        ]
    )
    distances = np.concatenate(
        [
            np.linspace(1500.0, 560.0, n_points // 2),
            np.linspace(560.0, 1500.0, n_points - n_points // 2),
        ]
    )

    sat_traj_t2 = SatelliteTrajectory(
        satellite=Satellite(
            name="EXAMPLE-SAT-T2",
            transmitter=Transmitter(power_dbw=10.0, antenna_pattern=sat_tx_pattern),
        ),
        times=times,
        azimuth=np.linspace(170.0, 190.0, n_points),
        altitude=elevations,
        distance_km=distances,
    )

    strategy_t2 = NadirLinkBudgetStrategy()
    result_t2 = strategy_t2.calculate(sat_traj_t2, ant_traj, facility_tier15, frequency)

    print("\n\n=== Tier 2: NadirLinkBudgetStrategy ===")
    print(f"Satellite TX pattern: {sat_tx_pattern.peak_gain_dbi} dBi peak,")
    print(f"  TX power: 10.0 dBW, peak EIRP: 40.0 dBW\n")

    if result_t2 is not None:
        power = result_t2.interference_level
        eirp = result_t2.metadata["eirp_dbw"]
        nadir = result_t2.metadata["nadir_angle_deg"]
        off_axis = result_t2.metadata["off_axis_deg"]
        gain = result_t2.metadata["gain_rx_dbi"]

        print(f"  Points analyzed: {len(power)}")
        print(f"  Nadir angle range: {nadir.min():.1f} to {nadir.max():.1f} deg")
        print(f"  EIRP range: {eirp.min():.1f} to {eirp.max():.1f} dBW")
        print(f"  RX gain range: {gain.min():.1f} to {gain.max():.1f} dBi")
        print(f"  Power range: {power.min():.1f} to {power.max():.1f} dBW")

        mid = n_points // 2
        print(f"\n  At closest approach (t={mid}s, overhead):")
        print(f"    Nadir angle: {nadir[mid]:.1f} deg (near boresight)")
        print(f"    EIRP:        {eirp[mid]:.1f} dBW")
        print(f"    RX off-axis: {off_axis[mid]:.2f} deg")
        print(f"    RX gain:     {gain[mid]:.1f} dBi")
        print(f"    Power:       {power[mid]:.1f} dBW")

        print(f"\n  At start (t=0s, low elevation):")
        print(f"    Nadir angle: {nadir[0]:.1f} deg (far from boresight)")
        print(f"    EIRP:        {eirp[0]:.1f} dBW")
        print(f"    RX off-axis: {off_axis[0]:.2f} deg")
        print(f"    RX gain:     {gain[0]:.1f} dBi")
        print(f"    Power:       {power[0]:.1f} dBW")

        diff = power[mid] - power[0]
        print(
            f"\n  Difference: {diff:.1f} dB (combines TX pattern + RX pattern + distance)"
        )


if __name__ == "__main__":
    main()
