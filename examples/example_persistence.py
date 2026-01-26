"""Example: saving and loading satellite trajectories.

Demonstrates how to persist trajectory data using SOPP's Arrow file format,
including single trajectory, batch, and time-filtered loading.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from sopp.io import save_trajectories
from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.trajectory import SatelliteTrajectory


def main():
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # --- Create sample trajectories ---
    base_time = datetime(2025, 6, 15, 12, 0, 0)
    n_points = 100

    iss = SatelliteTrajectory(
        satellite=Satellite(name="ISS (ZARYA)"),
        times=np.array(
            [base_time + timedelta(seconds=i) for i in range(n_points)],
            dtype=object,
        ),
        azimuth=np.linspace(45.0, 135.0, n_points),
        altitude=np.linspace(10.0, 45.0, n_points),
        distance_km=np.linspace(2000.0, 1500.0, n_points),
    )

    starlink = SatelliteTrajectory(
        satellite=Satellite(name="STARLINK-1234"),
        times=np.array(
            [base_time + timedelta(seconds=i) for i in range(n_points)],
            dtype=object,
        ),
        azimuth=np.linspace(200.0, 280.0, n_points),
        altitude=np.linspace(15.0, 60.0, n_points),
        distance_km=np.linspace(1800.0, 1200.0, n_points),
    )

    # --- Save a single trajectory ---
    single_path = iss.save(output_dir / "iss_trajectory.arrow")
    print(f"Saved single trajectory to: {single_path}")

    # Save to a directory (auto-generates filename from timestamps)
    auto_path = iss.save(directory=output_dir)
    print(f"Saved with auto filename: {auto_path}")

    # Save with observer metadata
    iss.save(
        output_dir / "iss_with_observer.arrow",
        observer_name="Green Bank Observatory",
        observer_lat=38.4331,
        observer_lon=-79.8397,
    )

    # --- Save multiple trajectories to one file ---
    batch_path = save_trajectories(
        [iss, starlink],
        output_dir / "batch.arrow",
    )
    print(f"Saved batch to: {batch_path}")

    # --- Load trajectories (always returns a list) ---
    loaded = SatelliteTrajectory.load(single_path)
    print(f"\nLoaded {len(loaded)} trajectory from single file")
    print(f"  Satellite: {loaded[0].satellite.name}")
    print(f"  Points: {len(loaded[0])}")
    print(
        f"  Time range: {loaded[0].overhead_time.begin} to {loaded[0].overhead_time.end}"
    )

    loaded_batch = SatelliteTrajectory.load(batch_path)
    print(f"\nLoaded {len(loaded_batch)} trajectories from batch file")
    for traj in loaded_batch:
        print(f"  {traj.satellite.name}: {len(traj)} points")

    # --- Load with time range filter ---
    start = base_time + timedelta(seconds=25)
    end = base_time + timedelta(seconds=75)
    filtered = SatelliteTrajectory.load(batch_path, time_range=(start, end))
    print(f"\nLoaded with time filter ({start} to {end}):")
    for traj in filtered:
        print(f"  {traj.satellite.name}: {len(traj)} points")


if __name__ == "__main__":
    main()
