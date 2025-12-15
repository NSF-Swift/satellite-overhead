import argparse
import time
from datetime import timedelta

from sopp.config.builder import ConfigurationBuilder
from sopp.io.satellites_loader import SatellitesLoaderFromFiles
from sopp.sopp import Sopp
from sopp.utils.helpers import read_datetime_string_as_utc


def run_benchmark(
    tle_file, limit, duration_hours, method_name="get_satellites_above_horizon"
):
    print(f"\n{'=' * 50}")
    print(f"{' SOPP PERFORMANCE BENCHMARK ':^50}")
    print(f"{'=' * 50}")

    # 1. Load Satellites
    print(f"\n[1/4] Loading satellites from '{tle_file}'...")
    all_satellites = SatellitesLoaderFromFiles(tle_file=tle_file).load_satellites()

    if limit > 0:
        satellites = all_satellites[:limit]
        print(
            f"      -> Selected first {len(satellites)} of {len(all_satellites)} satellites."
        )
    else:
        satellites = all_satellites
        print(f"      -> Using ALL {len(satellites)} satellites.")

    # 2. Setup Configuration
    start_time_str = "2025-11-24T00:00:00.000000"
    start_time = read_datetime_string_as_utc(start_time_str)

    print("\n[2/4] Configuring Simulation...")
    print(f"      Start Time: {start_time}")
    print(f"      Duration:   {duration_hours} hours")

    config_builder = (
        ConfigurationBuilder()
        .set_facility(
            latitude=40.8178049,
            longitude=-121.4695413,
            elevation=986,
            name="HCRO",
            beamwidth=3,
        )
        .set_frequency_range(bandwidth=10, frequency=135)
        .set_time_window(
            begin=start_time,
            end=start_time + timedelta(hours=duration_hours),
        )
        .set_observation_target(altitude=90, azimuth=90)
        .set_runtime_settings(
            concurrency_level=1,  # Force single-thread to measure algorithm speed
            time_resolution_seconds=1,
        )
    )
    # Manually inject the sliced satellite list
    config_builder.satellites = satellites
    config = config_builder.build()

    # 3. Initialize Sopp (Warmup)
    print("\n[3/4] Initializing Engine (Warmup)...")
    t_init_start = time.perf_counter()
    sopp = Sopp(config)
    # Force access to lazy property to trigger Skyfield timescale loading
    _ = sopp.ephemeris_calculator
    t_init_end = time.perf_counter()
    print(f"      -> Warmup complete in {t_init_end - t_init_start:.4f}s")

    # 4. Run Benchmark
    print(f"\n[4/4] Running '{method_name}'...")

    t0 = time.perf_counter()

    # Dynamically call the method requested
    target_method = getattr(sopp, method_name)
    results = target_method()

    t1 = time.perf_counter()

    # 5. Report
    total_time = t1 - t0
    sat_count = len(satellites)
    window_count = len(results)

    # Count total position points calculated if available (validation check)
    total_points = sum(len(w) for w in results) if results else 0

    print(f"\n{'=' * 50}")
    print(f"{' RESULTS ':^50}")
    print(f"{'=' * 50}")
    print(f"{'Metric':<25} | {'Value':<20}")
    print(f"{'-' * 25}-+-{'-' * 20}")
    print(f"{'Satellites Processed':<25} | {sat_count:<20}")
    print(f"{'Windows Found':<25} | {window_count:<20}")
    print(f"{'Total Positions Calc':<25} | {total_points:<20,}")
    print(f"{'Total Time':<25} | {total_time:<20.4f} s")
    print(f"{'Throughput':<25} | {sat_count / total_time:<20.2f} sats/sec")
    print(f"{'Time per Satellite':<25} | {(total_time / sat_count) * 1000:<20.4f} ms")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark SOPP Event Finder Performance"
    )

    parser.add_argument(
        "--tle",
        type=str,
        default="./supplements/satellites.tle",
        help="Path to the TLE file (default: ./supplements/satellites.tle)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Number of satellites to test (default: 0 = ALL)",
    )

    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Simulation duration in hours (default: 24)",
    )

    args = parser.parse_args()

    run_benchmark(tle_file=args.tle, limit=args.limit, duration_hours=args.hours)
