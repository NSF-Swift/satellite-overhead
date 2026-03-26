"""Observation planning example.

Finds satellite passes above the horizon, filters by elevation and
antenna tracking limits, selects non-overlapping observations, and
plots the result.

Requires a TLE file. Download one with:
    sopp download-tles
"""

from sopp.config.builder import ConfigurationBuilder
from sopp.models.ground.receiver import Receiver
from sopp.plotting import plot_elevation
from sopp.sopp import Sopp


def main():
    config = (
        ConfigurationBuilder()
        .set_facility(
            latitude=40.8178049,
            longitude=-121.4695413,
            elevation=986,
            name="HCRO",
            receiver=Receiver(beamwidth=3),
        )
        .set_frequency_range(bandwidth=10, frequency=135)
        .set_time_window(begin="2025-12-10T08:00:00", end="2025-12-10T09:00:00")
        .set_observation_target(altitude=90, azimuth=0)
        .set_runtime_settings(concurrency_level=8, time_resolution_seconds=10)
        .load_satellites(tle_file="./satellites.tle")
        .build()
    )

    engine = Sopp(config)
    passes = engine.get_satellites_above_horizon()
    print(f"Found {len(passes)} passes above horizon\n")

    # Filter: complete passes, above 25 deg, antenna can track
    observable = passes.filter(
        min_el=25,
        max_el=84,
        complete_only=True,
        max_az_rate=2.0,
        max_el_rate=1.5,
    )
    print(f"Observable passes: {len(observable)}")
    print(observable)

    # Select non-overlapping passes with 14 min separation
    selected = observable.select(min_separation_min=14)
    print(f"\nSelected {len(selected)} observations:")
    print(selected)

    # Plot (requires: pip install "sopp[plotting]")
    plot_elevation(selected)


if __name__ == "__main__":
    main()
