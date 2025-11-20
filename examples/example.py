from sopp.config.builder import ConfigurationBuilder
from sopp.satellite_selection.filters import (
    filter_name_does_not_contain,
    filter_orbit_is,
)
from sopp.sopp import Sopp


def main():
    configuration = (
        ConfigurationBuilder()
        .set_facility(
            latitude=40.8178049,
            longitude=-121.4695413,
            elevation=986,
            name="HCRO",
            beamwidth=3,
        )
        .set_frequency_range(bandwidth=10, frequency=135)
        .set_time_window(begin="2024-01-18T08:00:00.0", end="2024-01-18T08:30:00.0")
        .set_observation_target(
            declination="7d24m25.426s", right_ascension="5h55m10.3s"
        )
        .set_runtime_settings(concurrency_level=8, time_continuity_resolution=1)
        # Alternatively set all of the above settings from a config file
        # .set_from_config_file(config_file='./supplements/config.json')
        .set_satellites(tle_file="./supplements/satellites.tle")
        .add_filter(filter_name_does_not_contain("STARLINK"))
        .add_filter(filter_orbit_is(orbit_type="leo"))
        .build()
    )

    # Display configuration
    print("\nFinding satellite interference events for:\n")
    print(f"Facility: {configuration.reservation.facility.name}")
    print(
        f"Location: {configuration.reservation.facility.coordinates} at elevation "
        f"{configuration.reservation.facility.elevation}"
    )
    print(f"Reservation start time: {configuration.reservation.time.begin}")
    print(f"Reservation end time: {configuration.reservation.time.end}")
    print(f"Observation frequency: {configuration.reservation.frequency.frequency} MHz")

    # Determine Satellite Interference
    sopp = Sopp(configuration=configuration)
    interference_events = sopp.get_satellites_crossing_main_beam()

    print("\n==============================================================\n")
    print(
        f"There are {len(interference_events)} satellite interference\n"
        f"events during the reservation\n"
    )
    print("==============================================================\n")

    for i, window in enumerate(interference_events, start=1):
        max_alt = max(window.positions, key=lambda pt: pt.position.altitude)

        print(f"Satellite interference event #{i}:")
        print(f"Satellite: {window.satellite.name}")
        print(
            f"Satellite enters view: {window.overhead_time.begin} at "
            f"{window.positions[0].position.azimuth:.2f} "
            f"Distance: {window.positions[0].position.distance_km:.2f} km"
        )
        print(
            f"Satellite leaves view: {window.overhead_time.end} at "
            f"{window.positions[-1].position.azimuth:.2f} "
            f"Distance: {window.positions[-1].position.distance_km:.2f} km"
        )
        print(f"Satellite maximum altitude: {max_alt.position.altitude:.2f}")
        print("__________________________________________________\n")


if __name__ == "__main__":
    main()
