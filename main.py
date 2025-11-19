from sopp.sopp import Sopp
from sopp.tle_fetcher.tle_fetcher_celestrak import TleFetcherCelestrak
from sopp.builder.configuration_builder import ConfigurationBuilder
from sopp.utilities import (
    get_frequencies_filepath,
    get_satellites_filepath,
    get_default_config_file_filepath,
)
from sopp.satellites_filter.filters import filter_frequency
from sopp.graph_generator.graph_generator import GraphGenerator


def main():
    print('----------------------------------------------------------------------')
    print('|                                                                    |')
    print('|             Launching Satellite Orbit Preprocessor                 |')
    print('|                                                                    |')
    print('----------------------------------------------------------------------')
    print('Loading reservation parameters from config file...\n')  # make flag to specify config file, default .config
    tle_file = get_satellites_filepath()
    frequency_file = get_frequencies_filepath()
    config_file = get_default_config_file_filepath()

    builder = (
        ConfigurationBuilder()
            .set_from_config_file(config_file=config_file)
    )

    configuration = (
        builder
            .set_satellites(tle_file=tle_file, frequency_file=frequency_file)
            .add_filter(filter_frequency(builder.frequency_range))
            .build()
    )

    reservation = configuration.reservation

    print('Finding satellite interference events for:\n')
    print('Facility: ', reservation.facility.name, ' at ', reservation.facility.coordinates)
    print('Reservation start time: ', reservation.time.begin)
    print('Reservation end time: ', reservation.time.end)
    print('Observation frequency: ', reservation.frequency.frequency, ' MHz')
    print('\n----------------------------------------------------------------------')

    sopp = Sopp(configuration)

    interference_windows = sopp.get_satellites_crossing_main_beam()
    satellites_above_horizon = sopp.get_satellites_above_horizon()

    print("=======================================================================================\n")
    print('       Found ', len(interference_windows), ' instances of satellites crossing the main beam.')
    print('    There are ', len(satellites_above_horizon), ' satellites above the horizon during the reservation')
    print("                      Main Beam Interference events: \n")
    print("=======================================================================================\n")

    for i, window in enumerate(interference_windows, start=1):
        print('Interference event #', i, ':')
        print('Satellite:', window.satellite.name)
        print('Satellite enters view: ', window.overhead_time.begin)
        print('Satellite leaves view: ', window.overhead_time.end)
        print('__________________________________________________\n')
    GraphGenerator(search_window_start=reservation.time.begin,
                   search_window_end=reservation.time.end,
                   satellites_above_horizon=satellites_above_horizon,
                   interference_windows=interference_windows).generate_graph()


if __name__ == '__main__':
    satellites_filepath = get_satellites_filepath()

    if not satellites_filepath.exists():
        TleFetcherCelestrak(satellites_filepath).fetch_tles()

    try:
        main()
    finally:
        satellites_filepath.unlink(missing_ok=True)
