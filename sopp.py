from dataclasses import replace

from satellite_determination.TLE_fetcher.tle_fetcher import TleFetcher
from satellite_determination.config_file import ConfigFile
from satellite_determination.custom_dataclasses.frequency_range.support.get_frequency_data_from_csv import \
    GetFrequencyDataFromCsv
from satellite_determination.custom_dataclasses.position_time import PositionTime
from satellite_determination.custom_dataclasses.satellite.satellite import Satellite
from satellite_determination.main import Main
from satellite_determination.path_finder.observation_path_finder import ObservationPathFinder
from satellite_determination.utilities import get_frequencies_filepath, get_satellites_filepath
from satellite_determination.graph_generator.graph_generator import GraphGenerator


def main():
    print('----------------------------------------------------------------------')
    print('|                                                                    |')
    print('|             Launching Satellite Orbit Preprocessor                 |')
    print('|                                                                    |')
    print('----------------------------------------------------------------------')
    print('Loading reservation parameters from config file...\n')  # make flag to specify config file, default .config
    config_file = ConfigFile()
    reservation = config_file.configuration.reservation

    print('Finding satellite interference events for:\n')
    print('Facility: ', reservation.facility.name, ' at ', reservation.facility.coordinates)
    print('Reservation start time: ', reservation.time.begin)
    print('Reservation end time: ', reservation.time.end)
    print('Observation frequency: ', reservation.frequency.frequency, ' MHz')
    print('\n----------------------------------------------------------------------')

    tle_file = get_satellites_filepath()
    frequency_file = get_frequencies_filepath()
    satellite_list = Satellite.from_tle_file(tlefilepath=tle_file)
    frequency_list = GetFrequencyDataFromCsv(filepath=frequency_file).get()
    satellite_list_with_frequencies = [replace(satellite, frequency=frequency_list.get(satellite.tle_information.satellite_number, []))
                                       for satellite in satellite_list]

    antenna_direction_path = [PositionTime(position=config_file.configuration.static_antenna_position,
                                           time=config_file.configuration.reservation.time.begin)] \
        if config_file.configuration.static_antenna_position \
        else ObservationPathFinder(facility=reservation.facility,
                                   observation_target=config_file.configuration.observation_target,
                                   time_window=reservation.time).calculate_path()

    results = Main(antenna_direction_path=antenna_direction_path,
                   reservation=reservation,
                   satellites=satellite_list_with_frequencies).run()

    print("=======================================================================================\n")
    print('       Found ', len(results.interference_windows), ' instances of satellites crossing the main beam.')
    print('    There are ', len(results.satellites_above_horizon), ' satellites above the horizon during the reservation')
    print("                      Main Beam Interference events: \n")
    print("=======================================================================================\n")
    i = 1
    for window in results.interference_windows:
        print('Interference event #', i, ':')
        print('Satellite:', window.satellite.name)
        print('Satellite enters view: ', window.overhead_time.begin)
        print('Satellite leaves view: ', window.overhead_time.end)
        print('__________________________________________________\n')
        i += 1
    GraphGenerator(search_window_start=reservation.time.begin,
                   search_window_end=reservation.time.end,
                   satellites_above_horizon=results.satellites_above_horizon,
                   interference_windows=results.interference_windows).generate_graph()


if __name__ == '__main__':
    frequencies_filepath = get_satellites_filepath()
    if frequencies_filepath.exists():
        main()
    else:
        TleFetcher().get_tles_celestrak()
        try:
            main()
        finally:
            frequencies_filepath.unlink(missing_ok=True)
