from functools import cached_property
from typing import List, Type
from datetime import timedelta

import matplotlib.pyplot as plt


from sopp.event_finder.event_finder import EventFinder
from sopp.event_finder.event_finder_rhodesmill.event_finder_rhodesmill import EventFinderRhodesmill
from sopp.event_finder.event_finder_rhodesmill.power_finder_rhodesmill import PowerFinderRhodesmill
from sopp.custom_dataclasses.configuration import Configuration
from sopp.custom_dataclasses.overhead_window import OverheadWindow
from sopp.custom_dataclasses.satellite.satellite import Satellite
from sopp.custom_dataclasses.runtime_settings import RuntimeSettings
from sopp.custom_dataclasses.reservation import Reservation
from sopp.custom_dataclasses.position_time import PositionTime
from sopp.custom_dataclasses.time_window import TimeWindow
from sopp.custom_dataclasses.power_time import PowerTime
from sopp.custom_dataclasses.power_window import PowerWindow

from sopp.power_summer import sum_power
from sopp.custom_dataclasses.power_array import PowerArray


class PowerSopp:
    def __init__(
        self,
        configuration: Configuration,
        event_finder_class: Type[EventFinder] = PowerFinderRhodesmill
    ):
        self._configuration = configuration
        self._event_finder_class = event_finder_class

    def get_satellites_above_horizon(self) -> List[OverheadWindow]:
        return self._event_finder.get_satellites_above_horizon()

    def get_satellites_crossing_main_beam(self) -> List[OverheadWindow]:
        return self._event_finder.get_satellites_crossing_main_beam()
    
    def get_power_from_sats(self) -> PowerArray:
        return self._event_finder.get_satellite_power()

    @cached_property
    def _event_finder(self) -> EventFinder:
        self._validate_configuration()
        return self._event_finder_class(
            list_of_satellites=self._configuration.satellites,
            reservation=self._configuration.reservation,
            antenna_direction_path=self._configuration.antenna_direction_path,
            runtime_settings=self._configuration.runtime_settings,
        )

    def _validate_configuration(self):
        self._validate_satellites()
        self._validate_runtime_settings()
        self._validate_reservation()
        self._validate_antenna_direction_path()

    def _validate_satellites(self):
        satellites = self._configuration.satellites
        if not satellites:
            raise ValueError('Satellites list empty.')

    def _validate_runtime_settings(self):
        runtime_settings = self._configuration.runtime_settings
        if runtime_settings.time_continuity_resolution < timedelta(seconds=1):
            raise ValueError(f'time_continuity_resolution must be at least 1 second, provided: {runtime_settings.time_continuity_resolution}')
        if runtime_settings.concurrency_level < 1:
            raise ValueError(f'concurrency_level must be at least 1, provided: {runtime_settings.concurrency_level}')
        if runtime_settings.min_altitude < 0.0:
            raise ValueError(f'min_altitude must be non-negative, provided: {runtime_settings.min_altitude}')

    def _validate_reservation(self):
        reservation = self._configuration.reservation
        if reservation.time.begin >= reservation.time.end:
            raise ValueError(f'reservation.time.begin time is later than or equal to end time, provided begin: {reservation.time.begin} end: {reservation.time.end}')
        if reservation.facility.beamwidth <= 0:
            raise ValueError(f'reservation.facility.beamwidth must be greater than 0, provided: {reservation.facility.beamwidth}')

    def _validate_antenna_direction_path(self):
        antenna_direction_path = self._configuration.antenna_direction_path
        if not antenna_direction_path:
            raise ValueError('No antenna direction path provided.')

        for current_time, next_time in zip(antenna_direction_path, antenna_direction_path[1:]):
            if current_time.time >= next_time.time:
                raise ValueError('Times in antenna_direction_path must be increasing.')
"""
# Example usage:
# Create a Configuration instance (replace with actual configuration)
config = Configuration()

# Create an instance of PowerSopp
power_sopp = PowerSopp(config, PowerFinderRhodesmill)
sums = sum_power(power_sopp.get_power_from_sats())
power_array = power_sopp._event_finder.power_array #contains the array of powers for each time, with index 0 corresponding to start time, in seconds

    Plot float values as bars with indices on the x-axis and float values on the y-axis.
    
    Parameters:
    float_values (list or array): A list or array of float values to be plotted.
# Generate the x values (indices)
indices = list(range(len(power_array)))
    
# Create the plot
plt.figure(figsize=(10, 6))
plt.bar(indices, power_array, color='skyblue', edgecolor='black', label='Power Values')
    
# Add title and labels
plt.title('Power Values vs. Time')
plt.xlabel('Time')
plt.ylabel('Power')
    
# Add grid, legend, and show the plot
#plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add grid lines only for y-axis
#plt.legend()
plt.show()
"""







"""""
# Plot power vs. time
sums = sum_power(power_sopp.get_power_from_sats())
times = list(sums.keys())
powers = list(sums.values())


# Plotting
plt.figure(figsize=(10, 6))
plt.plot(times, powers, marker='o', linestyle='-', color='b', label='Power vs Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Power')
plt.title('Power vs Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
"""""