import numpy as np

from sopp.healpix  import HealpixGainPattern
from sopp.healpix  import HealpixLoader

from sopp.custom_dataclasses.position import Position
from sopp.custom_dataclasses.frequency_range.frequency_range import FrequencyRange
from sopp.gain_pattern import GainPattern

class Antenna:
    # def __init__(self, gain_pattern: GainPattern):
    def __init__(self, gain_pattern):#: np.ndarray):
    #def __init__(self, frequency_range: FrequencyRange, healpix_gain_pattern: HealpixGainPattern, direction: Position, steering_angle: float, polarization='linear', phased_array=None):
        #self.direction = direction  # Direction of the satellite in general: this is going to be set as directly downwards, i.e. nadir (opposite of zenith)
        #self.steering_angle = steering_angle # Phased array steering, which is going to encapsulate random steering of the beam
        #self.frequency_range = frequency_range  # FrequencyRange object for frequency band
        #self.polarization = polarization  # Polarization type (default: linear)
        #self.phased_array = phased_array  # Phased array properties (optional)
        self.gain_pattern = gain_pattern#gain_pattern
        """
        # Initialize gain pattern with HEALPix if provided
        if healpix_gain is not None:
            self.gain_pattern = HealpixGainPattern(healpix_gain)
        else:
            self.gain_pattern = 5.0  # Default gain pattern if HEALPix not provided
        """
    def get_gain(self, theta, phi):
        """
        Get antenna gain at specific spherical coordinates (theta, phi).
        """
        if isinstance(self.gain_pattern, float):  # Handle default gain pattern case
            return self.gain_pattern
        else:
            return self.gain_pattern.get_gain(theta, phi)

    def set_direction(self, steering_angle, satellite_direction):
        """
        Set the steering angle and satellite direction for the antenna.
        """
        self.direction['steering_angle'] = steering_angle
        self.direction['satellite_direction'] = satellite_direction

    def set_polarization(self, polarization):
        """
        Set the polarization type of the antenna.
        """
        self.polarization = polarization

    def set_phased_array_properties(self, phased_array):
        """
        Set properties specific to a phased array antenna.
        """
        self.phased_array = phased_array

# Example usage:
if __name__ == "__main__":
    # Define a gain pattern function (e.g., cosine pattern)
    def cosine_gain(theta, phi):
        return np.cos(theta)**2

    # Define frequency range
    frequency_range = FrequencyRange(min_frequency=2.3e9, max_frequency=2.5e9)

    # Create an instance of Antenna with additional properties
    phased_array_properties = {
        'element_positions': [(0, 0, 0), (1, 0, 0), (0, 1, 0)],  # Example element positions
        'phase_shifters': [0, np.pi/2, np.pi],  # Example phase shifters
        'beamforming_algorithm': 'Digital',
        # Add more properties as needed
    }
    antenna = Antenna(gain_pattern=cosine_gain, direction={'steering_angle': 30.0, 'satellite_direction': 'North'},
                      frequency_range=frequency_range, polarization='linear', phased_array=phased_array_properties)

    # Example: Calculate gain at specific angles
    theta = np.pi / 4  # 45 degrees elevation
    phi = np.pi / 2   # 90 degrees azimuth
    gain = antenna.get_gain(theta, phi)
    print(f"Gain of the antenna at (theta={theta}, phi={phi}) is {gain}")

    # Example: Set direction
    antenna.set_direction(45.0, 'South')
    print(f"New antenna direction: {antenna.direction}")

    # Example: Set polarization
    antenna.set_polarization('circular')
    print(f"New polarization type: {antenna.polarization}")

    # Example: Set phased array properties
    new_phased_array_properties = {
        'element_positions': [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        'phase_shifters': [0, np.pi/4, np.pi/2],
        'beamforming_algorithm': 'Analog',
    }
    antenna.set_phased_array_properties(new_phased_array_properties)
    print(f"New phased array properties: {antenna.phased_array}")

    # Example usage
if __name__ == "__main__":
    # Example CSV file path
    csv_file = 'antenna_gain_pattern.csv'

    # Create HealpixLoader instance
    healpix_loader = HealpixLoader(csv_file)

    # Load HEALPix gain pattern
    healpix_gain = healpix_loader.load_healpix_gain_pattern()

    # Create Antenna instance with HEALPix gain pattern
    frequency_range = {'min': 5.0, 'max': 6.0}  # Example frequency range
    antenna = Antenna(frequency_range, healpix_gain_pattern=healpix_gain)

    # Example usage: get gain at specific spherical coordinates
    theta_example = 45  # Elevation angle in degrees
    phi_example = 90   # Azimuth angle in degrees
    gain_example = antenna.get_gain(theta_example, phi_example)

    print(f"Gain at (theta={theta_example}°, phi={phi_example}°): {gain_example}")
