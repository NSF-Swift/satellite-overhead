import math
class Transmitter:
    def __init__(self, power: float = (10 ** (-15/10))*300, frequency: float = 11325):#FIXME: should not be a default value
    #def __init__(self, modulation: str, noise_temperature: float, frequency: float, bandwidth: float,  gain: float, power: float = 10):
        #self.modulation = modulation
        #self.noise_temperature = noise_temperature  # in Kelvin
        self.frequency = frequency  # in Hertz
        #self.bandwidth = bandwidth  # in Hertz
        self.power = power
        #self.gain = gain

    def set_modulation(self, modulation):
        """
        Set the modulation type of the transmitter.
        
        Parameters:
        - modulation: String representing the modulation type (e.g., 'QAM', 'PSK', 'OFDM').
        """
        self.modulation = modulation

    def set_noise_temperature(self, noise_temperature):
        """
        Set the noise temperature of the transmitter.
        
        Parameters:
        - noise_temperature: Noise temperature in Kelvin.
        """
        self.noise_temperature = noise_temperature

    def set_frequency(self, frequency):
        """
        Set the operating frequency of the transmitter.
        
        Parameters:
        - frequency: Frequency in Hertz.
        """
        self.frequency = frequency

    def set_bandwidth(self, bandwidth):
        """
        Set the bandwidth of the transmitter.
        
        Parameters:
        - bandwidth: Bandwidth in Hertz.
        """
        self.bandwidth = bandwidth

    def __str__(self):
        return f"Transmitter Properties: Modulation={self.modulation}, Noise Temp={self.noise_temperature} K, " \
               f"Frequency={self.frequency / 1e9} GHz, Bandwidth={self.bandwidth / 1e6} MHz"

# Example usage:
if __name__ == "__main__":
    # Create an instance of Transmitter
    transmitter = Transmitter()

    # Print initial properties
    print(transmitter)

    # Set new properties
    transmitter.set_modulation('OFDM')
    transmitter.set_noise_temperature(150)
    transmitter.set_frequency(3.5e9)
    transmitter.set_bandwidth(40e6)

    # Print updated properties
    print(transmitter)
