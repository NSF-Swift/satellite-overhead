class Receiver:
    def __init__(self, modulation: str, noise_figure: float, frequency: float, bandwidth: float,
                 antenna_gain: float, system_temperature: float, data_rate=None):
        self.modulation = modulation  # Modulation type
        self.noise_figure = noise_figure  # Noise figure in dB
        self.frequency = frequency  # Operating frequency in Hertz
        self.bandwidth = bandwidth  # Bandwidth in Hertz
        self.antenna_gain = antenna_gain  # Antenna gain in dB
        self.system_temperature = system_temperature  # System noise temperature in Kelvin
        self.data_rate = data_rate  # Data rate in bits per second (optional)

    def set_modulation(self, modulation):
        """
        Set the modulation type of the receiver.
        
        Parameters:
        - modulation: String representing the modulation type (e.g., 'QAM', 'PSK', 'OFDM').
        """
        self.modulation = modulation

    def set_noise_figure(self, noise_figure):
        """
        Set the noise figure of the receiver.
        
        Parameters:
        - noise_figure: Noise figure in dB.
        """
        self.noise_figure = noise_figure

    def set_frequency(self, frequency):
        """
        Set the operating frequency of the receiver.
        
        Parameters:
        - frequency: Frequency in Hertz.
        """
        self.frequency = frequency

    def set_bandwidth(self, bandwidth):
        """
        Set the bandwidth of the receiver.
        
        Parameters:
        - bandwidth: Bandwidth in Hertz.
        """
        self.bandwidth = bandwidth

    def set_antenna_gain(self, antenna_gain):
        """
        Set the antenna gain of the receiver.
        
        Parameters:
        - antenna_gain: Antenna gain in dB.
        """
        self.antenna_gain = antenna_gain

    def set_system_temperature(self, system_temperature):
        """
        Set the system noise temperature of the receiver.
        
        Parameters:
        - system_temperature: System noise temperature in Kelvin.
        """
        self.system_temperature = system_temperature

    def set_data_rate(self, data_rate):
        """
        Set the data rate of the receiver.
        
        Parameters:
        - data_rate: Data rate in bits per second.
        """
        self.data_rate = data_rate

    def __str__(self):
        return f"Receiver Properties: Modulation={self.modulation}, Noise Figure={self.noise_figure} dB, " \
               f"Frequency={self.frequency / 1e9} GHz, Bandwidth={self.bandwidth / 1e6} MHz, " \
               f"Antenna Gain={self.antenna_gain} dB, System Temp={self.system_temperature} K, " \
               f"Data Rate={self.data_rate} bps"

# Example usage:
if __name__ == "__main__":
    # Create an instance of Receiver
    receiver = Receiver()

    # Print initial properties
    print(receiver)

    # Set new properties
    receiver.set_modulation('PSK')
    receiver.set_noise_figure(3)
    receiver.set_frequency(3.5e9)
    receiver.set_bandwidth(40e6)
    receiver.set_antenna_gain(25)
    receiver.set_system_temperature(270)
    receiver.set_data_rate(1e6)  # 1 Mbps

    # Print updated properties
    print(receiver)
