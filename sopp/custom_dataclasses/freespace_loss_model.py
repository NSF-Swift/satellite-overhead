import numpy as np

class FreeSpaceLossModel:
    def __init__(self, frequency_MHz: float, distance_km: float):
        self.frequency_MHz = frequency_MHz  # Frequency in MHz
        self.distance_km = distance_km  # Distance in kilometers

        # Initialize parameters
        self.atmospheric_attenuation = 0.0  # Placeholder for atmospheric attenuation
        self.water_vapor_attenuation = 0.0  # Placeholder for water vapor attenuation
        # Add more parameters as needed

    def set_atmospheric_attenuation(self, attenuation_dB: float):
        """ Set atmospheric attenuation in dB. """
        self.atmospheric_attenuation = attenuation_dB

    def set_water_vapor_attenuation(self, attenuation_dB: float):
        """ Set water vapor attenuation in dB. """
        self.water_vapor_attenuation = attenuation_dB

    def calculate_free_space_loss(self) -> float:
        """
        Calculate free space loss in dB based on given parameters.

        Returns:
        - float: Free space loss in dB
        """
        # Basic free space loss calculation
        free_space_loss_dB = 20 * np.log10(self.distance_km) + 20 * np.log10(self.frequency_MHz) + 32.45

        # Add additional attenuation factors
        free_space_loss_dB += self.atmospheric_attenuation
        free_space_loss_dB += self.water_vapor_attenuation
        # Add more attenuation factors as needed

        return free_space_loss_dB

    def add_custom_attenuation(self, attenuation_dB: float):
        """
        Add custom attenuation to the free space loss model.

        Args:
        - attenuation_dB (float): Custom attenuation in dB to be added.
        """
        # Example of adding a custom attenuation
        # Modify as per specific requirements
        # For instance, you might add additional factors like rain attenuation, antenna gains, etc.
        pass

# Example usage:
if __name__ == "__main__":
    # Create an instance of FreeSpaceLossModel
    model = FreeSpaceLossModel(frequency_MHz=2000, distance_km=100)

    # Set atmospheric attenuation (example)
    model.set_atmospheric_attenuation(2.5)  # Assume 2.5 dB

    # Set water vapor attenuation (example)
    model.set_water_vapor_attenuation(1.8)  # Assume 1.8 dB

    # Calculate free space loss
    loss_dB = model.calculate_free_space_loss()
    print(f"Free space loss: {loss_dB:.2f} dB")
