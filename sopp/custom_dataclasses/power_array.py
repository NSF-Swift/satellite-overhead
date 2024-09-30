import numpy as np

class PowerArray:
    def __init__(self, length: int):
        self.length = length
        self.array = np.zeros(length, dtype=np.float64)  # Initialize as a NumPy array of zeros with float type

    def add_power(self, index, power):
        if 0 <= index < self.length:
            self.array[index] += float(power)
            # print("We got here!! index we're adding the power to is: " + str(index) + " the power is" + str(power))
            #print(self.array)
        else:
            raise IndexError("Index out of range")

    def get_power(self, index):
        if 0 <= index < self.length:
            return self.array[index]
        else:
            raise IndexError("Index out of range")

    def __repr__(self):
        return f"PowerArray(length={self.length}, array={self.array})"

# Example usage:
if __name__ == "__main__":
    power_array = PowerArray(5)
    print(power_array)  # Output: PowerArray(length=5, array=[0.0, 0.0, 0.0, 0.0, 0.0])
    
    power_array.set_power(2, 10.5)
    print(power_array.get_power(2))  # Output: 10.5

    # Trying to access an index out of range will raise an IndexError
    # power_array.get_power(10)  # This would raise an IndexError
