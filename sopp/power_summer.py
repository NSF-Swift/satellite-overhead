from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

import matplotlib.pyplot as plt

from sopp.custom_dataclasses.power_window import PowerWindow
from sopp.custom_dataclasses.power_time import PowerTime

### TO BE USED WITH POWER WINDOWS ###

def sum_power(powerwindows: List[PowerWindow]):
    # Dictionary to store accumulated power for each second
    power_sum = defaultdict(float)
    
    # Iterate through each powerwindow
    for powerwindow in powerwindows:
        powertimes = powerwindow.powertimes
        for powertime in powertimes:
            # Extract power_value and time from powertime object
            power_value = powertime.power
            time = powertime.time
            
            # Round the time to the nearest second (if needed), assuming time is in seconds
            second = int(round(time.timestamp()))
            
            # Accumulate power value for this second
            power_sum[second] += power_value
    
    # Convert defaultdict to regular dict if needed
    power_sum = dict(power_sum)
    
    return power_sum



# Extract time and power values for plotting
#times = list(power_sum.keys())
#powers = list(power_sum.values())

# Plotting
#plt.figure(figsize=(10, 6))
#plt.plot(times, powers, marker='o', linestyle='-', color='b', label='Power vs Time')
#plt.xlabel('Time (seconds)')
#plt.ylabel('Power')
#plt.title('Power vs Time')
#plt.grid(True)
#plt.legend()
#plt.tight_layout()
#plt.show()