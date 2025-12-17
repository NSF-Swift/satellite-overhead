from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from sopp.models.satellite.satellite import Satellite
from sopp.models.core import TimeWindow


@dataclass
class SatelliteTrajectory:
    """
    Represents the computed path (trajectory) of a satellite relative to a facility.

    Attributes:
        satellite (Satellite): The satellite object associated with this trajectory.
        times (np.ndarray): 1D array of datetime objects representing time steps.
        azimuth (np.ndarray): 1D array of azimuth angles in degrees.
        altitude (np.ndarray): 1D array of elevation/altitude angles in degrees.
        distance_km (np.ndarray): 1D array of distances to the satellite in kilometers.
    """

    satellite: Satellite
    times: npt.NDArray[np.object_]
    azimuth: npt.NDArray[np.float64]
    altitude: npt.NDArray[np.float64]
    distance_km: npt.NDArray[np.float64]

    def __len__(self):
        return len(self.times)

    @property
    def overhead_time(self):
        """
        A TimeWindow representing the duration that the satellite is tracked
        within this trajectory (e.g., entering and exiting view).
        Returns None if the trajectory contains no data.
        """
        if len(self.times) == 0:
            return None

        begin = self.times[0]
        end = self.times[-1]
        return TimeWindow(begin=begin, end=end)
