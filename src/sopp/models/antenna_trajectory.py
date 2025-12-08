from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from functools import cached_property


@dataclass
class AntennaTrajectory:
    times: npt.NDArray[np.object_]
    azimuth: npt.NDArray[np.float64]
    altitude: npt.NDArray[np.float64]

    def __len__(self):
        return len(self.times)

    def get_state_at(self, query_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolates the antenna position for the requested query_times.
        """
        # Convert datetimes to float timestamps for interpolation
        query_timestamps = np.array([t.timestamp() for t in query_times])

        interp_az_rad = np.interp(
            query_timestamps, self._ant_timestamps, self._unwrapped_az_rad
        )
        # Convert back to degrees and normalize
        interp_az = np.degrees(interp_az_rad) % 360.0

        interp_alt = np.interp(query_timestamps, self._ant_timestamps, self.altitude)

        return interp_az, interp_alt

    @cached_property
    def _ant_timestamps(self) -> np.ndarray:
        return np.array([t.timestamp() for t in self.times])

    @cached_property
    def _unwrapped_az_rad(self) -> np.ndarray:
        """
        To handle the 360/0 degree azimuth wrapping correctly
        """
        az_rad = np.radians(self.azimuth)
        return np.unwrap(az_rad)
