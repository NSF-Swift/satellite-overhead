from dataclasses import dataclass, field


@dataclass
class RuntimeSettings:
    """
    Configuration settings for controlling the execution and precision of the EventFinder.

    Attributes:
        time_resolution_seconds (float): The step size (granularity) of the simulation
            grid in seconds.
            (Default: 1.0)
        concurrency_level (int): The number of parallel processes to spawn for
            calculating satellite windows.
            (Default: 1)
        min_altitude (float): The minimum elevation angle (in degrees) required for
            a satellite to be considered visible or "above the horizon".
            (Default: 0.0)
    """

    time_resolution_seconds: float = field(default=1)
    concurrency_level: int = field(default=1)
    min_altitude: float = field(default=0.0)

    def __str__(self):
        return (
            f"{self.__class__.__name__}:\n"
            f"  Time Interval:      {self.time_resolution_seconds}\n"
            f"  Concurrency:        {self.concurrency_level}"
            f"  Min. Altitude:      {self.min_altitude}"
        )
