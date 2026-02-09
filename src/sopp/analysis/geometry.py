import numpy as np
import numpy.typing as npt


def calculate_angular_separation_sq(
    az1: npt.NDArray[np.float64],
    alt1: npt.NDArray[np.float64],
    az2: npt.NDArray[np.float64],
    alt2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Calculates the squared angular separation between two sets of coordinates
    using the Euclidean approximation adjusted for elevation scaling.

    Args:
        az1, alt1: Coordinates of the first object (e.g., Satellite) in degrees.
        az2, alt2: Coordinates of the second object (e.g., Antenna) in degrees.

    Returns:
        The squared angular separation in (degrees^2).
    """
    # Vertical Separation
    delta_el = alt1 - alt2

    # Handle Azimuth Wrap (359 vs 1 degree -> difference is 2)
    raw_delta_az = np.abs(az1 - az2)
    delta_az = np.minimum(raw_delta_az, 360.0 - raw_delta_az)

    # As elevation increases, lines of azimuth converge
    # We scale the azimuth difference by the cosine of the average elevation
    avg_el_rad = np.radians((alt1 + alt2) / 2.0)
    az_scaling = np.cos(avg_el_rad)

    # Pythagorean Distance Squared
    return (delta_el**2) + (delta_az * az_scaling) ** 2


def calculate_angular_separation(
    az1: npt.NDArray[np.float64],
    alt1: npt.NDArray[np.float64],
    az2: npt.NDArray[np.float64],
    alt2: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Calculates the angular separation between two sets of coordinates.

    Args:
        az1, alt1: Coordinates of the first object (e.g., Satellite) in degrees.
        az2, alt2: Coordinates of the second object (e.g., Antenna pointing) in degrees.

    Returns:
        The angular separation in degrees.
    """
    sep_sq = calculate_angular_separation_sq(az1, alt1, az2, alt2)
    return np.sqrt(sep_sq)
