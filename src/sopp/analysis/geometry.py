"""Geometric calculations for satellite interference analysis."""

import numpy as np
import numpy.typing as npt

EARTH_RADIUS_KM = 6371.0


def calculate_nadir_angle(
    elevation_deg: npt.NDArray[np.float64],
    distance_km: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Calculate the nadir angle at the satellite for each trajectory point.

    The nadir angle is the angle at the satellite between the nadir direction
    (toward Earth center) and the direction to the ground observer. For
    nadir-pointing satellite antennas, this equals the transmitter off-axis
    angle.

    Derived from the law of sines on the Earth-center / satellite / observer
    triangle.

    Args:
        elevation_deg: Elevation of the satellite as seen from the ground, in degrees.
        distance_km: Slant range from observer to satellite in km.

    Returns:
        Nadir angle in degrees.
    """
    el_rad = np.radians(elevation_deg)
    R = EARTH_RADIUS_KM

    #          S (satellite)
    #         /B\
    #        /   \
    #  r_sat/     \ d (slant range)
    #      /       \
    #     /         \
    #    /           \
    #   O----- R -----T (telescope, on surface)
    # (Earth center)
    #
    # Why y (angle at T) = 90 + el:
    #
    #         S       el is measured from the horizon up,
    #        / el     but T->O goes 90 deg below the horizon,
    #       T------   so y = el + 90.
    #       | 90
    #       |
    #       O
    y = np.pi / 2 + el_rad  # angle at T = 90 + el

    # Cosine rule: r_sat^2 = R^2 + d^2 - 2*R*d*cos(y)
    r_sat = np.sqrt(R**2 + distance_km**2 - 2.0 * R * distance_km * np.cos(y))

    # Law of sines: sin(y)/r_sat = sin(B)/R, so sin(B) = R*sin(y)/r_sat
    sin_nadir = np.clip(R * np.sin(y) / r_sat, -1.0, 1.0)

    return np.degrees(np.arcsin(sin_nadir))


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
