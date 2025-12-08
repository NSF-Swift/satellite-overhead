import numpy as np
from sopp.utils.geometry import calculate_angular_separation_sq


def test_separation_is_zero_for_identical_points():
    """
    Verifies that comparing a point to itself results in 0 distance.
    """
    az = np.array([10.0, 350.0])
    alt = np.array([45.0, 45.0])

    dist_sq = calculate_angular_separation_sq(az, alt, az, alt)

    # Use allclose for float comparisons
    np.testing.assert_allclose(dist_sq, 0.0)


def test_azimuth_wrap_around_logic():
    """
    Verifies that 359.5 and 0.5 are considered 1.0 degree apart (crossing North),
    rather than 359.0 degrees apart.
    """
    # Scenario: Sat at 359.5, Ant at 0.5.
    # Elevation 0 (Equator) so we don't have to account for cos(el) scaling.
    az1 = np.array([359.5])
    alt1 = np.array([0.0])

    az2 = np.array([0.5])
    alt2 = np.array([0.0])

    dist_sq = calculate_angular_separation_sq(az1, alt1, az2, alt2)

    # Distance is 1.0 degree. Squared is 1.0.
    np.testing.assert_allclose(dist_sq, 1.0)


def test_altitude_separation():
    """
    Verifies simple vertical distance logic.
    """
    # Scenario: Sat 10 deg above Ant.
    az1, alt1 = np.array([100.0]), np.array([50.0])
    az2, alt2 = np.array([100.0]), np.array([40.0])

    dist_sq = calculate_angular_separation_sq(az1, alt1, az2, alt2)

    # Distance 10. Squared 100.
    np.testing.assert_allclose(dist_sq, 100.0)


def test_elevation_scaling_at_high_altitude():
    """
    Tests the "Mercator" correction (cos(elevation)).
    At 60 deg elevation, cos(60) = 0.5.
    Therefore, horizontal degrees should be considered "half as wide".
    """
    # Scenario: 2 degrees apart in Azimuth, but high up.
    az1, alt1 = np.array([10.0]), np.array([60.0])
    az2, alt2 = np.array([12.0]), np.array([60.0])

    dist_sq = calculate_angular_separation_sq(az1, alt1, az2, alt2)

    # Delta Az = 2.0
    # Scale = cos(60) = 0.5
    # Effective Horizontal Dist = 2.0 * 0.5 = 1.0
    # Squared = 1.0
    np.testing.assert_allclose(dist_sq, 1.0)


def test_vectorization_works():
    """
    Ensure it handles arrays of multiple points correctly in one call.
    """
    az1 = np.array([0, 0, 0])
    alt1 = np.array([0, 0, 0])

    az2 = np.array([0, 1, 2])  # Increasing Az distance
    alt2 = np.array([1, 0, 0])  # Mixed with El distance

    dist_sq = calculate_angular_separation_sq(az1, alt1, az2, alt2)

    expected = np.array(
        [
            1.0,  # 0,0 vs 0,1 -> 1 deg vertical -> 1.0^2 = 1.0
            1.0,  # 0,0 vs 1,0 -> 1 deg horizontal @ 0 el -> 1.0^2 = 1.0
            4.0,  # 0,0 vs 2,0 -> 2 deg horizontal @ 0 el -> 2.0^2 = 4.0
        ]
    )

    np.testing.assert_allclose(dist_sq, expected)
