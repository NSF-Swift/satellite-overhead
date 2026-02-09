import numpy as np
import pytest

from sopp.models.antenna import AntennaPattern


class TestAntennaPattern:
    """Tests for AntennaPattern."""

    def test_creation_from_arrays(self):
        """Pattern can be created from angle and gain arrays."""
        angles = np.array([0.0, 1.0, 5.0, 10.0])
        gains = np.array([60.0, 55.0, 40.0, 20.0])

        pattern = AntennaPattern(angles_deg=angles, gains_dbi=gains)

        assert pattern.peak_gain_dbi == 60.0
        np.testing.assert_array_equal(pattern.angles_deg, angles)
        np.testing.assert_array_equal(pattern.gains_dbi, gains)

    def test_peak_gain_is_boresight(self):
        """Peak gain is the gain at 0 degrees."""
        pattern = AntennaPattern(
            angles_deg=np.array([0.0, 5.0, 10.0]),
            gains_dbi=np.array([65.0, 50.0, 30.0]),
        )

        assert pattern.peak_gain_dbi == 65.0

    def test_get_gain_at_known_angle(self):
        """get_gain returns exact value at known angles."""
        pattern = AntennaPattern(
            angles_deg=np.array([0.0, 5.0, 10.0]),
            gains_dbi=np.array([60.0, 45.0, 30.0]),
        )

        assert pattern.get_gain(0.0) == 60.0
        assert pattern.get_gain(5.0) == 45.0
        assert pattern.get_gain(10.0) == 30.0

    def test_get_gain_interpolates(self):
        """get_gain interpolates between known angles."""
        pattern = AntennaPattern(
            angles_deg=np.array([0.0, 10.0]),
            gains_dbi=np.array([60.0, 40.0]),
        )

        # Midpoint should be average
        assert pattern.get_gain(5.0) == 50.0

    def test_get_gain_vectorized(self):
        """get_gain works with arrays."""
        pattern = AntennaPattern(
            angles_deg=np.array([0.0, 10.0]),
            gains_dbi=np.array([60.0, 40.0]),
        )

        angles = np.array([0.0, 5.0, 10.0])
        gains = pattern.get_gain(angles)

        assert isinstance(gains, np.ndarray)
        np.testing.assert_array_equal(gains, [60.0, 50.0, 40.0])

    def test_validation_requires_matching_lengths(self):
        """Angles and gains must have same length."""
        with pytest.raises(ValueError, match="same length"):
            AntennaPattern(
                angles_deg=np.array([0.0, 5.0]),
                gains_dbi=np.array([60.0]),
            )

    def test_validation_requires_at_least_two_points(self):
        """Pattern must have at least 2 points."""
        with pytest.raises(ValueError, match="at least 2"):
            AntennaPattern(
                angles_deg=np.array([0.0]),
                gains_dbi=np.array([60.0]),
            )

    def test_validation_requires_zero_start(self):
        """First angle must be 0 (boresight)."""
        with pytest.raises(ValueError, match="must be 0"):
            AntennaPattern(
                angles_deg=np.array([1.0, 5.0]),
                gains_dbi=np.array([55.0, 40.0]),
            )
