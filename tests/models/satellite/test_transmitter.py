"""Tests for Transmitter model."""

import numpy as np
import pytest

from sopp.models.antenna import AntennaPattern
from sopp.models.satellite.transmitter import Transmitter


class TestTransmitterTier1:
    """Tests for Tier 1 (simple EIRP) mode."""

    def test_transmitter_creation(self):
        """Transmitter can be created with EIRP value."""
        tx = Transmitter(eirp_dbw=35.0)
        assert tx.eirp_dbw == 35.0

    def test_transmitter_negative_eirp(self):
        """Transmitter accepts negative EIRP (low power transmitters)."""
        tx = Transmitter(eirp_dbw=-10.0)
        assert tx.eirp_dbw == -10.0

    def test_transmitter_equality(self):
        """Transmitters with same EIRP are equal."""
        tx1 = Transmitter(eirp_dbw=35.0)
        tx2 = Transmitter(eirp_dbw=35.0)
        assert tx1 == tx2

    def test_transmitter_inequality(self):
        """Transmitters with different EIRP are not equal."""
        tx1 = Transmitter(eirp_dbw=35.0)
        tx2 = Transmitter(eirp_dbw=40.0)
        assert tx1 != tx2

    def test_get_eirp_returns_constant(self):
        """Tier 1: get_eirp_dbw returns constant regardless of angle."""
        tx = Transmitter(eirp_dbw=35.0)

        assert tx.get_eirp_dbw(0.0) == 35.0
        assert tx.get_eirp_dbw(10.0) == 35.0
        assert tx.get_eirp_dbw(90.0) == 35.0

    def test_get_eirp_ignores_angle(self):
        """Tier 1: get_eirp_dbw returns same scalar regardless of input angle."""
        tx = Transmitter(eirp_dbw=35.0)

        assert tx.get_eirp_dbw(0.0) == 35.0
        assert tx.get_eirp_dbw(45.0) == 35.0

    def test_peak_eirp(self):
        """Tier 1: peak_eirp_dbw equals eirp_dbw."""
        tx = Transmitter(eirp_dbw=35.0)
        assert tx.peak_eirp_dbw == 35.0


class TestTransmitterTier2:
    """Tests for Tier 2 (power + pattern) mode."""

    @pytest.fixture
    def antenna_pattern(self):
        """Sample antenna pattern for testing."""
        return AntennaPattern(
            angles_deg=np.array([0.0, 5.0, 10.0, 30.0]),
            gains_dbi=np.array([30.0, 25.0, 20.0, 5.0]),
        )

    def test_creation_with_power_and_pattern(self, antenna_pattern):
        """Transmitter can be created with power_dbw and antenna_pattern."""
        tx = Transmitter(power_dbw=10.0, antenna_pattern=antenna_pattern)

        assert tx.power_dbw == 10.0
        assert tx.antenna_pattern is antenna_pattern

    def test_get_eirp_varies_with_angle(self, antenna_pattern):
        """Tier 2: get_eirp_dbw returns P_t + G_t(angle)."""
        tx = Transmitter(power_dbw=10.0, antenna_pattern=antenna_pattern)

        # EIRP = power + gain at angle
        assert tx.get_eirp_dbw(0.0) == 10.0 + 30.0  # 40 dBW at boresight
        assert tx.get_eirp_dbw(5.0) == 10.0 + 25.0  # 35 dBW at 5 deg
        assert tx.get_eirp_dbw(10.0) == 10.0 + 20.0  # 30 dBW at 10 deg

    def test_get_eirp_vectorized(self, antenna_pattern):
        """Tier 2: get_eirp_dbw works with arrays."""
        tx = Transmitter(power_dbw=10.0, antenna_pattern=antenna_pattern)
        angles = np.array([0.0, 5.0, 10.0])

        result = tx.get_eirp_dbw(angles)

        expected = np.array([40.0, 35.0, 30.0])
        np.testing.assert_array_equal(result, expected)

    def test_peak_eirp_uses_pattern_peak(self, antenna_pattern):
        """Tier 2: peak_eirp_dbw is power + pattern peak gain."""
        tx = Transmitter(power_dbw=10.0, antenna_pattern=antenna_pattern)

        # Peak = 10 + 30 = 40 dBW
        assert tx.peak_eirp_dbw == 40.0


class TestTransmitterValidation:
    """Tests for Transmitter validation."""

    def test_requires_some_configuration(self):
        """Transmitter requires either eirp_dbw or power_dbw + pattern."""
        with pytest.raises(ValueError, match="requires either"):
            Transmitter()

    def test_power_without_pattern_fails(self):
        """power_dbw alone (without pattern) is not valid."""
        with pytest.raises(ValueError, match="requires either"):
            Transmitter(power_dbw=10.0)

    def test_pattern_without_power_fails(self):
        """antenna_pattern alone (without power) is not valid."""
        pattern = AntennaPattern(
            angles_deg=np.array([0.0, 10.0]),
            gains_dbi=np.array([30.0, 20.0]),
        )
        with pytest.raises(ValueError, match="requires either"):
            Transmitter(antenna_pattern=pattern)

    def test_eirp_takes_precedence_over_pattern(self):
        """If eirp_dbw is set, it's used even if pattern is also provided."""
        pattern = AntennaPattern(
            angles_deg=np.array([0.0, 10.0]),
            gains_dbi=np.array([30.0, 20.0]),
        )
        tx = Transmitter(eirp_dbw=50.0, power_dbw=10.0, antenna_pattern=pattern)

        # eirp_dbw takes precedence - returns constant
        assert tx.get_eirp_dbw(0.0) == 50.0
        assert tx.get_eirp_dbw(10.0) == 50.0
