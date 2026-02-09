"""Tests for link budget calculations."""

import numpy as np

from sopp.analysis.link_budget import free_space_path_loss_db, received_power_dbw


class TestFreeSpacePathLoss:
    """Tests for FSPL calculation."""

    def test_fspl_increases_with_distance(self):
        """Doubling distance adds ~6 dB to path loss."""
        d1 = 1000.0  # 1 km
        d2 = 2000.0  # 2 km
        freq = 10e9  # 10 GHz

        fspl1 = free_space_path_loss_db(d1, freq)
        fspl2 = free_space_path_loss_db(d2, freq)

        # 20*log10(2) = ~6.02 dB
        np.testing.assert_allclose(fspl2 - fspl1, 20 * np.log10(2), rtol=1e-6)

    def test_fspl_increases_with_frequency(self):
        """10x frequency adds 20 dB to path loss."""
        dist = 1000.0  # 1 km
        f1 = 1e9  # 1 GHz
        f2 = 10e9  # 10 GHz

        fspl1 = free_space_path_loss_db(dist, f1)
        fspl2 = free_space_path_loss_db(dist, f2)

        # 20*log10(10) = 20 dB
        np.testing.assert_allclose(fspl2 - fspl1, 20.0, rtol=1e-6)

    def test_fspl_known_value(self):
        """Verify against hand-calculated value.

        At 1 km distance and 10 GHz:
        FSPL = 20*log10(1000) + 20*log10(10e9) - 147.55
             = 60 + 200 - 147.55
             = 112.45 dB
        """
        dist = 1000.0  # 1 km
        freq = 10e9  # 10 GHz

        fspl = free_space_path_loss_db(dist, freq)

        np.testing.assert_allclose(fspl, 112.45, rtol=1e-4)

    def test_fspl_vectorized(self):
        """FSPL works with numpy arrays."""
        distances = np.array([1000.0, 2000.0, 4000.0])
        freq = 10e9

        fspl = free_space_path_loss_db(distances, freq)

        assert isinstance(fspl, np.ndarray)
        assert len(fspl) == 3
        # Each doubling adds ~6 dB
        np.testing.assert_allclose(fspl[1] - fspl[0], 20 * np.log10(2), rtol=1e-6)
        np.testing.assert_allclose(fspl[2] - fspl[1], 20 * np.log10(2), rtol=1e-6)


class TestReceivedPower:
    """Tests for received power calculation."""

    def test_received_power_formula(self):
        """P_rx = EIRP - FSPL + G_rx."""
        eirp = 35.0  # dBW
        dist = 550e3  # 550 km in meters
        freq = 10e9  # 10 GHz
        gain = 60.0  # dBi

        # Hand calculate expected value
        expected_fspl = 20 * np.log10(dist) + 20 * np.log10(freq) - 147.55
        expected_power = eirp - expected_fspl + gain

        power = received_power_dbw(eirp, dist, freq, gain)

        np.testing.assert_allclose(power, expected_power, rtol=1e-6)

    def test_received_power_decreases_with_distance(self):
        """Power decreases ~6 dB when distance doubles."""
        eirp = 35.0
        d1 = 500e3  # 500 km
        d2 = 1000e3  # 1000 km
        freq = 10e9
        gain = 60.0

        p1 = received_power_dbw(eirp, d1, freq, gain)
        p2 = received_power_dbw(eirp, d2, freq, gain)

        # Power should drop by ~6 dB
        np.testing.assert_allclose(p1 - p2, 20 * np.log10(2), rtol=1e-6)

    def test_received_power_vectorized(self):
        """Received power works with distance arrays."""
        eirp = 35.0
        distances = np.array([500e3, 600e3, 700e3])
        freq = 10e9
        gain = 60.0

        power = received_power_dbw(eirp, distances, freq, gain)

        assert isinstance(power, np.ndarray)
        assert len(power) == 3
        # Power should decrease as distance increases
        assert power[0] > power[1] > power[2]
