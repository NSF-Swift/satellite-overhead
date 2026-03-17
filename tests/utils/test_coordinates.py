"""Tests for coordinate transformation utilities."""

from datetime import datetime, timezone

import numpy as np

from sopp.models.core import Coordinates
from sopp.models.ground.facility import Facility
from sopp.utils.coordinates import altaz_to_radec


def _make_facility():
    return Facility(
        coordinates=Coordinates(latitude=40.8178, longitude=-121.4695),
        elevation=986.0,
        name="HCRO",
    )


class TestAltazToRadec:
    def test_returns_arrays(self):
        facility = _make_facility()
        times = np.array([datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc)])
        az = np.array([180.0])
        el = np.array([45.0])

        ra, dec = altaz_to_radec(az, el, times, facility)
        assert isinstance(ra, np.ndarray)
        assert isinstance(dec, np.ndarray)
        assert len(ra) == 1
        assert len(dec) == 1

    def test_zenith_gives_expected_dec(self):
        """Looking straight up from lat ~40.8 should give dec ~40.8."""
        facility = _make_facility()
        times = np.array([datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc)])
        az = np.array([0.0])
        el = np.array([90.0])

        ra, dec = altaz_to_radec(az, el, times, facility)
        assert abs(dec[0] - facility.coordinates.latitude) < 1.0

    def test_vectorized(self):
        facility = _make_facility()
        times = np.array([
            datetime(2026, 3, 17, 12, 0, i * 10, tzinfo=timezone.utc)
            for i in range(5)
        ])
        az = np.linspace(150, 210, 5)
        el = np.linspace(30, 60, 5)

        ra, dec = altaz_to_radec(az, el, times, facility)
        assert len(ra) == 5
        assert len(dec) == 5
        assert not np.any(np.isnan(ra))
        assert not np.any(np.isnan(dec))

    def test_empty_arrays(self):
        facility = _make_facility()
        ra, dec = altaz_to_radec(
            np.array([]), np.array([]), np.array([]), facility
        )
        assert len(ra) == 0
        assert len(dec) == 0

    def test_south_pointing_gives_negative_dec(self):
        """Looking due south at low elevation from lat ~40 should give negative dec."""
        facility = _make_facility()
        times = np.array([datetime(2026, 3, 17, 12, 0, 0, tzinfo=timezone.utc)])
        az = np.array([180.0])
        el = np.array([10.0])

        ra, dec = altaz_to_radec(az, el, times, facility)
        assert dec[0] < 0
