from datetime import datetime, timezone

import pytest

from sopp.models.core import Coordinates, TimeWindow
from sopp.models.ground.facility import Facility
from sopp.models.ground.target import ObservationTarget
from sopp.pointing.skyfield import ObservationPathFinderSkyfield


@pytest.mark.parametrize(
    "begin_str, end_str, expected_altitude, expected_azimuth",
    [
        # Star Rising in East
        ("2023-09-23T05:53:00", "2023-09-23T05:54:00", 0, 90),
        # Star Culminating (Transit) near Zenith/South
        ("2023-09-23T11:53:00", "2023-09-23T11:54:00", 90, 191),
        # Star Setting in West
        ("2023-09-23T17:53:00", "2023-09-23T17:54:00", 0, 270),
    ],
)
def test_observation_path_physics_accuracy(
    begin_str, end_str, expected_altitude, expected_azimuth
):
    """
    Verifies that the Skyfield path finder correctly calculates Az/El
    for a known target (RA 12h, Dec 0) observed from Null Island (Lat 0, Lon 0).
    """
    # 1. Setup
    facility = Facility(
        Coordinates(latitude=0.0, longitude=0.0),
        elevation=0,
        name="Null Island",
    )

    def to_utc(s):
        return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)

    time_window = TimeWindow(begin=to_utc(begin_str), end=to_utc(end_str))

    # Target: Celestial Equator, 12h RA (Autumnal Equinox)
    obs_target = ObservationTarget(declination="0d0m0s", right_ascension="12h0m0s")

    path_finder = ObservationPathFinderSkyfield(facility, obs_target, time_window)

    # 2. Execute
    # Default resolution is fine
    trajectory = path_finder.calculate_path()

    # 3. Verify
    # We check the first point
    actual_alt = trajectory.altitude[0]
    actual_az = trajectory.azimuth[0]

    # Use approximate comparison for floating point physics
    # Tolerance is 1.0 degree to account for atmospheric refraction differences
    # and slight timing/epoch offsets in de421.
    assert actual_alt == pytest.approx(expected_altitude, abs=1.0)
    assert actual_az == pytest.approx(expected_azimuth, abs=1.0)


@pytest.mark.parametrize(
    "declination, right_ascension, expected",
    [
        ("12d15m18s", "12h15m18s", (12.0, 15.0, 18.0)),
        ("12d15m18.5s", "12h15m18.5s", (12.0, 15.0, 18.5)),
        ("-38d6m50.8s", "-38h6m50.8s", (-38.0, 6.0, 50.8)),
    ],
)
def test_coordinate_parsing(declination, right_ascension, expected):
    """
    Verifies that string coordinates (d/m/s) are parsed into tuples correctly.
    """
    obs_target = ObservationTarget(
        declination=declination, right_ascension=right_ascension
    )

    actual_ra = ObservationPathFinderSkyfield.right_ascension_to_skyfield(obs_target)
    actual_dec = ObservationPathFinderSkyfield.declination_to_skyfield(obs_target)

    assert actual_ra == expected
    assert actual_dec == expected
