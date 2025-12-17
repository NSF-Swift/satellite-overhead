import pytest

from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.tle import MeanMotion, TleInformation


class TestSatellite:
    def test_orbits_per_day(self):
        radians_per_minute = 0.004375161567402408
        sat0 = self.sat0
        sat0.tle_information.mean_motion = MeanMotion(0, 0, radians_per_minute)
        assert sat0.orbits_per_day == pytest.approx(1.0, rel=0.01)

    @property
    def sat0(self):
        sat0 = Satellite(
            name="TestSatellite0", tle_information=self.arbitrary_tle_information
        )
        return sat0

    @property
    def arbitrary_tle_information(self):
        return TleInformation(
            argument_of_perigee=0.0,
            drag_coefficient=0.0,
            eccentricity=0.0,
            epoch_days=0.0,
            inclination=0.0,
            mean_anomaly=0.0,
            mean_motion=MeanMotion(0, 0, 0),
            revolution_number=0,
            right_ascension_of_ascending_node=0.0,
            satellite_number=0,
            classification="U",
            international_designator=None,
        )
