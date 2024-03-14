import pytest

from sopp.custom_dataclasses.satellite.satellite import Satellite
from sopp.custom_dataclasses.satellite.mean_motion import MeanMotion
from sopp.custom_dataclasses.satellite.tle_information import TleInformation
from sopp.satellites_filter.filters import (
    filter_name_contains,
    filter_name_does_not_contain,
    filter_name_is,
    filter_orbit_is,
)


class TestFilters:
    LEO_MEAN_MOTION = 0.00798
    MEO_MEAN_MOTION = 0.066
    GEO_MEAN_MOTION = 0.004375

    def test_name_contains_filter(self):
        expected = [self.sat0, self.sat1]
        actual = list(filter(filter_name_contains('TestSatellite'), self.satellites))
        assert actual == expected

    def test_name_is_none(self):
        expected = [self.sat0, self.sat1, self.sat2, self.sat3]
        actual = list(filter(filter_name_contains(substring=None), self.satellites))
        assert actual == expected

    def test_name_does_not_contain_filter(self):
        expected = [self.sat2, self.sat3]
        actual = list(filter(filter_name_does_not_contain('TestSatellite'), self.satellites))
        assert actual == expected

    def test_name_does_not_contain_is_none(self):
        expected = [self.sat0, self.sat1, self.sat2, self.sat3]
        actual = list(filter(filter_name_does_not_contain(substring=None), self.satellites))
        assert actual == expected

    def test_name_is_filter(self):
        expected =  [self.sat2]
        actual = list(filter(filter_name_is('ISS'), self.satellites))
        assert actual == expected

    def test_name_is_none(self):
        expected = [self.sat0, self.sat1, self.sat2, self.sat3]
        actual = list(filter(filter_name_is(substring=None), self.satellites))
        assert actual == expected

    def test_orbit_is_leo_filter(self):
        expected = [self.sat1, self.sat2]
        actual = list(filter(filter_orbit_is(orbit_type='leo'), self.satellites))
        assert actual == expected

    def test_orbit_is_meo_filter(self):
        expected = [self.sat0]
        actual = list(filter(filter_orbit_is(orbit_type='meo'), self.satellites))
        assert actual == expected

    def test_orbit_is_geo_filter(self):
        expected = [self.sat3]
        actual = list(filter(filter_orbit_is(orbit_type='geo'), self.satellites))
        assert actual == expected

    def test_orbit_is_none(self):
        expected = [self.sat0, self.sat1, self.sat2, self.sat3]
        actual = list(filter(filter_orbit_is(orbit_type=None), self.satellites))
        assert actual == expected

    def test_orbit_is_type_invalid(self):
        with pytest.raises(ValueError) as _:
            filter_orbit_is(orbit_type='error')(None)

    @property
    def sat0(self):
        sat0 = Satellite(name='TestSatellite0')
        sat0.tle_information = self.arbitrary_tle_information
        sat0.tle_information.mean_motion = MeanMotion(0, 0, self.LEO_MEAN_MOTION)
        return sat0

    @property
    def sat1(self):
        sat1 = Satellite(name='TestSatellite1')
        sat1.tle_information = self.arbitrary_tle_information
        sat1.tle_information.mean_motion=MeanMotion(0, 0, self.MEO_MEAN_MOTION)
        return sat1

    @property
    def sat2(self):
        sat2 = Satellite(name='ISS')
        sat2.tle_information = self.arbitrary_tle_information
        sat2.tle_information.mean_motion=MeanMotion(0, 0, self.MEO_MEAN_MOTION)
        return sat2

    @property
    def sat3(self):
        sat3 =Satellite(name='Cosmos')
        sat3.tle_information = self.arbitrary_tle_information
        sat3.tle_information.mean_motion=MeanMotion(0, 0, self.GEO_MEAN_MOTION)
        return sat3

    @property
    def satellites(self):
        return [self.sat0, self.sat1, self.sat2, self.sat3]

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
            classification='U',
            international_designator=None
        )
