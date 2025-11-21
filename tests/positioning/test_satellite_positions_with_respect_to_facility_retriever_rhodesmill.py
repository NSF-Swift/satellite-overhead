from dataclasses import replace
from datetime import datetime, timezone

from sopp.models.coordinates import Coordinates
from sopp.models.facility import Facility
from sopp.models.position_time import PositionTime
from sopp.models.satellite.international_designator import (
    InternationalDesignator,
)
from sopp.models.satellite.mean_motion import MeanMotion
from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.tle_information import TleInformation
from sopp.positioning.rhodesmill import (
    SatellitePositionsWithRespectToFacilityRetrieverRhodesmill,
)


class TestSatellitePositionsWithRespectToFacilityRetrieverRhodesmill:
    def test_altitude_can_be_negative(self):
        timestamp = datetime(year=2023, month=6, day=7, tzinfo=timezone.utc)
        facility = Facility(Coordinates(latitude=0, longitude=0))
        position = self._get_satellite_position(facility=facility, timestamp=timestamp)
        assert position.position.altitude < 0

    def test_azimuth_can_be_greater_than_180(self):
        timestamp = datetime(year=2023, month=6, day=7, tzinfo=timezone.utc)
        facility = Facility(Coordinates(latitude=0, longitude=0))
        position = self._get_satellite_position(facility=facility, timestamp=timestamp)
        assert position.position.azimuth > 180

    def test_altitude_decreases_as_elevation_increases(self):
        timestamp = datetime(year=2023, month=6, day=7, tzinfo=timezone.utc)
        facility_where_satellite_has_zero_altitude = Facility(
            Coordinates(latitude=0, longitude=-24.66605)
        )
        same_facility_with_higher_elevation = replace(
            facility_where_satellite_has_zero_altitude, elevation=1000
        )
        position_at_horizon = self._get_satellite_position(
            facility=facility_where_satellite_has_zero_altitude, timestamp=timestamp
        )
        position_with_higher_elevation = self._get_satellite_position(
            facility=same_facility_with_higher_elevation, timestamp=timestamp
        )
        assert (
            position_with_higher_elevation.position.altitude
            < position_at_horizon.position.altitude
        )

    def _get_satellite_position(
        self, facility: Facility, timestamp: datetime
    ) -> PositionTime:
        retriever = SatellitePositionsWithRespectToFacilityRetrieverRhodesmill(
            datetimes=[timestamp], facility=facility
        )

        return retriever.run(self._arbitrary_satellite)[0]

    @property
    def _arbitrary_satellite(self) -> Satellite:
        """
        From 0 COSMOS 1932 DEB
        """
        return Satellite(
            name="ARBITRARY SATELLITE",
            tle_information=TleInformation(
                argument_of_perigee=5.153187590939126,
                drag_coefficient=0.00015211,
                eccentricity=0.0057116,
                epoch_days=26633.28893622,
                inclination=1.1352005427406557,
                international_designator=InternationalDesignator(
                    year=88, launch_number=19, launch_piece="F"
                ),
                mean_anomaly=4.188343400497881,
                mean_motion=MeanMotion(
                    first_derivative=2.363466695408988e-12,
                    second_derivative=0.0,
                    value=0.060298700041442894,
                ),
                revolution_number=95238,
                right_ascension_of_ascending_node=2.907844197528697,
                satellite_number=28275,
                classification="U",
            ),
            frequency=[],
        )
