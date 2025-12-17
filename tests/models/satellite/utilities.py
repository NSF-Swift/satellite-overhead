from sopp.models.core import FrequencyRange
from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.tle import (
    InternationalDesignator,
    MeanMotion,
    TleInformation,
)


def expected_international_space_station_tle_as_satellite_cu() -> Satellite:
    return Satellite(
        name="FAKE ISS (ZARYA)",
        tle_information=TleInformation(
            argument_of_perigee=0.3083420829620822,
            drag_coefficient=3.8792e-05,
            eccentricity=0.0007417,
            epoch_days=25545.69339541,
            inclination=0.9013560935706996,
            international_designator=InternationalDesignator(
                year=98, launch_number=67, launch_piece="A"
            ),
            mean_anomaly=1.4946964807494398,
            mean_motion=MeanMotion(
                first_derivative=5.3450708342326346e-11,
                second_derivative=0.0,
                value=0.06763602333248933,
            ),
            revolution_number=20248,
            right_ascension_of_ascending_node=3.686137125541276,
            satellite_number=25544,
        ),
        frequency=[FrequencyRange(frequency=136.65, bandwidth=None, status="active")],
    )
