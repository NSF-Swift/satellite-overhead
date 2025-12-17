import os

from sopp.io.satellites_loader import (
    SatellitesLoaderFromFiles,
)
from sopp.models.frequency_range import FrequencyRange
from sopp.models.satellite.international_designator import (
    InternationalDesignator,
)
from sopp.models.satellite.mean_motion import MeanMotion
from sopp.models.satellite.satellite import Satellite
from sopp.models.satellite.tle_information import TleInformation


class TestSatellitesLoaderFromFiles:
    def test_load_satellites(self):
        test_script_directory = os.path.dirname(os.path.abspath(__file__))

        tle_file = os.path.join(test_script_directory, "satellites.tle")
        frequency_file = os.path.join(
            test_script_directory, "satellite_frequencies.csv"
        )

        satellites = SatellitesLoaderFromFiles(
            tle_file=tle_file, frequency_file=frequency_file
        ).load_satellites()

        assert satellites[0] == Satellite(
            name="ROSEYCUBESAT-1",
            tle_information=TleInformation(
                argument_of_perigee=3.051819898184213,
                drag_coefficient=0.00074327,
                eccentricity=0.0014514,
                epoch_days=26884.2706323,
                inclination=1.699973380722753,
                international_designator=InternationalDesignator(
                    year=23, launch_number=54, launch_piece="AL"
                ),
                mean_anomaly=3.233793161984898,
                mean_motion=MeanMotion(
                    first_derivative=5.987145953152077e-10,
                    second_derivative=0.0,
                    value=0.06663126045136201,
                ),
                revolution_number=1828,
                right_ascension_of_ascending_node=2.0230268439498955,
                satellite_number=56212,
                classification="U",
            ),
            frequency=[
                FrequencyRange(frequency=436.825, bandwidth=None, status="inactive"),
                FrequencyRange(frequency=436.825, bandwidth=None, status="active"),
                FrequencyRange(frequency=436.825, bandwidth=None, status="active"),
            ],
        )

        assert satellites[-1] == Satellite(
            name="INSPIRE-SAT 7",
            tle_information=TleInformation(
                argument_of_perigee=3.03985392083254,
                drag_coefficient=0.00064189,
                eccentricity=0.0014248,
                epoch_days=26884.61562828,
                inclination=1.699962908747241,
                international_designator=InternationalDesignator(
                    year=23, launch_number=54, launch_piece="AK"
                ),
                mean_anomaly=3.2457870646046025,
                mean_motion=MeanMotion(
                    first_derivative=5.111148233097283e-10,
                    second_derivative=0.0,
                    value=0.06661612844674722,
                ),
                revolution_number=1833,
                right_ascension_of_ascending_node=2.0281371679997346,
                satellite_number=56211,
                classification="U",
            ),
            frequency=[
                FrequencyRange(frequency=437.41, bandwidth=None, status="active"),
                FrequencyRange(frequency=435.2, bandwidth=None, status="active"),
            ],
        )

    def test_load_satellites_no_freq_file(self):
        test_script_directory = os.path.dirname(os.path.abspath(__file__))

        tle_file = os.path.join(test_script_directory, "satellites.tle")
        satellites = SatellitesLoaderFromFiles(tle_file=tle_file).load_satellites()

        assert satellites[0] == Satellite(
            name="ROSEYCUBESAT-1",
            tle_information=TleInformation(
                argument_of_perigee=3.051819898184213,
                drag_coefficient=0.00074327,
                eccentricity=0.0014514,
                epoch_days=26884.2706323,
                inclination=1.699973380722753,
                international_designator=InternationalDesignator(
                    year=23, launch_number=54, launch_piece="AL"
                ),
                mean_anomaly=3.233793161984898,
                mean_motion=MeanMotion(
                    first_derivative=5.987145953152077e-10,
                    second_derivative=0.0,
                    value=0.06663126045136201,
                ),
                revolution_number=1828,
                right_ascension_of_ascending_node=2.0230268439498955,
                satellite_number=56212,
                classification="U",
            ),
            frequency=[],
        )

        assert satellites[-1] == Satellite(
            name="INSPIRE-SAT 7",
            tle_information=TleInformation(
                argument_of_perigee=3.03985392083254,
                drag_coefficient=0.00064189,
                eccentricity=0.0014248,
                epoch_days=26884.61562828,
                inclination=1.699962908747241,
                international_designator=InternationalDesignator(
                    year=23, launch_number=54, launch_piece="AK"
                ),
                mean_anomaly=3.2457870646046025,
                mean_motion=MeanMotion(
                    first_derivative=5.111148233097283e-10,
                    second_derivative=0.0,
                    value=0.06661612844674722,
                ),
                revolution_number=1833,
                right_ascension_of_ascending_node=2.0281371679997346,
                satellite_number=56211,
                classification="U",
            ),
            frequency=[],
        )
