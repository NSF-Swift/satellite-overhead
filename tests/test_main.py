from datetime import datetime, timezone
import pytest
import pytz

from sopp.custom_dataclasses.coordinates import Coordinates
from sopp.custom_dataclasses.facility import Facility
from sopp.custom_dataclasses.frequency_range.frequency_range import FrequencyRange
from sopp.custom_dataclasses.overhead_window import OverheadWindow
from sopp.custom_dataclasses.position import Position
from sopp.custom_dataclasses.position_time import PositionTime
from sopp.custom_dataclasses.reservation import Reservation
from sopp.custom_dataclasses.satellite.international_designator import InternationalDesignator
from sopp.custom_dataclasses.satellite.mean_motion import MeanMotion
from sopp.custom_dataclasses.satellite.satellite import Satellite
from sopp.custom_dataclasses.satellite.tle_information import TleInformation
from sopp.custom_dataclasses.time_window import TimeWindow
from sopp.main import Main, MainResults
from tests.event_finder.event_finder_rhodesmill.definitions import create_overhead_window

def assert_overhead_windows_eq(actual: OverheadWindow, expected: OverheadWindow) -> None:
    assert actual.satellite == expected.satellite
    for actual_position, expected_position in zip(actual.positions, expected.positions):
        assert actual_position.position.altitude == pytest.approx(expected_position.position.altitude, abs=1e-6, rel=1e-6)
        assert actual_position.position.azimuth == pytest.approx(expected_position.position.azimuth, abs=1e-6, rel=1e-6)
        assert actual_position.time == expected_position.time

class TestMain:
    def test_arbitrary_inputs_match_expected_output(self):
        antenna_positions = [PositionTime(position=Position(altitude=32, azimuth=320), time=self._arbitrary_reservation.time.begin)]
        actual = Main(reservation=self._arbitrary_reservation,
                      satellites=self._satellites,
                      antenna_direction_path=antenna_positions).run()

        expected = MainResults(
            satellites_above_horizon=[
                OverheadWindow(
                    satellite=self._satellite_in_mainbeam,
                    positions=[
                        PositionTime(
                            position=Position(altitude=31.92827689000652, azimuth=322.2152123600712),
                            time=datetime(2023, 3, 30, 14, 39, 32, tzinfo=pytz.UTC)
                        ),
                        PositionTime(
                            position=Position(altitude=32.10476096624609, azimuth=321.73184343501606),
                            time=datetime(2023, 3, 30, 14, 39, 33, tzinfo=pytz.UTC)
                        ),
                        PositionTime(
                            position=Position(altitude=32.28029629612362, azimuth=321.24277001092725),
                            time=datetime(2023, 3, 30, 14, 39, 34, tzinfo=pytz.UTC)
                        ),
                        PositionTime(
                            position=Position(altitude=32.45481011166138, azimuth=320.74796378603236),
                            time=datetime(2023, 3, 30, 14, 39, 35, tzinfo=pytz.UTC)
                        )
                    ]
                ),
                OverheadWindow(
                    satellite=self._satellite_inside_frequency_range_and_above_horizon_and_outside_mainbeam,
                    positions=[
                        PositionTime(
                            position=Position(altitude=0.011527751634842421, azimuth=31.169677715036304),
                            time=datetime(2023, 3, 30, 14, 39, 35, tzinfo=pytz.UTC)
                        )
                    ]
                )
            ],
            interference_windows=[
                OverheadWindow(
                    satellite=self._satellite_in_mainbeam,
                    positions=[
                        PositionTime(
                            position=Position(altitude=32.10476096624609, azimuth=321.73184343501606),
                            time=datetime(2023, 3, 30, 14, 39, 33, tzinfo=pytz.UTC)
                        ),
                        PositionTime(
                            position=Position(altitude=32.28029629612362, azimuth=321.24277001092725),
                            time=datetime(2023, 3, 30, 14, 39, 34, tzinfo=pytz.UTC)
                        ),
                        PositionTime(
                            position=Position(altitude=32.45481011166138, azimuth=320.74796378603236),
                            time=datetime(2023, 3, 30, 14, 39, 35, tzinfo=pytz.UTC)
                        )
                    ]
                )
            ]
        )

        assert_overhead_windows_eq(actual.satellites_above_horizon[0], expected.satellites_above_horizon[0])
        assert_overhead_windows_eq(actual.satellites_above_horizon[1], expected.satellites_above_horizon[1])
        assert_overhead_windows_eq(actual.interference_windows[0], expected.interference_windows[0])

    @property
    def _arbitrary_reservation(self) -> Reservation:
        time_window = TimeWindow(begin=datetime(year=2023, month=3, day=30, hour=14, minute=39, second=32, tzinfo=pytz.UTC),
                                 end=datetime(year=2023, month=3, day=30, hour=14, minute=39, second=36, tzinfo=pytz.UTC))
        return Reservation(
            facility=Facility(
                beamwidth=3.5,
                coordinates=Coordinates(latitude=40.8178049, longitude=-121.4695413),
                name='ARBITRARY_1',
            ),
            time=time_window,
            frequency=FrequencyRange(
                frequency=135,
                bandwidth=10
            )
        )

    @property
    def _satellites(self):
        return [
            self._satellite_in_mainbeam,
            self._satellite_inside_frequency_range_and_above_horizon_and_outside_mainbeam,
            self._satellite_inside_frequency_range_and_below_horizon,
            self._satellite_outside_frequency_range
        ]
    
    @property
    def _satellite_in_mainbeam(self) -> Satellite:
        return Satellite(name='LILACSAT-2',
                         tle_information=TleInformation(argument_of_perigee=5.179163326196557,
                                                        drag_coefficient=0.00020184,
                                                        eccentricity=0.0012238,
                                                        epoch_days=26801.52502783,
                                                        inclination=1.7021271170197139,
                                                        international_designator=InternationalDesignator(year=15,
                                                                                                         launch_number=49,
                                                                                                         launch_piece='K'),
                                                        mean_anomaly=1.1039888197272412,
                                                        mean_motion=MeanMotion(first_derivative=1.2756659984194665e-10,
                                                                               second_derivative=0.0,
                                                                               value=0.06629635188282393),
                                                        revolution_number=42329,
                                                        right_ascension_of_ascending_node=2.4726638018364304,
                                                        satellite_number=40908,
                                                        classification='U'),
                         frequency=[FrequencyRange(frequency=437.2, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=437.225, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=437.2, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=437.2, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=144.39, bandwidth=None, status='active')])

    @property
    def _satellite_inside_frequency_range_and_above_horizon_and_outside_mainbeam(self) -> Satellite:
        return Satellite(name='NOAA 15',
                         tle_information=TleInformation(argument_of_perigee=0.8036979406046088,
                                                        drag_coefficient=0.00010892000000000001,
                                                        eccentricity=0.0011139, epoch_days=26801.4833696,
                                                        inclination=1.7210307781480645,
                                                        international_designator=InternationalDesignator(
                                                            year=98, launch_number=30, launch_piece='A'),
                                                        mean_anomaly=5.4831490673456615,
                                                        mean_motion=MeanMotion(
                                                            first_derivative=6.6055864051174274e-12,
                                                            second_derivative=0.0,
                                                            value=0.06223475712876591),
                                                        revolution_number=30102,
                                                        right_ascension_of_ascending_node=2.932945522830879,
                                                        satellite_number=25338, classification='U'),
                         frequency=[FrequencyRange(frequency=137.62, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=137.5, bandwidth=None, status='inactive'),
                                    FrequencyRange(frequency=137.77, bandwidth=None, status='inactive'),
                                    FrequencyRange(frequency=1544.5, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=1702.5, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=465.9875, bandwidth=None, status='invalid'),
                                    FrequencyRange(frequency=137.35, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=2247.5, bandwidth=None, status='active')])

    @property
    def _satellite_inside_frequency_range_and_below_horizon(self) -> Satellite:
        return Satellite(name='ISS (ZARYA)', 
                         tle_information=TleInformation(argument_of_perigee=6.2680236319675116, 
                                                        drag_coefficient=0.00018991,
                                                        eccentricity=0.0006492, 
                                                        epoch_days=26801.40295236, 
                                                        inclination=0.9012601004618398,
                                                        international_designator=InternationalDesignator(year=98, 
                                                                                                         launch_number=67, 
                                                                                                         launch_piece='A'), 
                                                        mean_anomaly=0.0168668618912732, 
                                                        mean_motion=MeanMotion(first_derivative=3.185528893440345e-10, 
                                                                               second_derivative=0.0, 
                                                                               value=0.06764422624907401), 
                                                        revolution_number=39717, 
                                                        right_ascension_of_ascending_node=2.027426818994173, 
                                                        satellite_number=25544, 
                                                        classification='U'), 
                         frequency=[FrequencyRange(frequency=437.525, bandwidth=None, status='inactive'),
                                    FrequencyRange(frequency=468.1, bandwidth=None, status='invalid'), 
                                    FrequencyRange(frequency=145.8, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=130.167, bandwidth=None, status='active'), 
                                    FrequencyRange(frequency=437.8, bandwidth=None, status='active'), 
                                    FrequencyRange(frequency=2213.5, bandwidth=None, status='active'), 
                                    FrequencyRange(frequency=437.8, bandwidth=None, status='active'), 
                                    FrequencyRange(frequency=400.575, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=2216.0, bandwidth=None, status='active'), 
                                    FrequencyRange(frequency=637.5, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=2265.0, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=137.6257, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=143.625, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=145.825, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=632.0, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=437.023, bandwidth=None, status='inactive'),
                                    FrequencyRange(frequency=400.5, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=630.128, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=145.8, bandwidth=None, status='invalid'),
                                    FrequencyRange(frequency=435.4, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=145.8, bandwidth=None, status='inactive'),
                                    FrequencyRange(frequency=437.05, bandwidth=None, status='inactive'),
                                    FrequencyRange(frequency=121.1, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=2205.5, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=145.48, bandwidth=None, status='inactive'),
                                    FrequencyRange(frequency=145.825, bandwidth=None, status='inactive'),
                                    FrequencyRange(frequency=121.75, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=468.1, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=2425.0, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=145.8, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=2375.0, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=417.1, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=414.2, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=437.55, bandwidth=None, status='inactive'),
                                    FrequencyRange(frequency=437.8, bandwidth=None, status='invalid'),
                                    FrequencyRange(frequency=145.8, bandwidth=None, status='inactive'),
                                    FrequencyRange(frequency=121.275, bandwidth=None, status='active')])

    @property
    def _satellite_outside_frequency_range(self) -> Satellite:
        return Satellite(name='EYESAT A (AO-27)',
                         tle_information=TleInformation(argument_of_perigee=2.6114942718570675, drag_coefficient=6.5858e-05,
                                                        eccentricity=0.0009025, epoch_days=26801.12744469,
                                                        inclination=1.7251410285365112,
                                                        international_designator=InternationalDesignator(year=93,
                                                                                                         launch_number=61,
                                                                                                         launch_piece='C'),
                                                        mean_anomaly=3.674670312355673,
                                                        mean_motion=MeanMotion(first_derivative=3.787606883668251e-12,
                                                                               second_derivative=0.0,
                                                                               value=0.06240853642079434),
                                                        revolution_number=54626,
                                                        right_ascension_of_ascending_node=3.150839407966859,
                                                        satellite_number=22825, classification='U'),
                         frequency=[FrequencyRange(frequency=436.795, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=436.795, bandwidth=None, status='active'),
                                    FrequencyRange(frequency=2218.0, bandwidth=None, status='active')])
