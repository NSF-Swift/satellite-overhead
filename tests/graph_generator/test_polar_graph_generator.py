from datetime import datetime, timezone

import numpy as np
import pytest
from sopp.models.position import Position
from sopp.models.position_time import PositionTime
from sopp.graph_generator.graph_polar import GraphGeneratorPolar

observation_data = [
    PositionTime(
        position=Position(altitude=0.037991772033771, azimuth=225.42667523243927),
        time=datetime(2023, 9, 15, 3, 13, 31, tzinfo=timezone.utc),
    ),
    PositionTime(
        position=Position(altitude=0.1004175887946367, azimuth=225.4135720960317),
        time=datetime(2023, 9, 15, 3, 13, 32, tzinfo=timezone.utc),
    ),
    PositionTime(
        position=Position(altitude=0.16302916811324802, azimuth=225.40037964262117),
        time=datetime(2023, 9, 15, 3, 13, 33, tzinfo=timezone.utc),
    ),
    PositionTime(
        position=Position(altitude=0.22582815942805465, azimuth=225.38709702253257),
        time=datetime(2023, 9, 15, 3, 13, 34, tzinfo=timezone.utc),
    ),
    PositionTime(
        position=Position(altitude=0.28881623145852403, azimuth=225.37372337554274),
        time=datetime(2023, 9, 15, 3, 13, 35, tzinfo=timezone.utc),
    ),
]


@pytest.fixture
def graph_generator():
    return GraphGeneratorPolar(observation_data=observation_data, sat_name="ISS")


class TestGraphGeneratorPolar:
    def test_title(self, graph_generator):
        assert graph_generator._title == "ISS 2023-09-15 03:13:31 - 03:13:35 UTC"

    def test_altitude_values(self, graph_generator):
        assert len(graph_generator._altitude_values) == len(observation_data)
        assert (
            graph_generator._altitude_values[0] == observation_data[0].position.altitude
        )
        assert (
            graph_generator._altitude_values[-1]
            == observation_data[-1].position.altitude
        )

    def test_azimuth_values(self, graph_generator):
        assert len(graph_generator._azimuth_values) == len(observation_data)
        assert graph_generator._azimuth_values[0] == np.radians(
            observation_data[0].position.azimuth
        )
        assert graph_generator._azimuth_values[-1] == np.radians(
            observation_data[-1].position.azimuth
        )

    def test_time_values(self, graph_generator):
        assert len(graph_generator._time_values) == len(observation_data)
        assert graph_generator._time_values[0] == observation_data[0].time
        assert graph_generator._time_values[-1] == observation_data[-1].time
