import json
from skyfield.api import load, wgs84
import datetime
import sys
import numpy as nps
from datetime import datetime
from satellite_determination.dataclasses.coordinates import Coordinates
from satellite_determination.dataclasses.facility import Facility
from satellite_determination.dataclasses.time_window import TimeWindow
from satellite_determination.dataclasses.reservation import Reservation
from satellite_determination.dataclasses.overhead_window import OverheadWindow
from satellite_determination.retrievers.satellite_retriever.skyfield_satellite_retriever import SkyfieldSatelliteList
from tests.event_finder.test_validator import TestValidatorRhodesMill
from satellite_determination.dataclasses.time_window import TimeWindow

from skyfield.timelib import Timescale
from skyfield.api import utc

def get_reference_list():
    validator = TestValidatorRhodesMill()
    list_of_satellites = SkyfieldSatelliteList.load_tle('TLEdata/test.txt')
    reservation = Reservation(
            facility=Facility(
                angle_of_visibility_cone=20.1,
                point_coordinates=Coordinates(latitude=4., longitude=5.),
                name='ArbitraryFacilityName2'
            ),
            time=TimeWindow(
                begin=datetime(year=2022, month=12, day=15, hour=16),
                end=datetime(year=2022, month=12, day=15, hour=17)
            )
    )
    interferers = validator.overhead_list(list_of_satellites, reservation)
    dict = {
        "satellite_name": []
    }
    for interferer in interferers:
            print(interferer.satellite.name)
            dict["satellite_name"].append(interferer.satellite.name)
    with open ("satellite_reference_file", "a") as outfile:
        json.dump(dict, outfile)
        outfile.close()

get_reference_list()
