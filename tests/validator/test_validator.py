from skyfield.api import load, wgs84
import datetime
import sys
import os
from pathlib import Path
import numpy as nps
from datetime import datetime
from satellite_determination.dataclasses.coordinates import Coordinates
from satellite_determination.dataclasses.facility import Facility
from satellite_determination.dataclasses.time_window import TimeWindow
from satellite_determination.dataclasses.reservation import Reservation
from satellite_determination.dataclasses.overhead_window import OverheadWindow
from satellite_determination.retrievers.satellite_retriever.skyfield_satellite_retriever import SkyfieldSatelliteList
from satellite_determination.validator.validator import Validator
from satellite_determination.dataclasses.time_window import TimeWindow
from satellite_determination.utilities import convert_dt_to_utc
from tests.utilities import get_script_directory
import json
import filecmp
from skyfield.timelib import Timescale
from skyfield.api import utc

class TestValidatorRhodesMill(Validator):

    def overhead_list(self, list_of_satellites: SkyfieldSatelliteList, reservation: Reservation):
        ts = load.timescale()
        interferers = []
        t0 = ts.utc(convert_dt_to_utc(reservation.time.begin))
        t1 = ts.utc(convert_dt_to_utc(reservation.time.end))
        coordinates = wgs84.latlon(reservation.facility.point_coordinates.latitude, reservation.facility.point_coordinates.longitude)
        for sat in list_of_satellites.satellites:
            t, events = sat.find_events(coordinates, t0, t1, altitude_degrees=reservation.facility.angle_of_visibility_cone)
            if events.size == 0:
                continue
            else:
                for ti, event in zip(t, events):
                    if event == 0:
                        begin = ti
                    elif event == 2:
                        end = ti
                time_window = TimeWindow(begin, end)
                overhead = OverheadWindow(sat, time_window)
                interferers.append(overhead)
        return interferers

    def test_can_get_overhead_list(self):
        tle_file = Path(get_script_directory(__file__), 'TLEdata', 'test.txt')
        list_of_satellites = SkyfieldSatelliteList.load_tle(str(tle_file))
        reservation = Reservation(
                facility=Facility(
                    angle_of_visibility_cone=20.1,
                    point_coordinates=Coordinates(latitude=4., longitude=5.),
                    name='ArbitraryFacilityName2'
                ),
                time=TimeWindow(
                    begin=datetime(year=2022, month=12, day=1, hour=16),
                    end=datetime(year=2022, month=12, day=1, hour=17)
                )
        )
        interferers = self.overhead_list(list_of_satellites, reservation)
        dict = {
            "satellite_name": []
        }
        for interferer in interferers:
                print(interferer.satellite.name)
                dict["satellite_name"].append(interferer.satellite.name)
        with open ("satellite_overhead_test", "a") as outfile:
            json.dump(dict, outfile)
            outfile.close()
        assert filecmp.cmp('./tests/validator/satellite_reference_file', 'satellite_overhead_test') == 1
        os.remove("satellite_overhead_test")