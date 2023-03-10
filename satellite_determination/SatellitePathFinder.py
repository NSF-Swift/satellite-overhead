from skyfield.api import load, wgs84
from ListOfSatellites import loadSatellites
import datetime
import sys

HCRO = wgs84.latlon(40.8178049, -121.4695413) #Coordinate of Hat Creek Radio Observatory
ts = load.timescale()

'''
Function takes a location and reservation start and end time and computes where each satellite is
at start and end of window and if there are any events where it rises above a certain altitude on
horizon

300
'''

def FindOverflight(facility, res_start, res_end):
    i = 0
    y = 0
    t0 = ts.utc(2022, 10, 30, 0, 0)
    t1 = ts.utc(2022, 10, 30, 0, 15)
    satellites = loadSatellites('./TLEdata/test.txt')
    interferers = []
    incident_time = []
    for sat in satellites:
        t, events = sat.find_events(facility, t0, t1, altitude_degrees=85.0)
        for ti, event in zip(t, events):
            name = ('rise above 85°', 'culminate', 'set below 85°')[event]
            interferers.append(sat)
            incident_time.append(ti)
            print(ti.utc_strftime('%Y %b %d %H:%M:%S'), name)
            i+=1
    print('# of satellites overhead during reservation window: ', i/3)
    for sat in interferers:
        time = incident_time[y]
        print('Satellite name: ', sat.name, 'Time of interference: ', time.utc_strftime('%Y %b %d %H:%M:%S'))
        y+=1


