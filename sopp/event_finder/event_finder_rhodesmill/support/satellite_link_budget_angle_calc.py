import math
import numpy as np
from numpy import matmul

from typing import List

from sopp.custom_dataclasses.position_time import PositionTime
from sopp.custom_dataclasses.satellite.satellite import Satellite

from sopp.custom_dataclasses.position import CartesianCoordinate
from sopp.custom_dataclasses.facility import Facility

from sopp.event_finder.event_finder_rhodesmill.support.gcrs_geodetic_local_switcher import compute_rotation_matrix_gcrs_to_geodetic
from sopp.event_finder.event_finder_rhodesmill.support.gcrs_geodetic_local_switcher import compute_rotation_matrix_geodetic_to_local



from skyfield.api import load

R = 6371.0  # approximate radius of Earth in km
base_alt = 500 # relative average of altitude in km, tbd might need to change or alter value

ts = load.timescale()

class SatelliteLinkBudgetAngleCalculator:
    def __init__(self, facility: Facility, ground_antenna_direction: PositionTime, satellite_position: PositionTime, satellite: Satellite):
        self.facility = facility
        self.ground_antenna_direction = ground_antenna_direction
        self.satellite_position = satellite_position
        self.satellite = satellite

    def get_link_angles(self):
        results = []
        ground_ab = self.calculate_ab_ground()
        # sat_ab = self.calculate_ab_sat()
        results.append(ground_ab[0])
        results.append(ground_ab[1])
        # results.append(sat_ab[0])
        # results.append(sat_ab[1])#FIXME:
        return results  #results are in alpha, beta for each 
    
    def calculate_ab_ground(self) -> List[float]: #calculates the alpha and beta angles for gain pattern of the ground antenna
        self.satellite_position.position.azimuth = -(self.satellite_position.position.azimuth) #flips the azimuth into a right handed coordinate system
        satellite_cartesian = self.satellite_position.position.to_cartesian()
        rotated_vector = satellite_cartesian.pass_to_rotation_matrix(-self.ground_antenna_direction.position.azimuth, (90-self.ground_antenna_direction.position.altitude)) #negative is because we are using right handed system, and azimuth not measured right handedly
        x = rotated_vector[0]
        y = rotated_vector[1]
        z = rotated_vector[2]
        new_coordinate = CartesianCoordinate(x,y,z)
        return new_coordinate.cartesian_to_spherical()
    
    def calculate_ab_sat(self) -> List[float]: #calculates the alpha and beta angles for gain pattern of the satellite antenna
        dt = self.satellite_position.time
        t = ts.utc(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        earth_sat = self.satellite.to_rhodesmill()
        position, velocity, _, error = earth_sat._at(t) # gives the velocity in x,y,z for the geocentric coordinate system
        velocity_cartesian = CartesianCoordinate(velocity[0],velocity[1],velocity[2])
        matOne = compute_rotation_matrix_gcrs_to_geodetic(self.facility.coordinates.latitude,self.facility.coordinates.longitude)
        matTwo = compute_rotation_matrix_geodetic_to_local(self.facility.coordinates.latitude,self.facility.coordinates.longitude)
        rotated_velocity = velocity_cartesian.pass_to_gsrc_local_matrix(matOne,matTwo)
        theta = self.calculate_altitude_difference_space() #altitude difference
        vert_angle = 360 - (self.satellite_position.position.altitude + theta + 90)
        rotated_velocity_coord = CartesianCoordinate(rotated_velocity[0],rotated_velocity[1],rotated_velocity[2])
        phi = self.calculate_azimuth_difference_space(rotated_velocity_coord) #azimuth difference
        horiz_angle = phi + self.ground_antenna_direction.position.azimuth
        ground_cartesian = self.satellite_position.position.to_cartesian()
        ground_cartesian.x = -(ground_cartesian.x)
        ground_cartesian.y = -(ground_cartesian.y)
        ground_cartesian.z = -(ground_cartesian.z)
        rotated_vector = ground_cartesian.pass_to_rotation_matrix(horiz_angle, vert_angle)
        x = rotated_vector[0]
        y = rotated_vector[1]
        z = rotated_vector[2]
        new_coordinate = CartesianCoordinate(x,y,z)
        return new_coordinate.cartesian_to_spherical()

    #### some of the functions below this were used before and might become useful for additional algorithms in the future, but are not in use in this file


    def calculate_azimuth_difference_ground(self): #between antenna direction and direction to satellite
       
        satellite_azimuth = self.satellite_position.position.azimuth
        ground_azimuth = self.ground_antenna_direction.position.altitude

        azimuth_difference = satellite_azimuth - ground_azimuth

        return azimuth_difference

    def calculate_altitude_difference_ground(self): #between antenna direction and direction to satellite
        
        satellite_altitude = self.satellite_position.position.altitude
        ground_altitude = self.ground_antenna_direction.position.altitude

        altitude_difference = satellite_altitude - ground_altitude #sign is going to be important here so we are not calculating absolute value

        return altitude_difference

    def calculate_altitude_difference_space(self): #between satellite pointing direction and direction to ground antenna
        R = 6371.0  # approximate radius of Earth in km
        dist = self.satellite_position.position.distance_km
        theta_deg = self.satellite_position.position.altitude
        theta_rad = math.radians(theta_deg)
        base_alt = dist * math.sin(theta_rad) # relative average of altitude in km, tbd might need to change or alter value

        # satellite_altitude = self.satellite_position.position.altitude
        #satellite_pointing_altitude = self.satellite.antenna.direction.altitude

        # altitude_difference = satellite_altitude - satellite_pointing_altitude #sign is going to be important here so we are not calculating absolute value

        dist = self.satellite_position.position.distance_km
        # print("Distance to the satellite is: " + str(dist)) This one works so don't worry this is ok
        return self.law_of_cosines_angle_c(dist, (R + base_alt), R)
    
    def calculate_azimuth_difference_space(self, coord: CartesianCoordinate): #between satellite pointing direction and direction to ground antenna
        R = 6371.0  # approximate radius of Earth in km
        base_alt = 500 # relative average of altitude in km, tbd might need to change or alter value

        angle_radians = math.atan2(coord.x, coord.y)
    
        # Convert the angle to degrees
        angle_degrees = math.degrees(angle_radians)
    
        return angle_degrees
    

    def law_of_cosines_angle_c(self, a, b, c):
        # Ensure sides are positive (valid triangle sides)
        if a <= 0 or b <= 0 or c <= 0:
            raise ValueError("Side lengths must be positive.")

        # Calculate cos(C) using the law of cosines
        cos_C = (a**2 + b**2 - c**2) / (2 * a * b)

        # Check for valid cosine value range
        if cos_C < -1 or cos_C > 1:
            raise ValueError("Invalid triangle sides - no such triangle exists.")

        # Calculate angle C in radians
        angle_C_rad = math.acos(cos_C)

        # Convert angle from radians to degrees
        angle_C_deg = math.degrees(angle_C_rad)

        return angle_C_deg
        

# Example usage:
if __name__ == "__main__":
    # Example input parameters
    ground_antenna_direction = (120.0, 45.0)  # azimuth and elevation angles in degrees
    satellite_position = (35.0, 140.0, 35786.0)  # latitude, longitude, and altitude in degrees and meters
    satellite_pointing_position = (200.0, 30.0)  # azimuth and elevation angles in degrees

    # Create an instance of the calculator
    calculator = SatelliteLinkBudgetAngleCalculator(ground_antenna_direction, satellite_position, satellite_pointing_position)

    # Calculate the link budget
    link_budget = calculator.calculate_link_budget()

    print(f"Link Budget: {link_budget:.2f} dB")

