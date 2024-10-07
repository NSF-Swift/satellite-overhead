from dataclasses import dataclass
from typing import Optional

import numpy as np
from typing import List

import math


class CartesianCoordinate:
    def __init__(self, x: float, y: float, z: float):
        self.x = x  # x-coordinate
        self.y = y  # y-coordinate
        self.z = z  # z-coordinate

    def set_coordinates(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def get_coordinates(self):
        return self.x, self.y, self.z
    
    def pass_to_rotation_matrix(self, gamma: float, phi: float): #gamma and phi are in degrees when we input them, so we must convert to radians
        gamma = np.deg2rad(gamma)
        phi = np.deg2rad(phi)
        matrixOne = np.array([
            [np.cos(gamma), -np.sin(gamma), 0],
            [np.sin(gamma), np.cos(gamma), 0],
            [0, 0, 1]
        ])

        matrixTwo = np.array([
            [np.cos(phi), 0, np.sin(phi)],
            [0, 1, 0],
            [-np.sin(phi), 0, np.cos(phi)]
        ])
        origMatrix = np.matmul(matrixOne, matrixTwo)
        totMatrix = origMatrix.T
        cartesianArray = np.array([self.x,self.y,self.z])
        return self.apply_rotation(totMatrix,cartesianArray)
    
    def pass_to_gsrc_local_matrix(self, matrixOne: np.array, matrixTwo: np.array):
        origMatrix = np.matmul(matrixOne,matrixTwo)
        #totMatrix = origMatrix.T
        cartesianArray = np.array([self.x,self.y,self.z])
        return self.apply_rotation(origMatrix,cartesianArray)
    
    def apply_rotation(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """
        Apply rotation matrix to a vector of Cartesian coordinates.
    
        Args:
        - matrix (np.ndarray): 3x3 rotation matrix
        - vector (np.ndarray): 1x3 vector of Cartesian coordinates [x, y, z]
    
        Returns:
        - np.ndarray: Resulting vector after rotation
        """
        return np.matmul(matrix, vector)
    
    def cartesian_to_spherical(self) -> List[float]:
        """
        Convert Cartesian coordinates to spherical coordinates.
        """

        x = self.x
        y = self.y
        z = self.z

        r = np.sqrt(x**2 + y**2 + z**2)  # Radial distance

        theta = np.arccos(z/r)

        if x > 0:
            phi = np.arctan(y/x)
        if x < 0 and y >= 0:
            phi = np.arctan(y/x) + np.pi
        if x < 0 and y < 0:
            phi = np.arctan(y/x) - np.pi
        if x == 0 and y > 0:
            phi = np.pi/2
        if x == 0 and y < 0:
            phi = -np.pi/2
        if x == 0 and y == 0:
            phi = 0 # default value because this is undefined, maybe change this to some other value?

        return [np.degrees(theta), np.degrees(phi)]

@dataclass
class Position:
    """
    Represents a position relative to an observer on Earth.

    Attributes:
    + altitude (float): The altitude angle of the object in degrees. It ranges
      from 0° at the horizon to 90° directly overhead at the zenith. A negative
      altitude means the satellite is below the horizon.
    + azimuth (float): The azimuth angle of the object in degrees, measured
      clockwise around the horizon. It runs from 0° (geographic north) through
      east (90°), south (180°), and west (270°) before returning to the north.
    + distance (Optional[float]): The straight-line distance between the
      object and the observer in kilometers. If not provided, it is set to
      None.
    """
    altitude: float
    azimuth: float
    distance_km: Optional[float] = None

    def to_cartesian(self) -> CartesianCoordinate:
        if self.distance_km is None:
            raise ValueError("Distance must be provided to convert to Cartesian coordinates.")
        
        # Convert angles to radians
        theta = math.radians(90 - self.altitude)
        phi = math.radians(self.azimuth)
        
        # Convert to Cartesian coordinates
        x = self.distance_km * math.sin(theta) * math.cos(phi)
        y = self.distance_km * math.sin(theta) * math.sin(phi)
        z = self.distance_km * math.cos(theta)
        
        return CartesianCoordinate(x, y, z)
    
