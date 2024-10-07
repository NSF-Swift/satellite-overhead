import pandas as pd
import numpy as np

from scipy.spatial import cKDTree

class GainPattern:
    def __init__(self, csv_file):
    
        self.csv_file = csv_file
        self.df = self.load_antenna_gain_data()  # Load and store the data
        self.tree = self.build_kd_tree()

    def build_kd_tree(self):
        # Build the KD-tree from the DataFrame points
        points = self.df[['alpha', 'beta']].values
        return cKDTree(points)
    
    def load_antenna_gain_data(self):

        df = pd.read_csv(self.csv_file)
        
        # Ensure the necessary columns are present
        if not all(col in df.columns for col in ['alpha', 'beta', 'gain']):
            raise ValueError("CSV file must contain 'alpha', 'beta', and 'gain' columns")
        
        # Note that beta is the azimuth value, therefore alpha is the altitude value

        return df
    
    def get_gain(self, theta: float, phi: float): # must be in altitude,azmith format THETA IS ALTITUDE

        if (phi <= 0):
            phi += 360
        """
        Get gain at specific spherical coordinates (theta, phi) in degrees.
        
        :param theta_deg: Theta in degrees (altitude).
        :param phi_deg: Phi in degrees (azimuth).
        :return: Gain value at the specified (theta, phi), or the closest value if exact match is not found.
        """
        query_point = np.array([theta, phi])

        # Query the KD-tree for the nearest neighbor
        distance, index = self.tree.query(query_point)
        
        # Return the gain value of the closest point
        closest_gain = self.df.iloc[index]['gain']
        
        return closest_gain
        """
        # Compute distances between query_point and all points in DataFrame
        df_points = self.df[['alpha', 'beta']].values
        distances = np.linalg.norm(df_points - query_point, axis=1)
        
        # Find the index of the closest point
        closest_index = np.argmin(distances)
        
        # Return the gain value of the closest point
        closest_gain = self.df.iloc[closest_index]['gain']
        
        return closest_gain
        """

# Example usage:
# file_path = 'path/to/your/gain_pattern.csv'
# gain_pattern = GainPattern(file_path)
# gain_value = gain_pattern.get_gain(45, 90)
# print(f"Gain value: {gain_value}")
