import numpy as np

def compute_rotation_matrix_geodetic_to_local(latitude, longitude):
    # Convert degrees to radians
    phi = np.radians(latitude)
    lambda_ = np.radians(longitude)
    
    # Rotation about the Z-axis (Longitude)
    Rz = np.array([
        [np.cos(-lambda_), np.sin(-lambda_), 0],
        [-np.sin(-lambda_), np.cos(-lambda_), 0],
        [0, 0, 1]
    ], dtype=float)
    
    # Rotation about the Y-axis (Latitude)
    Ry = np.array([
        [np.cos(-phi), 0, np.sin(-phi)],
        [0, 1, 0],
        [-np.sin(-phi), 0, np.cos(-phi)]
    ], dtype=float)
    
    # Matrix multiplication
    R = np.dot(Ry, Rz)
    
    return R

def compute_rotation_matrix_gcrs_to_geodetic(latitude, longitude):
    # Convert degrees to radians
    phi = np.radians(latitude)
    lambda_ = np.radians(longitude)
    
    # Rotation about the Z-axis (Longitude)
    Rz = np.array([
        [np.cos(-lambda_), np.sin(-lambda_), 0],
        [-np.sin(-lambda_), np.cos(-lambda_), 0],
        [0, 0, 1]
    ], dtype=float)
    
    # Rotation about the Y-axis (Latitude)
    Ry = np.array([
        [np.cos(-phi), 0, np.sin(-phi)],
        [0, 1, 0],
        [-np.sin(-phi), 0, np.cos(-phi)]
    ], dtype=float)
    
    # Matrix multiplication
    R = np.dot(Ry, Rz)
    
    return R
"""
# Example usage
latitude = 45.0   # Latitude in degrees
longitude = 30.0  # Longitude in degrees

# Geodetic to Local Geodetic
rotation_matrix_geodetic_to_local = compute_rotation_matrix_geodetic_to_local(latitude, longitude)
print("Geodetic to Local Geodetic Rotation Matrix:")
print(rotation_matrix_geodetic_to_local)

# GCRS to Geodetic
rotation_matrix_gcrs_to_geodetic = compute_rotation_matrix_gcrs_to_geodetic(latitude, longitude)
print("\nGCRS to Geodetic Rotation Matrix:")
print(rotation_matrix_gcrs_to_geodetic)
"""
