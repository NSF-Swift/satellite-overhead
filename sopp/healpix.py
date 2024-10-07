import numpy as np
import pandas as pd
import healpy as hp
from scipy.interpolate import griddata, LinearNDInterpolator

import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

class HealpixLoader:
    def __init__(self, csv_file, nside=2048):
        self.csv_file = csv_file
        self.nside = nside  # Resolution parameter, can be changed as needed
    
    def load_antenna_gain_data(self):
        """
        Load antenna gain data from CSV file.
        Assumes the CSV file has columns 'Azimuth', 'Elevation', and 'Gain_dB'.
        """
        df = pd.read_csv(self.csv_file)
        if df.shape[1] < 3:
            index = df['index'].values
            elevation, azimuth = hp.pix2ang(self.nside, index)
            gain = df['gain'].values
        else:
            elevation = df['alpha'].values #elevation - technically, the zenith angle 
            azimuth = df['beta'].values #azimuth
            gain = df['gain'].values
        return azimuth, elevation, gain

    def create_healpix_object(self, azimuth, elevation, gain):#FIXME: PBM HERE, returns zero array
        """
        Create HEALPix object for antenna gain.
        """
        # Convert azimuth and elevation to spherical coordinates
        theta = np.radians(elevation) #not 90-elevation because we now measure this with the zenith angle, measured from the main lobe direction
        phi = np.radians(azimuth)

        # Convert gain from dB to linear scale
        #gain_linear = 10 ** (gain_dB / 10.0)

        # Initialize HEALPix map
        healpix_gain = np.zeros(hp.nside2npix(self.nside))
        counts = np.zeros(hp.nside2npix(self.nside)) ##added for averaging***

        # Aggregate gains to HEALPix pixels
        pixel_indices = hp.ang2pix(self.nside, theta, phi)
        
        for i in range(len(gain)):
            healpix_gain[pixel_indices[i]] += gain[i]
            #healpix_gain[pixel_indices[i]] = gain[i]
            counts[pixel_indices[i]] += 1   ###added for averaging***
        ### this stuff added for averaging
        with np.errstate(divide='ignore', invalid='ignore'):
            healpix_gain = np.divide(healpix_gain, counts, where=(counts > 0))

        return healpix_gain

    def load_healpix_gain_pattern(self):
        azimuth, elevation, gain = self.load_antenna_gain_data()
        healpix_gain = self.create_healpix_object(azimuth, elevation, gain)
        return healpix_gain


class HealpixInterLoader(HealpixLoader):
    def interpolate_data(self, azimuth, elevation, gain, method='linear'):
        """
        Interpolate antenna gain data to ensure even sampling.
        """
        # Convert azimuth and elevation to radians
        theta = np.radians(elevation)
        phi = np.radians(azimuth)

        # npix_mesh = hp.nside2npix(self.nside)
        # #elevation_mesh = np.zeros(npix_mesh)
        # elevation_mesh = []
        # #azimuth_mesh = np.zeros(npix_mesh)
        # azimuth_mesh = []

        ind_first_az = np.where(azimuth == 0)[0]

        gain_interpolated = LinearNDInterpolator(list(zip(np.append(theta, theta[ind_first_az]),
                                                          np.append(phi, phi[ind_first_az]+2*np.pi))), 
                                                 np.append(gain, gain[ind_first_az]))

        # for i in range(npix_mesh):
        #     theta_s,phi_s = hp.pix2ang(self.nside, i)
        #     elevation_mesh.append(theta_s)
        #     azimuth_mesh.append(phi_s)

        # # Define grid for interpolation
        # # azimuth_grid = np.linspace(0, 360, grid_resolution)
        # # elevation_grid = np.linspace(-90, 90, grid_resolution)
        # # azimuth_mesh, elevation_mesh = np.meshgrid(azimuth_grid, elevation_grid)

        # # Interpolate gain values
        # # gain_interpolated = RegularGridInterpolator((phi,theta), gain_dB)
        # grid_points = np.column_stack((azimuth_mesh, elevation_mesh))


        # gain_interpolated = griddata((phi,theta), gain_dB, grid_points, method=method, fill_value=0.0)

        # return azimuth_mesh, elevation_mesh, gain_interpolated
        return gain_interpolated#, grid_points

    def create_healpix_object(self, azimuth, elevation, gain, method='linear'):
        """
        Create HEALPix object for antenna gain with interpolation.
        """
        # gain_interpolated, grid_points = self.interpolate_data(azimuth,
        # elevation, gain_dB, method=method)
        gain_interpolated = self.interpolate_data(azimuth, elevation, gain, method=method)

        # Convert interpolated azimuth and elevation to spherical coordinates
        # theta_mesh = np.radians(90 - elevation_mesh)
        # phi_mesh = np.radians(azimuth_mesh)

        # Initialize HEALPix map
        healpix_gain = np.zeros(hp.nside2npix(self.nside))
        for i in range(len(healpix_gain)):
            theta_s,phi_s = hp.pix2ang(self.nside, i)
            healpix_gain[i] = gain_interpolated(theta_s, phi_s)
        
        # npix_mesh = hp.nside2npix(self.nside)
        """
        for i in range(len(healpix_gain)):
            theta_s,phi_s = hp.pix2ang(self.nside, i)
            ## now that we have an angle, we must conif (gain_value == 0):
        print("oops")vert the angle to an index that corresponds to a query point
             # Convert theta_s and phi_s to radians (already in radians)
            query_point = np.array([phi_s, theta_s])

            # Retrieve interpolated gain value
            if np.isnan(gain_interpolated).all():
                # Handle case where all interpolated values are NaN
                healpix_gain[i] = np.nan
                print("somethings wrong here")
            else:
                # Use np.where to find closest index in grid_points
                distances = np.linalg.norm(grid_points - query_point, axis=1)
                closest_index = np.argmin(distances)
                healpix_gain[i] = gain_interpolated[closest_index]
            # healpix_gain[i] = 10* np.log10(gain_interpolated[i]) #index accesses the index of the query point (third argument)
            # healpix_gain[i] = gain_interpolated[i]
        # Aggregate gains to HEALPix pixels
        # pixel_indices = hp.ang2pix(self.nside, theta_mesh.ravel(), phi_mesh.ravel())
        # for i, pixel_index in enumerate(pixel_indices):
        #    healpix_gain[pixel_index] += gain_interpolated.ravel()[i]
        """
        # return gain_interpolated
        return healpix_gain


class HealpixGainPattern:
    def __init__(self, healpix_gain: np.ndarray, nside=2048):
        self.healpix_gain = healpix_gain
        self.nside = nside

    def get_gain(self, theta: float, phi: float) -> float:
        """
        Get gain at specific spherical coordinates (theta, phi).
        """
        npix = hp.nside2npix(self.nside)
        #pixel_index = hp.ang2pix(self.nside, np.radians(90-theta), np.radians(phi))
        pixel_index = hp.ang2pix(self.nside, np.radians(theta), np.radians(phi))
        return self.healpix_gain[pixel_index]#FIXME: check with the healpix interpolator
    
"""
# Example usage IF the data is already in Healpix format
if __name__ == "__main__":
    # Example CSV file path
    csv_file = 'antenna_gain_pattern.csv'

    # Create HealpixLoader instance
    healpix_loader = HealpixLoader(csv_file)

    # Load HEALPix gain pattern
    healpix_gain = healpix_loader.load_healpix_gain_pattern()

    # Create HealpixGainPattern instance
    healpix_pattern = HealpixGainPattern(healpix_gain)

    # Example usage: get gain at specific spherical coordinates
    theta_example = 45  # Elevation angle in degrees
    phi_example = 90   # Azimuth angle in degrees
    gain_example = healpix_pattern.get_gain(theta_example, phi_example)

    print(f"Gain at (theta={theta_example}°, phi={phi_example}°): {gain_example}")

# Example usage IF the data is not already evenly sampled, and we need to use some interpolation
if __name__ == "__main__":
    # Example CSV file path
    csv_file = 'antenna_gain_pattern.csv'

    # Create HealpixInterLoader instance
    healpix_loader = HealpixInterLoader(csv_file)

    # Load HEALPix gain pattern
    healpix_gain = healpix_loader.load_healpix_gain_pattern()

    # Create HealpixGainPattern instance
    healpix_pattern = HealpixGainPattern(healpix_gain)

    # Example usage: get gain at specific spherical coordinates
    theta_example = 45  # Elevation angle in degrees
    phi_example = 90   # Azimuth angle in degrees
    gain_example = healpix_pattern.get_gain(theta_example, phi_example)

    print(f"Gain at (theta={theta_example}°, phi={phi_example}°): {gain_example}")
"""
