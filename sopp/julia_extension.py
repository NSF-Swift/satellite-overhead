import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapz
from scipy.interpolate import RegularGridInterpolator

# Constants
k_boltz = 1.38e-23  # Boltzmann's constant in J/K
π = np.pi
c = 3e8  # Speed of light in m/s
rad = np.pi / 180.0
free_space_impedance = 377  # Free space impedance in Ohms


def power_pattern_from_cut_file(file_path, free_sp_imp=377, verbose=False):
    """
    Reads a .cut file containing antenna pattern data and computes the power pattern in dBW.

    Args:
    - file_path (str): Path to the .cut file.
    - free_sp_imp (float): Free-space impedance (default: 377 Ohms).
    - verbose (bool): Verbosity flag (default: False).

    Returns:
    - pattern (pd.DataFrame): DataFrame containing alpha, beta, and power.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    pattern = []
    k = 0
    while k < len(lines):
        if "Field" in lines[k]:
            header = lines[k].split()
            α_start = float(header[1])
            α_step = float(header[2])
            nb_α = int(header[3])
            for t in range(1, nb_α + 1):
                α = α_start + (t - 1) * α_step
                θ = float(header[4])
                # Power pattern in dBW
                u = np.sum(np.array(lines[k + t].split()[0:4], dtype=float)**2) / (2 * free_sp_imp)
                pattern.append({'alpha': α, 'beta': θ, 'power': u})
            k = k + nb_α + 1
        else:
            k += 1

    # Ensure correct orientation and domain adjustments
    pattern_df = pd.DataFrame(pattern)
    pattern_df.loc[pattern_df['alpha'] == 180.0, 'alpha'] *= -1
    pattern_df = pattern_df.sort_values(by=['beta', 'alpha']).reset_index(drop=True)

    return pattern_df


def radiated_power_to_gain(rad_pow, alphas, betas, eta_rad=1.0):
    """
    Converts radiated power pattern to antenna gain pattern in dB.

    Args:
    - rad_pow (np.ndarray): Array of radiated power values.
    - alphas (np.ndarray): Array of alpha angles (in degrees).
    - betas (np.ndarray): Array of beta angles (in degrees).
    - eta_rad (float): Radiation efficiency (default: 1.0).

    Returns:
    - gain (np.ndarray): Array of antenna gain values.
    """
    rad = np.pi / 180.0
    rad_pow_map = np.hstack((np.reshape(rad_pow, (len(alphas), len(betas))), rad_pow[0:len(alphas)]))
    a = alphas * rad
    b = np.hstack((betas, 360.0)) * rad
    rad_pow_avg = trapz(rad_pow_map * np.sin(a), x=b) / (4 * np.pi)

    return eta_rad * rad_pow / rad_pow_avg


def gain_to_effective_aperture(gain, frequency):
    """
    Converts antenna gain to effective aperture.

    Args:
    - gain (np.ndarray): Array of antenna gain values.
    - frequency (float): Operating frequency (in Hz).

    Returns:
    - effective_aperture (np.ndarray): Array of effective aperture values.
    """
    wavelength = c / frequency
    return gain * (wavelength**2 / (4 * np.pi))


def spher_to_cart_coord(theta, phi, r=1.0):
    """
    Converts spherical coordinates to Cartesian coordinates.

    Args:
    - theta (float): Theta angle (in radians).
    - phi (float): Phi angle (in radians).
    - r (float): Radius (default: 1.0).

    Returns:
    - cart_coord (np.ndarray): Cartesian coordinates [x, y, z].
    """
    return np.array([r * np.sin(theta) * np.cos(phi),
                     r * np.sin(theta) * np.sin(phi),
                     r * np.cos(theta)])


def cart_to_sphe_coord(x, y, z):
    """
    Converts Cartesian coordinates to spherical coordinates.

    Args:
    - x (float): x-coordinate.
    - y (float): y-coordinate.
    - z (float): z-coordinate.

    Returns:
    - theta (float): Theta angle (in radians).
    - phi (float): Phi angle (in radians).
    - r (float): Radius.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return theta, phi, r


def ground_to_beam_coord(dec_obj, az_obj, dec_tel, az_tel):
    """
    Converts ground-based coordinates to beam coordinates relative to a telescope.

    Args:
    - dec_obj (float): Declination of the object (in radians).
    - az_obj (float): Azimuth of the object (in radians).
    - dec_tel (float): Declination of the telescope (in radians).
    - az_tel (float): Azimuth of the telescope (in radians).

    Returns:
    - theta_beam (float): Theta angle in the beam coordinate system (in radians).
    - phi_beam (float): Phi angle in the beam coordinate system (in radians).
    """
    # Rotation matrices
    R_z_gamma = np.array([[np.cos(az_tel), -np.sin(az_tel), 0],
                          [np.sin(az_tel), np.cos(az_tel), 0],
                          [0, 0, 1]])

    R_w_psi = np.array([[np.cos(dec_tel), 0, np.sin(dec_tel)],
                        [0, 1, 0],
                        [-np.sin(dec_tel), 0, np.cos(dec_tel)]])

    # Object coordinates in NEZ (North-East-Zenith)
    obj_NEZ = spher_to_cart_coord(dec_obj, az_obj)
    # Object coordinates in PGB
    obj_pgb = np.dot(R_w_psi.T, np.dot(R_z_gamma.T, obj_NEZ))

    # Convert PGB coordinates to spherical
    theta_beam, phi_beam, _ = cart_to_sphe_coord(obj_pgb[0], obj_pgb[1], obj_pgb[2])

    return theta_beam, phi_beam


def estim_temp(flux, effective_aperture):
    """
    Estimates the temperature of a source from its flux and the antenna effective aperture.

    Args:
    - flux (float): Flux of the source.
    - effective_aperture (float): Effective aperture of the antenna.

    Returns:
    - temperature (float): Estimated temperature of the source.
    """
    return flux * 1e-26 / (2 * k_boltz) * effective_aperture


def estim_casA_flux(center_freq):
    """
    Estimates the flux of Cas A, given a frequency.

    Args:
    - center_freq (float): Center frequency (in Hz).

    Returns:
    - casA_flux (float): Estimated flux of Cas A.
    """
    decay = 0.97 - 0.3 * np.log10(center_freq * 1e-9)
    casA_flux = 10**(5.745 - 0.770 * np.log10(center_freq * 1e-6)) * (1 - decay * 43 / 100)
    return casA_flux


def datetime_to_unix(dt):
    """
    Converts datetime object to Unix timestamp.

    Args:
    - dt (datetime): Datetime object.

    Returns:
    - unix_time (float): Unix timestamp.
    """
    return (dt - datetime(1970, 1, 1)).total_seconds()


def unix_to_datetime(unix_time):
    """
    Converts Unix timestamp to datetime object.

    Args:
    - unix_time (float): Unix timestamp.

    Returns:
    - dt (datetime): Datetime object.
    """
    return datetime.utcfromtimestamp(unix_time)


if __name__ == "__main__":
    # Load satellite trajectories
    traj_sats_w = pd.read_arrow("traj_files/Starlink_trajectory_Westford_2024-05-13T00:00:00.000_2024-05-13T01:00:00.000.arrow")
    traj_sats_w['timestamp'] = pd.to_datetime(traj_sats_w['timestamp'])

    start_window = datetime(2024, 5, 13)
    end_window = datetime(2024, 5, 13, 1)

    traj_sats_in_window = traj_sats_w[(traj_sats_w['timestamp'] >= start_window) & 
                                      (traj_sats_w['timestamp'] <= end_window)]

    # Load satellite antenna pattern
    file_path = "single_cut_res.cut"
    gain_ant = power_pattern_from_cut_file(file_path)

    alphas = np.unique(gain_ant['alpha'].values)
   
