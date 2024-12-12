# kinematics.py

import numpy as np
import time
import pandas as pd
import numba as nb
from .interpolation_functions import (
    _searchsorted_opt,
    _bilinear_interpolation,
    _trilinear_interpolation,
    _fill_distr_2D,
    _fill_distr_3D
)
from .ship_setup import (
    z_min,
    z_max,
    x_max,
    y_max,
    theta_max_dec_vol
)

class Grids:
    """
    A class used to represent the grid and perform interpolations for particle distribution.

    Attributes
    ----------
    Distr : pd.DataFrame
        Dataframe containing the particle distribution.
    Energy_distr : pd.DataFrame
        Dataframe containing the energy distribution.
    nPoints : int
        Number of random points for interpolation.
    m : float
        Mass of the particle.
    c_tau : float
        Lifetime (c*tau) of the particle.
    thetamin : float
        Minimum angle in the distribution.
    theta_max : float
        Maximum angle in the distribution, set as min(self.Distr[1].max(), theta_max_dec_vol).
    grid_m : np.ndarray
        Unique mass grid values.
    grid_a : np.ndarray
        Unique angle grid values.
    energy_distr : np.ndarray
        2D grid of energy distribution values.
    grid_x : np.ndarray
        Unique x-coordinates of the grid.
    grid_y : np.ndarray
        Unique y-coordinates of the grid.
    grid_z : np.ndarray
        Unique z-coordinates of the grid.
    distr : np.ndarray
        3D grid of distribution values.
    theta : np.ndarray
        Array of sampled angles.
    mass : np.ndarray
        Array of sampled mass values.
    max_energy : np.ndarray
        Array of interpolated maximum energy values.
    energy : np.ndarray
        Array of sampled energy values.
    e_min_sampling : np.ndarray
        Array of minimum energy sampling values.
    interpolated_values : np.ndarray
        Array of interpolated distribution values.
    rsample_size : int
        Number of resampled points.
    true_points_indices : np.ndarray
        Indices of true points after resampling.
    r_theta : np.ndarray
        Array of resampled angles.
    r_energy : np.ndarray
        Array of resampled energy values.
    phi : np.ndarray
        Array of sampled phi angles.
    kinematics_dic : dict
        Dictionary of kinematic properties of the particles.
    """

    def __init__(self, Distr, Energy_distr, nPoints, mass, c_tau):
        """
        Initialize the Grids class with distributions, number of points, mass, and lifetime.

        Parameters
        ----------
        Distr : pd.DataFrame
            Dataframe containing the particle distribution.
        Energy_distr : pd.DataFrame
            Dataframe containing the energy distribution.
        nPoints : int
            Number of random points for interpolation.
        mass : float
            Mass of the particle.
        c_tau : float
            Lifetime (c*tau) of the particle.
        """
        self.Distr = Distr
        self.Energy_distr = Energy_distr
        self.nPoints = nPoints
        self.m = mass
        self.c_tau = c_tau

        # Angle range
        self.thetamin = self.Distr[1].min()
        self.theta_max = min(self.Distr[1].max(), theta_max_dec_vol)

        # Grids

        # Prepare the grid and distribution values for max energy
        self.grid_m = np.unique(self.Energy_distr.iloc[:, 0])
        self.grid_a = np.unique(self.Energy_distr.iloc[:, 1])

        # Filling 2D grid with values of the original distribution `Energy_distr`
        self.energy_distr = _fill_distr_2D(self.grid_m, self.grid_a, self.Energy_distr)

        # Prepare the grid and distribution values for distribution
        self.grid_x = np.unique(self.Distr.iloc[:, 0])
        self.grid_y = np.unique(self.Distr.iloc[:, 1])
        self.grid_z = np.unique(self.Distr.iloc[:, 2])

        # Filling 3D grid with values of the original distribution `Distr`
        self.distr = _fill_distr_3D(self.grid_x, self.grid_y, self.grid_z, self.Distr)

    def interpolate(self, timing=False):
        """
        Perform bilinear and trilinear interpolation on the distribution and energy grids.

        Parameters
        ----------
        timing : bool, optional
            If True, print the time taken for interpolation. Default is False.
        """
        if timing:
            t = time.time()

        # Sample angle points
        self.theta = np.random.uniform(self.thetamin, self.theta_max, self.nPoints)
        self.mass = self.m * np.ones(self.nPoints)

        # Interpolated values for max energy
        point_bilinear_interpolation = np.column_stack((self.mass, self.theta))
        self.max_energy = _bilinear_interpolation(
            point_bilinear_interpolation, self.grid_m, self.grid_a, self.energy_distr
        )

        # Define e_min_sampling. Depends on the LLP lifetime and mass.
        # If it is too small, to get good sampling, one needs to sample only the energies
        # for which Exp[-z_min/(c*tau*p/m)] is not tiny.
        # So the minimal sampled energy is fixed to be such that the exponent is at least Exp[-15]
        self.e_min_sampling = np.maximum(
            self.m,
            np.minimum(2.133 * self.m / self.c_tau, 0.5 * self.max_energy)
        )

        # Sample energy and define interpolation points
        self.energy = np.random.uniform(self.e_min_sampling, self.max_energy)
        points_to_interpolate = np.column_stack((self.mass, self.theta, self.energy))

        # Interpolated values for particle distribution
        self.interpolated_values = _trilinear_interpolation(
            points_to_interpolate,
            self.grid_x,
            self.grid_y,
            self.grid_z,
            self.distr,
            self.max_energy
        )

        if timing:
            print(f"\nInterpolation time t = {time.time() - t} s")

    def resample(self, rsample_size, timing=False):
        """
        Resample points based on the interpolated distribution values.

        Parameters
        ----------
        rsample_size : int
            Number of points to resample.
        timing : bool, optional
            If True, print the time taken for resampling. Default is False.
        """
        if timing:
            t = time.time()

        self.rsample_size = rsample_size
        weights = self.interpolated_values * (self.max_energy - self.e_min_sampling)
        self.weights = weights  # Store weights for later use
        weights_sum = np.sum(weights)
        if weights_sum == 0:
            raise ValueError("Sum of weights is zero. Cannot perform resampling.")
        probabilities = weights / weights_sum
        self.true_points_indices = np.random.choice(
            self.nPoints, size=self.rsample_size, p=probabilities
        )

        # Resampled angles and energies
        self.r_theta = self.theta[self.true_points_indices]
        self.r_energy = self.energy[self.true_points_indices]
        
        # Combine theta and energy into a two-column array
        resampled_data = np.column_stack((self.r_theta, self.r_energy))

        # Save to a text file without headers
        # np.savetxt('resampled_data.txt', resampled_data, fmt='%.6f', delimiter=' ')

        # Compute epsilon_polar
        self.epsilon_polar = np.sum(weights) * (self.theta_max - self.thetamin) / len(weights)

        if timing:
            print(f"Resample angle and energy  t = {time.time() - t} s")

    def true_samples(self, timing=False):
        """
        Calculate true samples of kinematic properties and decay probabilities.

        Parameters
        ----------
        timing : bool, optional
            If True, print the time taken for sampling vertices. Default is False.
        """
        if timing:
            t = time.time()

        # Angle phi sampling
        self.phi = np.random.uniform(-np.pi, np.pi, len(self.true_points_indices))

        # Momentum calculation
        momentum = np.sqrt(self.r_energy ** 2 - (self.m * np.ones_like(self.true_points_indices)) ** 2)
        px = momentum * np.cos(self.phi) * np.sin(self.r_theta)
        py = momentum * np.sin(self.phi) * np.sin(self.r_theta)
        pz = momentum * np.cos(self.r_theta)

        # Z values
        cmin = 1 - np.exp(-z_min * self.m / (np.cos(self.r_theta) * self.c_tau * momentum))
        cmax = 1 - np.exp(-z_max * self.m / (np.cos(self.r_theta) * self.c_tau * momentum))
        c = np.random.uniform(cmin, cmax, size=self.rsample_size)
        safe_c = np.minimum(c, 0.9999999995)
        z_computed = np.cos(self.r_theta) * self.c_tau * (momentum / self.m) * np.log(1 / (1 - safe_c))

        # If c > 0.9999999995, set z = z_min
        # Otherwise, compute z using the original formula
        z = np.where(
            c > 0.9999999995,
            z_min,
            z_computed
        )
        # X and Y values
        x = z * np.cos(self.phi) * np.tan(self.r_theta)
        y = z * np.sin(self.phi) * np.tan(self.r_theta)

        # Decay probability
        P_decay = (
            np.exp(-z_min * self.m / (np.cos(self.r_theta) * self.c_tau * momentum))
            - np.exp(-z_max * self.m / (np.cos(self.r_theta) * self.c_tau * momentum))
        )

        # Mask for particles decaying inside the volume
        mask = (
            (-x_max(z) < x)
            & (x < x_max(z))
            & (-y_max(z) < y)
            & (y < y_max(z))
            & (z_min <= z)
            & (z <= z_max)
        )

        self.kinematics_dic = {
            "px": px[mask],
            "py": py[mask],
            "pz": pz[mask],
            "energy": self.r_energy[mask],
            "m": self.m * np.ones_like(px[mask]),
            "PDG": 12345678 * np.ones_like(px[mask]),
            "P_decay": P_decay[mask],
            "x": x[mask],
            "y": y[mask],
            "z": z[mask]
        }

        if timing:
            print(f"Sampling vertices t = {time.time() - t} s")

        self.momentum = np.column_stack((px[mask], py[mask], pz[mask], self.r_energy[mask]))

    def get_kinematics(self):
        dic = self.kinematics_dic
        kinematics = np.column_stack(list(dic.values()))
        return kinematics

    def save_kinematics(self, path, name):
        """
        Save the kinematic properties to a CSV file.

        Parameters
        ----------
        path : str
            Directory path to save the kinematics file.
        name : str
            Name prefix for the kinematics file.
        """
        kinetics_df = pd.DataFrame(self.kinematics_dic)
        kinetics_df.to_csv(f"{path}/{name}_kinematics_sampling.dat", sep="\t", index=False)

    def get_energy(self):
        """
        Retrieve the resampled energy values stored in the instance.

        Returns
        -------
        np.ndarray or None
            The array of energy values if set, otherwise None.
        """
        return self.r_energy

    def get_theta(self):
        """
        Retrieve the resampled angle (theta) values stored in the instance.

        Returns
        -------
        np.ndarray or None
            The array of angle values if set, otherwise None.
        """
        return self.r_theta

    def get_momentum(self):
        """
        Retrieve the 4-momentum stored in the instance.

        Returns
        -------
        np.ndarray or None
            The 4-momentum, otherwise None.
        """
        return self.momentum

