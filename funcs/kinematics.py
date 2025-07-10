# funcs/kinematics.py
"""
Sampling, interpolation and re-weighting of LLP lab-frame kinematics.

Key addition
------------
Grids(..., theta_max_sim=<value>)

* `theta_max_sim` defaults to `theta_max_dec_vol` (detector acceptance);
  scripts can pass e.g. `math.pi / 2` to sample the full hemisphere.
"""

import numpy as np
import time
import pandas as pd
import numba as nb

from .interpolation_functions import (
    _searchsorted_opt,
    _bilinear_interpolation,
    _trilinear_interpolation,
    _fill_distr_2D,
    _fill_distr_3D,
)
from .ship_setup import (
    z_min,
    z_max,
    x_max,
    y_max,
    theta_max_dec_vol,   # default angular upper limit
)


class Grids:
    """
    Handle interpolation of tabulated (θ, E) distributions and resampling
    with geometry / weighting corrections.

    Parameters
    ----------
    Distr, Energy_distr : pandas.DataFrame
        Original tabulated 3-D and 2-D distributions.
    nPoints : int
        Raw Monte-Carlo points for the internal interpolation grid.
    mass : float
        Mass of the LLP in GeV.
    c_tau : float
        Proper decay length (m).
    theta_max_sim : float, optional
        *Upper* θ limit (rad) for random generation.  Defaults to
        `theta_max_dec_vol`, but callers may pass `np.pi/2` etc.
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        Distr,
        Energy_distr,
        nPoints,
        mass,
        c_tau,
        theta_max_sim=theta_max_dec_vol,
    ):
        self.Distr = Distr
        self.Energy_distr = Energy_distr
        self.nPoints = nPoints
        self.m = mass
        self.c_tau = c_tau

        # ----- θ range -------------------------------------------------
        self.thetamin = self.Distr[1].min()
        # honour either table limit or user limit, whichever is smaller
        self.theta_max = min(self.Distr[1].max(), theta_max_sim)

        # ----- build 2-D grid for max-energy interpolation -------------
        self.grid_m = np.unique(self.Energy_distr.iloc[:, 0])
        self.grid_a = np.unique(self.Energy_distr.iloc[:, 1])
        self.energy_distr = _fill_distr_2D(
            self.grid_m, self.grid_a, self.Energy_distr
        )

        # ----- build 3-D grid for full distribution -------------------
        self.grid_x = np.unique(self.Distr.iloc[:, 0])
        self.grid_y = np.unique(self.Distr.iloc[:, 1])
        self.grid_z = np.unique(self.Distr.iloc[:, 2])
        self.distr = _fill_distr_3D(
            self.grid_x, self.grid_y, self.grid_z, self.Distr
        )

    # ==================================================================
    # Interpolation of θ, E and weight grid
    # ==================================================================
    def interpolate(self, timing=False):
        if timing:
            t0 = time.time()

        # random θ in [θ_min, θ_max_sim]
        self.theta = np.random.uniform(self.thetamin, self.theta_max, self.nPoints)
        self.mass = self.m * np.ones(self.nPoints)

        # max energy E_max(m, θ) by bilinear interpolation
        points_2d = np.column_stack((self.mass, self.theta))
        self.max_energy = _bilinear_interpolation(
            points_2d, self.grid_m, self.grid_a, self.energy_distr
        )

        # minimal sampled energy to avoid tiny exponential weights
        self.e_min_sampling = np.maximum(
            self.m,
            np.minimum(2.133 * self.m / self.c_tau, 0.5 * self.max_energy),
        )

        # draw energies uniformly in [E_min, E_max]
        self.energy = np.random.uniform(self.e_min_sampling, self.max_energy)
        points_3d = np.column_stack((self.mass, self.theta, self.energy))

        # distribution weight f(m, θ, E)
        self.interpolated_values = _trilinear_interpolation(
            points_3d,
            self.grid_x,
            self.grid_y,
            self.grid_z,
            self.distr,
            self.max_energy,
        )

        if timing:
            print(f"\nInterpolation time: {time.time() - t0:.2f} s")

    # ==================================================================
    # Resampling with acceptance weights
    # ==================================================================
    def resample(self, rsample_size, timing=False):
        if timing:
            t0 = time.time()

        self.rsample_size = rsample_size
        weights = self.interpolated_values * (self.max_energy - self.e_min_sampling)
        self.weights = weights
        prob = weights / weights.sum()
        self.true_points_indices = np.random.choice(
            self.nPoints, size=self.rsample_size, p=prob
        )

        self.r_theta = self.theta[self.true_points_indices]
        self.r_energy = self.energy[self.true_points_indices]

        # acceptance ε_polar (same as original)
        self.epsilon_polar = (
            np.sum(weights) * (self.theta_max - self.thetamin) / len(weights)
        )

        if timing:
            print(f"Resample time: {time.time() - t0:.2f} s")

    # ==================================================================
    # Generate true decay vertices and momenta with SHiP cuts
    # ==================================================================
    def true_samples(self, timing=False):
        if timing:
            t0 = time.time()

        self.phi = np.random.uniform(-np.pi, np.pi, len(self.true_points_indices))
        momentum_abs = np.sqrt(self.r_energy**2 - self.m**2)

        px = momentum_abs * np.cos(self.phi) * np.sin(self.r_theta)
        py = momentum_abs * np.sin(self.phi) * np.sin(self.r_theta)
        pz = momentum_abs * np.cos(self.r_theta)

        # longitudinal decay position z
        cmin = 1 - np.exp(-z_min * self.m / (np.cos(self.r_theta) * self.c_tau * momentum_abs))
        cmax = 1 - np.exp(-z_max * self.m / (np.cos(self.r_theta) * self.c_tau * momentum_abs))
        c = np.random.uniform(cmin, cmax)
        safe_c = np.minimum(c, 0.9999999995)
        z = np.where(
            c > 0.9999999995,
            z_min,
            np.cos(self.r_theta) * self.c_tau * (momentum_abs / self.m) * np.log(1 / (1 - safe_c)),
        )

        # transverse decay coordinates
        x = z * np.cos(self.phi) * np.tan(self.r_theta)
        y = z * np.sin(self.phi) * np.tan(self.r_theta)

        # decay-inside-volume flag
        geom_acceptance = (
            (-x_max(z) < x)
            & (x < x_max(z))
            & (-y_max(z) < y)
            & (y < y_max(z))
            & (z_min <= z)
            & (z <= z_max)
        )

        P_decay = (
            np.exp(-z_min * self.m / (np.cos(self.r_theta) * self.c_tau * momentum_abs))
            - np.exp(-z_max * self.m / (np.cos(self.r_theta) * self.c_tau * momentum_abs))
        )

        self.kinematics_dic = {
            "px": px[geom_acceptance],
            "py": py[geom_acceptance],
            "pz": pz[geom_acceptance],
            "energy": self.r_energy[geom_acceptance],
            "m": self.m * np.ones_like(px[geom_acceptance]),
            "PDG": 12345678 * np.ones_like(px[geom_acceptance]),
            "P_decay": P_decay[geom_acceptance],
            "x": x[geom_acceptance],
            "y": y[geom_acceptance],
            "z": z[geom_acceptance],
        }

        self.momentum = np.column_stack(
            (
                px[geom_acceptance],
                py[geom_acceptance],
                pz[geom_acceptance],
                self.r_energy[geom_acceptance],
            )
        )

        if timing:
            print(f"Vertex sampling time: {time.time() - t0:.2f} s")

    # ==================================================================
    # Convenience getters
    # ==================================================================
    def get_kinematics(self):
        return np.column_stack(list(self.kinematics_dic.values()))

    def save_kinematics(self, path, name):
        pd.DataFrame(self.kinematics_dic).to_csv(
            f"{path}/{name}_kinematics_sampling.dat", sep="\t", index=False
        )

    def get_energy(self):
        return self.r_energy

    def get_theta(self):
        return self.r_theta

    def get_momentum(self):
        return self.momentum

