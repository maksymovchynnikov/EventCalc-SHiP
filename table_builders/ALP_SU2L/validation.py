import numpy as np
import os
import matplotlib.pyplot as plt

from .constants import (THETA_CUT_FINAL)

def validate_with_eventcalc_interpolation(
    distribution_table,
    emax_table,
    alp_mass,
    theta_max_sim,
    n_points=100000, # so if n_points= NONE then this is n_points=100000?
):
    """
    Validate the table using EventCalc's interpolation machinery.
    """
    import pandas as pd
    from funcs import kinematics

    Distr = pd.DataFrame(distribution_table)
    Energy_distr = pd.DataFrame(emax_table)

    grids = kinematics.Grids(
        Distr,
        Energy_distr,
        n_points,
        alp_mass,
        c_tau=1e99,
        theta_max_sim=theta_max_sim,
    )
    grids.interpolate()

    weights = grids.interpolated_values * (grids.max_energy - grids.e_min_sampling)
    theta_range = grids.theta_max - grids.thetamin

    integral = np.mean(weights) * theta_range
    mc_error = np.std(weights) / np.sqrt(n_points) * theta_range

    return integral, mc_error

def plot_debug_distributions(
    alp_mass,
    theta,
    energy,
    theta_edges,
    energy_edges,
    density,
    emax_table,
    fraction_out,
    plot_folder,
    ifShowPlots,
):
    """
    Make simple plots for debugging.
    """
    os.makedirs(plot_folder, exist_ok=True)

    mass_label = f"ma_{alp_mass:.6g}".replace(".", "p")
    theta_cut = THETA_CUT_FINAL
    plot_title = (
        rf"$m_a = {alp_mass:.3g}$ GeV"
        + "\n"
        + rf"$f(\theta_a < {theta_cut:.4g}\,\mathrm{{rad}}) = {fraction_out:.3e}$"
    )

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.hist(theta, bins=100)
    ax.axvline(theta_cut, linestyle="--", color='C1')
    ax.set_xlabel(r"$\theta_a$ [rad]")
    ax.set_ylabel("Counts")
    ax.set_title(plot_title)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_folder, f"theta_{mass_label}.png"), dpi=300)
    if ifShowPlots:
        plt.show()
    else:
        plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.hist(energy, bins=100)
    ax.set_xlabel(r"$E_a$ [GeV]")
    ax.set_ylabel("Counts")
    ax.set_title(plot_title)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_folder, f"energy_{mass_label}.png"), dpi=300)
    if ifShowPlots:
        plt.show()
    else:
        plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    density_plot = np.full_like(density, np.nan, dtype=float)
    mask = density > 0.0
    density_plot[mask] = np.log10(density[mask])

    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    energy_centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])

    mesh = ax.pcolormesh(theta_centers, energy_centers, density_plot.T, shading="auto")
    ax.plot(emax_table[:, 1], emax_table[:, 2])
    ax.set_xlabel(r"$\theta_a$ [rad]")
    ax.set_ylabel(r"$E_a$ [GeV]")
    ax.set_title(plot_title)
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label(r"$\log_{10}\left[f_a\;(\mathrm{GeV}^{-1}\,\mathrm{rad}^{-1})\right]$")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_folder, f"theta_energy_{mass_label}.png"), dpi=300)
    if ifShowPlots:
        plt.show()
    else:
        plt.close(fig)



def plot_energy_spectrum_from_density(
    alp_mass,
    theta_edges,
    energy_edges,
    density,
    plot_folder,
    ifShowPlots,
):
    """
    Plot df_a/dE_a by integrating d2f_a/(dtheta dE) over theta.

    This correctly accounts for non-uniform bin sizes.
    """
    os.makedirs(plot_folder, exist_ok=True)

    mass_label = f"ma_{alp_mass:.6g}".replace(".", "p")
    theta_cut = THETA_CUT_FINAL

    dtheta = np.diff(theta_edges)
    dE = np.diff(energy_edges)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    # Full angular range:
    dfdE_full = np.sum(density * dtheta[:, None], axis=0)

    # SHiP-relevant angular range:
    theta_mask = theta_centers < theta_cut
    dfdE_ship = np.sum(density[theta_mask, :] * dtheta[theta_mask, None], axis=0)

    integral_full = np.sum(dfdE_full * dE)
    integral_ship = np.sum(dfdE_ship * dE)

    for yscale in ["linear", "log"]:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        ax.stairs(
            dfdE_full,
            energy_edges,
            label=rf"Full $\theta$ range, integral = {integral_full:.3f}",
        )
        ax.stairs(
            dfdE_ship,
            energy_edges,
            label=rf"$\theta_a < {theta_cut:.4g}$ rad, integral = {integral_ship:.3f}",
        )

        ax.set_xlabel(r"$E_a$ [GeV]")
        ax.set_ylabel(r"$df_a/dE_a$ [GeV$^{-1}$]")
        ax.set_yscale(yscale)
        ax.set_title(rf"$m_a = {alp_mass:.3g}$ GeV")
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            os.path.join(plot_folder, f"dFdE_{yscale}_{mass_label}.png"),
            dpi=300,
        )

        if ifShowPlots:
            plt.show()
        else:
            plt.close(fig)