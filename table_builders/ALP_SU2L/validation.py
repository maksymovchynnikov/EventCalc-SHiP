import numpy as np
import os
import matplotlib.pyplot as plt

from .constants import THETA_MAX_TABLE, THETA_MAX_SHIP


def validate_with_eventcalc_interpolation(
    distribution_table,
    emax_table,
    alp_mass,
    theta_max_sim,
    n_points=100000,  # so if n_points= NONE then this is n_points=100000?
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
    fraction_a_ship,
    fraction_B_ship,
    plot_folder,
    ifShowPlots,
):
    os.makedirs(plot_folder, exist_ok=True)
    mass_label = f"ma_{alp_mass:.6g}".replace(".", "p")
    plot_title = (
        rf"$m_a = {alp_mass:.3g}\,\mathrm{{GeV}}$"
        + "\n"
        + rf"$f_a(\theta_a < \theta_{{\rm SHiP\,limit}})"
        + rf" = {fraction_a_ship:.3f}$"
        + "\n"
        + rf"$f_B(\theta_B < \theta_{{\rm SHiP\,limit}})"
        + rf" = {fraction_B_ship:.3f}$"
    )

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    ax.hist(theta, bins=100)
    ax.axvline(THETA_MAX_SHIP, linestyle="--", label="SHiP limit")
    ax.axvline(THETA_MAX_TABLE, linestyle=":", label="SHiP limit + margin")
    ax.legend()
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

    # PLOT: theta_a_energy_log 
    theta_edges_plot = np.geomspace(1.0e-6, np.pi, 241)
    energy_edges_plot = np.geomspace(max(alp_mass, np.min(energy)), 1.001 * np.max(energy), 181)

    hist_plot, _, _ = np.histogram2d(
        theta,
        energy,
        bins=(theta_edges_plot, energy_edges_plot),
    )

    density_for_plot = hist_plot / (
        len(theta) * np.diff(theta_edges_plot)[:, None] * np.diff(energy_edges_plot)[None, :]
    )

    density_plot = np.full_like(density_for_plot, np.nan, dtype=float)
    mask = density_for_plot > 0.0
    density_plot[mask] = np.log10(density_for_plot[mask])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    mesh = ax.pcolormesh(
        theta_edges_plot,
        energy_edges_plot,
        density_plot.T,
        shading="flat",
        rasterized=True,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(max(0.95 * alp_mass, energy_edges_plot[0]), energy_edges_plot[-1])
    ax.axvline(
        THETA_MAX_SHIP,
        linestyle="--",
        linewidth=1.2,
        label=rf"SHiP limit: ${THETA_MAX_SHIP:.5f}$ rad",
    )
    ax.axvline(
        THETA_MAX_TABLE,
        linestyle=":",
        linewidth=1.2,
        label=rf"SHiP limit + margin: ${THETA_MAX_TABLE:.5f}$ rad",
    )
    ax.set_xlabel(r"$\theta_a$ [rad]")
    ax.set_ylabel(r"$E_a$ [GeV]")
    ax.legend(loc="lower left")
    ax.set_title(plot_title)
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label(
        r"$\log_{10}\left["
        r"\mathrm{d}^2f_a/"
        r"(\mathrm{d}\theta_a\,\mathrm{d}E_a)"
        r"\right]$"
        r" [GeV$^{-1}$ rad$^{-1}$]"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(plot_folder, f"theta_a_energy_log_{mass_label}.png"), dpi=300)
    if ifShowPlots:
        plt.show()
    else:
        plt.close(fig)


def plot_B_theta_energy_distribution(
    theta_B,
    energy_B,
    fraction_B_ship,
    plot_folder,
    ifShowPlots,
):
    os.makedirs(plot_folder, exist_ok=True)

    theta_edges_plot = np.geomspace(1.0e-6, np.pi, 351)
    energy_edges_B = np.geomspace(np.min(energy_B), 1.001 * np.max(energy_B), 241)

    hist_B, _, _ = np.histogram2d(
        theta_B,
        energy_B,
        bins=(theta_edges_plot, energy_edges_B),
    )

    density_B = hist_B / (
        len(theta_B) * np.diff(theta_edges_plot)[:, None] * np.diff(energy_edges_B)[None, :]
    )

    density_plot = np.full_like(density_B, np.nan, dtype=float)
    mask = density_B > 0.0
    density_plot[mask] = np.log10(density_B[mask])
    #theta_edges_plot[0] = max(theta_edges_plot[1] * 1.0e-3, 1.0e-8)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    mesh = ax.pcolormesh(
        theta_edges_plot,
        energy_edges_B,
        density_plot.T,
        shading="flat",
        rasterized=True,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(
        THETA_MAX_SHIP,
        linestyle="--",
        linewidth=1.2,
        label=rf"SHiP limit: ${THETA_MAX_SHIP:.5f}$ rad",
    )
    ax.axvline(
        THETA_MAX_TABLE,
        linestyle=":",
        linewidth=1.2,
        label=rf"SHiP limit + margin: ${THETA_MAX_TABLE:.5f}$ rad",
    )
    ax.set_xlabel(r"$\theta_B$ [rad]")
    ax.set_ylabel(r"$E_B$ [GeV]")

    ax.set_title(
        r"Input $B$-meson distribution"
        + "\n"
        + rf"$f_B(\theta_B < \theta_{{\rm SHiP\,limit}})"
        + rf" = {fraction_B_ship:.3f}$"
    )

    ax.legend(loc="lower left")
    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label(
        r"$\log_{10}\left["
        r"\mathrm{d}^2f_B/"
        r"(\mathrm{d}\theta_B\,\mathrm{d}E_B)"
        r"\right]$"
        r" [GeV$^{-1}$ rad$^{-1}$]"
    )

    fig.tight_layout()
    fig.savefig(
        os.path.join(plot_folder, "theta_B_energy_log.png"),
        dpi=300,
    )
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

    dtheta = np.diff(theta_edges)
    dE = np.diff(energy_edges)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    # Full angular range:
    dfdE_full = np.sum(density * dtheta[:, None], axis=0)

    # SHiP-relevant angular range:
    theta_mask_ship = theta_centers < THETA_MAX_SHIP
    theta_mask_table = theta_centers < THETA_MAX_TABLE
    dfdE_ship = np.sum(density[theta_mask_ship, :] * dtheta[theta_mask_ship, None], axis=0)
    dfdE_table = np.sum(density[theta_mask_table, :] * dtheta[theta_mask_table, None], axis=0)

    integral_full = np.sum(dfdE_full * dE)
    integral_ship = np.sum(dfdE_ship * dE)
    integral_table = np.sum(dfdE_table * dE)

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
        label=rf"$\theta_a < {THETA_MAX_SHIP:.4g}$ rad, integral = {integral_ship:.3f}",
    )
    ax.stairs(
        dfdE_table,
        energy_edges,
        label=rf"$\theta_a < {THETA_MAX_TABLE:.4g}$ rad, integral = {integral_table:.3f}",
    )

    ax.set_xlabel(r"$E_a$ [GeV]")
    ax.set_ylabel(r"$df_a/dE_a$ [GeV$^{-1}$]")
    ax.set_yscale("log")
    ax.set_title(rf"$m_a = {alp_mass:.3g}$ GeV")
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        os.path.join(plot_folder, f"dFdE_lin_{mass_label}.png"),
        dpi=300,
    )

    if ifShowPlots:
        plt.show()
    else:
        plt.close(fig)
