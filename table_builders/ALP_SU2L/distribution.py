import numpy as np

from .config import (
    EMAX_RELATIVE_THRESHOLD,
    EMAX_THETA_PADDING_BINS,
)

def theta_energy_from_momenta(alp_lab):
    """
    Convert lab-frame ALP momenta to theta_a and E_a.
    """
    px = alp_lab[:, 0]
    py = alp_lab[:, 1]
    pz = alp_lab[:, 2]
    energy = alp_lab[:, 3]

    pT = np.sqrt(px**2 + py**2)
    theta = np.arctan2(pT, pz)

    return theta, energy


def make_distribution_table(alp_mass, theta, energy, theta_edges, energy_edges):
    """
    Build table with columns:
        ma, theta_center, E_center, d2f/(dtheta dE)

    The distribution is normalized to the total number of simulated ALPs.
    """
    hist, _, _ = np.histogram2d(theta, energy, bins=(theta_edges, energy_edges))

    size_total = len(theta)
    dtheta = np.diff(theta_edges)
    dE = np.diff(energy_edges)

    density = hist / (size_total * dtheta[:, None] * dE[None, :])

    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    energy_centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])

    rows = []
    for i, theta_i in enumerate(theta_centers):
        for j, energy_j in enumerate(energy_centers):
            rows.append([alp_mass, theta_i, energy_j, density[i, j]])

    return np.array(rows), hist, density


def make_emax_table(
    alp_mass,
    theta_edges,
    energy_edges,
    density,
    relative_threshold=EMAX_RELATIVE_THRESHOLD,
    theta_padding_bins=EMAX_THETA_PADDING_BINS,
):
    """
    Build table with columns:
        ma, theta_center, Emax(theta)

    Emax(theta) is defined by the local relative-threshold prescription:
    in a small theta layer around theta_i, keep energies where

        density(E, theta layer) > relative_threshold * max(density in theta layer).

    This implements the 10^(-XX) rule with XX = 7 by default.
    """
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    rows = []
    for i, theta_i in enumerate(theta_centers):
        i_min = max(0, i - theta_padding_bins)
        i_max = min(len(theta_centers), i + theta_padding_bins + 1)

        # Local theta layer: use the maximum density over nearby theta bins.
        density_i = np.max(density[i_min:i_max, :], axis=0)
        density_max = np.max(density_i)

        if density_max <= 0.0:
            Emax_i = alp_mass
        else:
            mask = density_i > relative_threshold * density_max

            if np.any(mask):
                last_index = np.where(mask)[0][-1]
                Emax_i = energy_edges[last_index + 1]
            else:
                Emax_i = alp_mass

        rows.append([alp_mass, theta_i, max(Emax_i, alp_mass)])

    return np.array(rows)


def make_constant_emax_table(alp_mass, theta_edges, energy_edges):
    """
    Conservative Emax table used only for validation.
    """
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    return np.column_stack((
        np.full(len(theta_centers), alp_mass),
        theta_centers,
        np.full(len(theta_centers), energy_edges[-1]),
    ))


def make_theta_edges(theta_max_out, n_theta_forward, n_theta_tail):
    """
    Make a theta grid dense in the SHiP-relevant forward region and still cover 0 < theta < pi.
    """
    theta_edges_forward = np.linspace(0.0, theta_max_out, n_theta_forward + 1)
    theta_edges_tail = np.linspace(theta_max_out, np.pi, n_theta_tail + 1)

    return np.unique(np.concatenate((theta_edges_forward, theta_edges_tail[1:])))


def make_energy_edges(masses, B_momenta):
    """
    Make a non-uniform energy grid.

    The density calculation divides by dE, so variable bin widths are allowed.
    """
    energy_min = np.min(masses)
    energy_max = 1.001 * np.max(B_momenta[:, 3])

    edges = [energy_min]

    segments = [
        (5.0, 12),
        (20.0, 18),
        (40.0, 18),
        (70.0, 36),
        (120.0, 20),
        (energy_max, 16),
    ]

    for stop, n_bins in segments:
        start = edges[-1]
        stop = min(stop, energy_max)

        if stop <= start:
            continue

        new_edges = np.linspace(start, stop, n_bins + 1)
        edges.extend(new_edges[1:])

        if stop >= energy_max:
            break

    return np.array(edges)

def truncate_table_in_theta(table, theta_cut, theta_column=1):
    """
    Keep only rows with theta_center < theta_cut.
    Important: This does not renormalize the density.
    """
    return table[table[:, theta_column] < theta_cut]

def integrate_density(density_full, theta_edges_full, energy_edges):
    return np.sum(
            density_full
            * np.diff(theta_edges_full)[:, None]
            * np.diff(energy_edges)[None, :]
        )