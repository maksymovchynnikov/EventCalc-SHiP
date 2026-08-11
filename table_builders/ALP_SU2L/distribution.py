import numpy as np

from .config import (
    EMAX_RELATIVE_THRESHOLD,
    EMAX_THETA_PADDING_BINS,
)


def polar_angle_from_momenta(momenta):
    """
    Return the lab-frame polar angle for four-momenta with columns
    px, py, pz, E, ...
    """
    momenta = np.asarray(momenta, dtype=np.float64)

    px = momenta[:, 0]
    py = momenta[:, 1]
    pz = momenta[:, 2]

    pT = np.sqrt(px**2 + py**2)
    theta = np.arctan2(pT, pz)

    return theta


def theta_energy_from_momenta(alp_lab):
    """
    Convert lab-frame ALP momenta to theta_a and E_a.
    """
    theta = polar_angle_from_momenta(alp_lab)
    energy = alp_lab[:, 3]

    return theta, energy


def make_distribution_table_from_histogram(
    alp_mass,
    hist,
    number_generated,
    theta_edges,
    energy_edges,
):
    """
    Construct the normalized distribution table from an accumulated
    theta-energy histogram.

    This allows several independent B -> X + a decay simulations
    to be accumulated before normalization.
    """
    hist = np.asarray(
        hist,
        dtype=np.float64,
    )

    number_generated = int(number_generated)

    if number_generated <= 0:
        raise ValueError("number_generated must be positive.")

    expected_shape = (
        len(theta_edges) - 1,
        len(energy_edges) - 1,
    )

    if hist.shape != expected_shape:
        raise ValueError("Histogram shape does not match the supplied bin edges.")

    dtheta = np.diff(theta_edges)

    dE = np.diff(energy_edges)

    density = hist / (number_generated * dtheta[:, None] * dE[None, :])

    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    energy_centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])

    number_theta_bins = len(theta_centers)

    number_energy_bins = len(energy_centers)

    table = np.column_stack(
        (
            np.full(
                number_theta_bins * number_energy_bins,
                float(alp_mass),
            ),
            np.repeat(
                theta_centers,
                number_energy_bins,
            ),
            np.tile(
                energy_centers,
                number_theta_bins,
            ),
            density.ravel(),
        )
    )

    return table, hist, density


def make_distribution_table(
    alp_mass,
    theta,
    energy,
    theta_edges,
    energy_edges,
):
    """
    Build table with columns:
        ma, theta_center, E_center, d2f/(dtheta dE)
    """
    hist, _, _ = np.histogram2d(
        theta,
        energy,
        bins=(
            theta_edges,
            energy_edges,
        ),
    )

    return make_distribution_table_from_histogram(
        alp_mass=alp_mass,
        hist=hist,
        number_generated=len(theta),
        theta_edges=theta_edges,
        energy_edges=energy_edges,
    )


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

    return np.column_stack(
        (
            np.full(len(theta_centers), alp_mass),
            theta_centers,
            np.full(len(theta_centers), energy_edges[-1]),
        )
    )


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

    base_edges = np.asarray(
        edges,
        dtype=float,
    )

    mass_threshold_edges = np.asarray(
        masses,
        dtype=float,
    )

    all_edges = np.concatenate(
        (
            base_edges,
            mass_threshold_edges,
        )
    )

    all_edges = all_edges[(all_edges >= energy_min) & (all_edges <= energy_max)]

    return np.unique(np.sort(all_edges))


def truncate_table_in_theta(table, theta_cut, theta_column=1):
    """
    Keep only rows with theta_center < theta_cut.
    Important: This does not renormalize the density.
    """
    return table[table[:, theta_column] < theta_cut]


def integrate_density(density_full, theta_edges_full, energy_edges):
    return np.sum(
        density_full * np.diff(theta_edges_full)[:, None] * np.diff(energy_edges)[None, :]
    )
