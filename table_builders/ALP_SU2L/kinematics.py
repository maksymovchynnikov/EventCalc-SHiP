# rest-frame decay + boost

import numpy as np

from .channels_file import (
    BPLUS_TO_XA_CHANNELS,
    M_B_PLUS,
    PDG_ALP,
    get_allowed_channels,
)

from .config import (rng)


def simulate_B_to_Xa_rest_frame_fast(
    size,
    alp_mass,
    probabilities_by_name,
    channels=BPLUS_TO_XA_CHANNELS,
):
    """
    Vectorized simulation of the ALP in B+ -> X_s + a in the B rest frame.

    This replaces the old simulate_B_to_Xa_rest_frame(...) function, but returns
    the same ALP-only array format:

        px, py, pz, E, mass, pdg

    The recoil particle is not stored because the later distribution only uses
    the ALP kinematics.
    """
    allowed_channels = get_allowed_channels(alp_mass, channels)

    if not allowed_channels:
        raise ValueError("No kinematically allowed channels.")

    probabilities = np.array(
        [probabilities_by_name[channel["name"]] for channel in allowed_channels],
        dtype=np.float64,
    )

    probability_sum = np.sum(probabilities)
    if probability_sum <= 0.0:
        raise ValueError("Total channel probability is zero.")

    probabilities = probabilities / probability_sum

    channel_indices = rng.choice(
        len(allowed_channels),
        size=int(size),
        p=probabilities,
    )

    alp_rest = np.empty((int(size), 6), dtype=np.float64)
    alp_rest[:, 4] = alp_mass
    alp_rest[:, 5] = PDG_ALP

    for i, channel in enumerate(allowed_channels):
        mask = channel_indices == i
        n_events = int(np.sum(mask))

        if n_events == 0:
            continue

        m_X = float(channel["mass"])

        # Two-body energy of particle 1 = ALP in the B rest frame.
        E_star = (M_B_PLUS**2 + alp_mass**2 - m_X**2) / (2.0 * M_B_PLUS)
        p2_star = E_star**2 - alp_mass**2
        p_star = np.sqrt(max(p2_star, 0.0))

        # Isotropic direction in the B rest frame.
        cos_theta_star = rng.uniform(-1.0, 1.0, n_events)
        phi_star = rng.uniform(0.0, 2.0 * np.pi, n_events)
        sin_theta_star = np.sqrt(1.0 - cos_theta_star**2)

        alp_rest[mask, 0] = p_star * sin_theta_star * np.cos(phi_star)
        alp_rest[mask, 1] = p_star * sin_theta_star * np.sin(phi_star)
        alp_rest[mask, 2] = p_star * cos_theta_star
        alp_rest[mask, 3] = E_star

    return alp_rest


def boost_alp_rest_to_lab_fast(m_mother, B_momenta, alp_rest):
    """
    Vectorized Lorentz boost of ALP four-momenta from the B rest frame to lab.

    This replaces boost.tab_boosted_decay_products(m_mother, B_momenta, alp_rest)
    for the special case where alp_rest contains only one daughter particle per
    event with columns:

        px, py, pz, E, mass, pdg

    It returns the same column convention.
    """
    B_momenta = np.asarray(B_momenta, dtype=np.float64)
    alp_rest = np.asarray(alp_rest, dtype=np.float64)

    if B_momenta.ndim != 2 or B_momenta.shape[1] != 4:
        raise ValueError("B_momenta must have shape (N, 4) with columns px, py, pz, E.")

    if alp_rest.ndim != 2 or alp_rest.shape[1] < 6:
        raise ValueError("alp_rest must have shape (N, >=6) with columns px, py, pz, E, mass, pdg.")

    if B_momenta.shape[0] != alp_rest.shape[0]:
        raise ValueError("B_momenta and alp_rest must contain the same number of events.")

    mother_p = B_momenta[:, :3]
    mother_E = B_momenta[:, 3]

    beta = mother_p / mother_E[:, None]
    beta2 = np.sum(beta * beta, axis=1)
    gamma = mother_E / float(m_mother)

    p_rest = alp_rest[:, :3]
    E_rest = alp_rest[:, 3]

    beta_dot_p = np.sum(beta * p_rest, axis=1)

    # For beta = 0 the factor is mathematically unused and should be zero.
    gamma_minus_one_over_beta2 = np.zeros_like(beta2)
    moving = beta2 > 0.0
    gamma_minus_one_over_beta2[moving] = (gamma[moving] - 1.0) / beta2[moving]

    E_lab = gamma * (E_rest + beta_dot_p)
    p_lab = (
        p_rest
        + gamma[:, None] * beta * E_rest[:, None]
        + gamma_minus_one_over_beta2[:, None] * beta * beta_dot_p[:, None]
    )

    alp_lab = np.empty_like(alp_rest)
    alp_lab[:, :3] = p_lab
    alp_lab[:, 3] = E_lab
    alp_lab[:, 4:] = alp_rest[:, 4:]

    return alp_lab
