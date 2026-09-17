"""Flat Lorentz-invariant phase space for explicit five-or-more-body decays.

The returned layout matches the other EventCalc decay kernels:
``[px, py, pz, E, mass, pdg, charge, stability]`` for each daughter.
"""

import numpy as np

from .sampling_bounds import chain_mass_sampler
from .thresholds import safe_momentum_squared


def _breakup_momentum(parent, left, right):
    """Vectorized two-body breakup momentum."""
    parent = np.asarray(parent, dtype=float)
    s = np.square(parent)
    plus = np.square(left + right)
    minus = np.square(left - right)
    # Each step of the chain places the parent mass above the sum of the two
    # masses it splits into, so a negative product here is floating-point
    # roundoff and nothing else; safe_momentum_squared clamps that and raises on
    # anything larger.
    radicand = safe_momentum_squared((s - plus) * (s - minus), np.square(s))
    return np.sqrt(radicand) / (2.0 * parent)


def _boost(p4, frame):
    """Boost ``p4`` from a subsystem rest frame into ``frame``."""
    beta = frame[:, :3] / frame[:, 3, None]
    beta2 = np.sum(np.square(beta), axis=1)
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - beta2, 1.0e-30))
    bp = np.sum(beta * p4[:, :3], axis=1)
    factor = np.zeros_like(beta2)
    moving = beta2 > 1.0e-28
    factor[moving] = (
        (gamma[moving] - 1.0) * bp[moving] / beta2[moving]
        + gamma[moving] * p4[moving, 3]
    )
    out = np.empty_like(p4)
    out[:, :3] = p4[:, :3] + factor[:, None] * beta
    out[:, 3] = gamma * (p4[:, 3] + bp)
    return out


def _intermediate_masses(mother_mass, masses, n_candidates, rng):
    """Draw Raubold--Lynch intermediate masses and phase-space weights.

    The intermediate masses come from the kinetic energy of the parent shared
    out by sorted uniform draws, and the weight of each set is the product of
    the momenta the chain releases.  This is the reference form of the chain
    weight that ``funcs.sampling_bounds.chain_weight_bound`` bounds.
    """
    n_daughters = len(masses)
    kinetic = mother_mass - float(np.sum(masses))
    random = np.sort(
        rng.random((n_candidates, n_daughters - 2)), axis=1
    )
    intermediate = np.empty((n_candidates, n_daughters), dtype=float)
    intermediate[:, 0] = masses[0]
    cumulative = np.cumsum(masses)
    for index in range(1, n_daughters - 1):
        intermediate[:, index] = (
            cumulative[index] + kinetic * random[:, index - 1]
        )
    intermediate[:, -1] = mother_mass

    weight = np.ones(n_candidates, dtype=float)
    for index in range(1, n_daughters):
        weight *= _breakup_momentum(
            intermediate[:, index],
            masses[index],
            intermediate[:, index - 1],
        )
    return intermediate, weight


def decay_products(
    mother_mass,
    pdgs,
    masses,
    charges,
    stabilities,
    n_events,
    rng=None,
):
    """Generate a massive flat-LIPS sample for five or more daughters."""
    mother_mass = float(mother_mass)
    n_events = int(n_events)
    pdgs = np.asarray(pdgs, dtype=int)
    masses = np.asarray(masses, dtype=float)
    charges = np.asarray(charges, dtype=float)
    stabilities = np.asarray(stabilities, dtype=float)
    n_daughters = len(pdgs)
    if rng is None:
        rng = np.random

    if n_daughters < 5:
        raise ValueError(
            "N-body phase-space generation requires at least five daughters"
        )
    if not (
        len(masses)
        == len(charges)
        == len(stabilities)
        == n_daughters
    ):
        raise ValueError("N-body particle-property arrays have different lengths")
    if n_events < 0:
        raise ValueError("number of N-body events must be non-negative")

    threshold = float(np.sum(masses))
    if mother_mass + 1.0e-12 < threshold:
        raise ValueError(
            "N-body channel is below threshold: M=%g < sum(m_i)=%g"
            % (mother_mass, threshold)
        )
    if n_events == 0:
        return np.empty((0, 8 * n_daughters), dtype=float)
    if mother_mass - threshold <= 1.0e-14:
        raise ValueError("N-body phase space has zero volume at threshold")

    # The intermediate masses come from chain_mass_sampler, which cuts the range
    # of each of them into cells and carries, for each combination of cells, a
    # value the flat-phase-space density provably never exceeds inside it.
    # Combinations are drawn in proportion to that value times their volume and
    # kept with probability density over value, so every returned event is an
    # independent draw from flat phase space, none is a duplicate of another,
    # and the number of events asked for is not limited by any pool size.  The
    # table depends on the parent mass and the daughter masses alone, so it is
    # built once for a channel and reused for every event of it.
    sampler = chain_mass_sampler(mother_mass, tuple(masses))
    intermediate = np.empty((n_events, n_daughters), dtype=float)
    intermediate[:, 0] = masses[0]
    intermediate[:, 1:n_daughters - 1] = sampler.draw(n_events, rng)
    intermediate[:, -1] = mother_mass

    subsystem = np.zeros((n_events, 4), dtype=float)
    subsystem[:, 3] = mother_mass
    four_vectors = np.empty(
        (n_events, n_daughters, 4), dtype=float
    )
    for index in range(n_daughters - 1, 0, -1):
        parent_mass = intermediate[:, index]
        residual_mass = intermediate[:, index - 1]
        momentum = _breakup_momentum(
            parent_mass, masses[index], residual_mass
        )
        cosine = rng.uniform(-1.0, 1.0, n_events)
        sine = np.sqrt(np.maximum(1.0 - np.square(cosine), 0.0))
        phi = rng.uniform(0.0, 2.0 * np.pi, n_events)
        vector = np.column_stack(
            (
                momentum * sine * np.cos(phi),
                momentum * sine * np.sin(phi),
                momentum * cosine,
            )
        )
        daughter_rest = np.column_stack(
            (
                vector,
                np.sqrt(np.square(momentum) + masses[index] ** 2),
            )
        )
        residual_rest = np.column_stack(
            (
                -vector,
                np.sqrt(np.square(momentum) + np.square(residual_mass)),
            )
        )
        four_vectors[:, index, :] = _boost(daughter_rest, subsystem)
        subsystem = _boost(residual_rest, subsystem)
    four_vectors[:, 0, :] = subsystem

    output = np.empty((n_events, n_daughters, 8), dtype=float)
    output[:, :, :4] = four_vectors
    output[:, :, 4] = masses[None, :]
    output[:, :, 5] = pdgs[None, :]
    output[:, :, 6] = charges[None, :]
    output[:, :, 7] = stabilities[None, :]
    return output.reshape(n_events, 8 * n_daughters)
