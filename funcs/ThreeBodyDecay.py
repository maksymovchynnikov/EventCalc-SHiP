import importlib

import numpy as np
from numba import njit
from numpy.random import uniform
from . import rotateVectors
from . import sampling_bounds
from .thresholds import (PARTON_PDGS, parton_system_threshold,
                         string_end_two_hadron_mass)

# Diagnostics of the exact sampler, readable by tests.
SAMPLER_STATS = {"negative_weight_points": 0, "candidates": 0, "accepted": 0,
                 "upper_bounds_built": 0}

# An upper bound is expensive to work out and depends only on the channel and
# the parent mass, so a bounded number of them is kept between calls.
_UPPER_BOUND_CACHE = {}
_UPPER_BOUND_CACHE_LIMIT = 64

# The minimum invariant mass one configured exHad release accepts for each
# parton pair, held per release root and built the first time a pair is looked
# up.
_RELEASE_PAIR_W_MIN = {}


def release_pair_w_min(pdg_pair):
    """The minimum invariant mass the configured exHad release accepts for a pair.

    The release checks the W of every quark-antiquark pair it is handed
    against a minimum of its own, per current, and refuses the whole request
    below it.  Those numbers are the release's own and are not recomputed
    here: they are not the two-hadron bound of ``funcs.thresholds`` (the
    release routes ``c sbar`` through ``D0 K+`` rather than ``Ds+ pi0``, and
    it hands the heavy currents to showered Pythia at their own W), and
    ``model_info`` does not carry them, so they are read from the release's
    own ``exhad.hnl`` through the binding ``funcs.exhad_release`` already
    imports the release with: ``PRIMARY_W_MIN`` for the currents the release
    hadronizes itself, ``PYTHIA_W_MIN`` for the ones it showers, and
    ``CC_PAIRS`` / ``NC_CURRENTS`` for the current each pair makes.

    ``None`` says no release is configured, or that this pair is a current of
    none, and then no event of this row reaches a release.
    """
    from . import exhad_release
    try:
        root = str(exhad_release.resolve_root())
    except ValueError:
        return None
    minima = _RELEASE_PAIR_W_MIN.get(root)
    if minima is None:
        hnl = importlib.import_module(
            exhad_release.release(root).__name__ + '.hnl')
        currents = [((up, down), name)
                    for name, (up, down) in hnl.CC_PAIRS.items()]
        currents += [((flavour, flavour), name)
                     for flavour, name in hnl.NC_CURRENTS.items()]
        minima = {}
        for flavours, name in currents:
            stated = [table[name]
                      for table in (hnl.PRIMARY_W_MIN, hnl.PYTHIA_W_MIN)
                      if name in table]
            if stated:
                minima[tuple(sorted(abs(int(f)) for f in flavours))] = float(
                    max(stated))
        _RELEASE_PAIR_W_MIN[root] = minima
    first, second = (int(pdg) for pdg in pdg_pair)
    if first * second >= 0:
        return None  # a gluon pair, or not a quark with its antiquark
    return minima.get(tuple(sorted((abs(first), abs(second)))))


def parton_pair_sampling_floor(pdg_pair):
    """Smallest invariant mass the sampler may give a parton pair.

    It is the larger of two numbers: the lightest two-hadron state the pair
    can physically reach, which ``funcs.thresholds`` owns and which gates the
    row's rate, and the lightest state the hadronizer this pair is handed to
    actually builds.  Neither alone is enough -- the first can sit below what
    the hadronizer accepts, and the second is a property of a generator rather
    than of the physics.

    The hadronizer is the configured exHad release, and then the second number
    is the release's own minimum W for the current this pair makes, which
    :func:`release_pair_w_min` reads from the release: EventCalc feeds the
    release and must never hand it a W it refuses.  With no release configured
    the pair goes to EventCalc's own Pythia instead, and the second number is
    ``funcs.thresholds.string_end_two_hadron_mass``, the lightest state string
    fragmentation makes of the pair.  The release's minimum is at or above
    that state for every current it covers.
    """
    if any(abs(int(pdg)) == 6 for pdg in pdg_pair):
        raise ValueError(
            "parton pair %r holds a top quark, which decays before it binds: "
            "no hadronizer builds a state from it, and the shipped decay "
            "tables hold no such row"
            % ([int(pdg) for pdg in pdg_pair],))
    floor = parton_system_threshold(pdg_pair)
    hadronizer_floor = release_pair_w_min(pdg_pair)
    if hadronizer_floor is None:
        hadronizer_floor = string_end_two_hadron_mass(pdg_pair)
    if hadronizer_floor is not None:
        floor = max(floor, hadronizer_floor)
    return floor


@njit
def seed_random(seed):
    """Seed Numba's own RNG inside njit kernels.

    ``np.random.seed`` in the host does not reach the RNG numba compiles into
    ``@njit`` code, so every module with njit sampling exposes this entry point
    and the host seeds all of them from one card seed.
    """
    np.random.seed(seed)

def dalitz_energy_ranges(m, m1, m2, m3, pdg1, pdg2, pdg3):
    """The rectangle of daughter energies the sampler draws candidates from.

    Energy 1 and energy 3 each run from the daughter mass up to the value the
    daughter reaches when the other two recoil together at rest.  When two of
    the three daughters are partons, the pair has to carry at least the floor
    the pair is sampled above, which caps the energy of the remaining
    daughter further: a parton pair of squared invariant mass
    ``m^2 + m_other^2 - 2 m E_other`` must stay above that floor.
    :func:`parton_pair_sampling_floor` states the floor.

    Returns the two ranges as pairs.
    """
    energy1_max = (m ** 2 + m1 ** 2 - (m2 + m3) ** 2) / (2 * m)
    energy3_max = (m ** 2 + m3 ** 2 - (m1 + m2) ** 2) / (2 * m)

    is_parton = [abs(int(pdg)) in PARTON_PDGS for pdg in (pdg1, pdg2, pdg3)]
    if sum(is_parton) >= 2:
        if not is_parton[0]:
            threshold = parton_pair_sampling_floor((pdg2, pdg3))
            energy1_max = min(energy1_max,
                              (m ** 2 + m1 ** 2 - threshold ** 2) / (2 * m))
        elif not is_parton[2]:
            threshold = parton_pair_sampling_floor((pdg1, pdg2))
            energy3_max = min(energy3_max,
                              (m ** 2 + m3 ** 2 - threshold ** 2) / (2 * m))
        else:
            raise ValueError(
                "a three-body channel with the single non-parton in the middle "
                "position is not described by the decay tables")
    return (m1, energy1_max), (m3, energy3_max)


def weights_non_uniform_comp(tabe1e3, MASSM, MASS1, MASS2, MASS3, distr):
    """
    Compute the weights for non-uniformly distributed decay events.

    Parameters:
    -----------
    tabe1e3 : np.ndarray
        Array of energy pairs [E1, E3] for the decay products.
    MASSM : float
        Total mass of the decaying particle.
    MASS1 : float
        Mass of the first decay product.
    MASS2 : float
        Mass of the second decay product.
    MASS3 : float
        Mass of the third decay product.
    distr : function
        Function that computes the matrix element for given energies.

    Returns:
    --------
    np.ndarray
        Array of weights for each event.
    """
    e1 = tabe1e3[:, 0]
    e3 = tabe1e3[:, 1]

    # Calculate the matrix element for each energy pair.  The daughter masses
    # are passed explicitly: the decay tables write the lepton masses
    # symbolically as m1/m2/m3 rather than freezing them as decimals, so a
    # compiled matrix element takes six arguments.
    ME = distr(MASSM, e1, e3, MASS1, MASS2, MASS3)
    return ME

def matrix_element_upper_bound(m, m1, m2, m3, distr, pdg1, pdg2, pdg3):
    """The largest value one channel's squared matrix element can reach.

    The accept-reject step in :func:`block_random_energies` divides the
    sampled squared matrix element by this value, so the value has to lie above
    every value the expression can take anywhere in the allowed region.  It is
    obtained by bounding the expression over boxes of the Dalitz region, not by
    evaluating it at sampled points, so a peak that a trial sample would miss
    is still covered and a sampled weight above the bound is a defect that
    raises rather than a rare draw.

    The result is a cover of the Dalitz region by boxes, each carrying the
    largest value the expression can reach inside it, and ``global_ceiling``
    holds the largest of those.  Working it out costs far more than drawing an
    event and it depends only on the channel and the parent mass, so a bounded
    number of them is kept.
    """
    ranges = dalitz_energy_ranges(m, m1, m2, m3, pdg1, pdg2, pdg3)
    key = (distr, float(m), float(m1), float(m2), float(m3), ranges)
    upper_bound = _UPPER_BOUND_CACHE.get(key)
    if upper_bound is None:
        upper_bound = sampling_bounds.dalitz_upper_bound(
            distr, m, m1, m2, m3, ranges[0], ranges[1])
        if len(_UPPER_BOUND_CACHE) >= _UPPER_BOUND_CACHE_LIMIT:
            _UPPER_BOUND_CACHE.pop(next(iter(_UPPER_BOUND_CACHE)))
        _UPPER_BOUND_CACHE[key] = upper_bound
        SAMPLER_STATS["upper_bounds_built"] += 1
    return upper_bound


def block_random_energies(m, m1, m2, m3, Nevents, distr, pdg1, pdg2, pdg3):
    """
    Draw daughter energies distributed as the channel's matrix element.

    Parameters:
    -----------
    m : float
        Total mass of the decaying particle.
    m1 : float
        Mass of the first decay product.
    m2 : float
        Mass of the second decay product.
    m3 : float
        Mass of the third decay product.
    Nevents : int
        Number of decay events to simulate.
    distr : function
        Function that computes the matrix element for given energies.
    pdg1, pdg2, pdg3 : int
        PDG identifiers for the three decay products.

    Returns:
    --------
    np.ndarray
        Array of energy pairs [E1, E3], one row per event.
    """
    # Exact accept-reject sampling of the Dalitz plane.
    #
    # A candidate rectangle is drawn with probability proportional to its area
    # times the value the matrix element cannot exceed inside it, a point is
    # drawn uniformly inside that rectangle, and the point is kept with
    # probability |M|^2 divided by that same value.  Because that value is a
    # bound worked out from the form of the expression before any event is
    # drawn, rather than the largest value a trial sample happened to show, the
    # keep probability never exceeds one and the accepted points follow |M|^2
    # over the allowed region exactly.  Every returned event is an independent
    # draw and no event is returned twice.
    if Nevents <= 0:
        return np.empty((0, 2), dtype=float)

    # Threshold gate: a row whose daughters do not fit inside the parent has
    # rate exactly zero and is never sampled.  See funcs/thresholds.py.
    from .thresholds import require_open
    require_open(m, (m1, m2, m3))

    upper_bound = matrix_element_upper_bound(
        m, m1, m2, m3, distr, pdg1, pdg2, pdg3)

    def _weights(energy1, energy3, inside):
        values = np.zeros(len(energy1), dtype=float)
        if not np.any(inside):
            return values
        points = np.column_stack((energy1[inside], energy3[inside]))
        sampled = np.asarray(
            weights_non_uniform_comp(points, m, m1, m2, m3, distr))
        if np.iscomplexobj(sampled):
            sampled = sampled.real
        sampled = np.asarray(sampled, dtype=float)
        # A constant matrix element is a legitimate phase-space model.
        # Lambdified constant expressions return one scalar rather than an
        # array, so expand that scalar to one weight per sampled Dalitz point.
        if sampled.ndim == 0:
            sampled = np.full(len(points), float(sampled))
        elif sampled.shape != (len(points),):
            raise ValueError(
                "Matrix-element weights have shape %r; expected one weight per "
                "sampled energy pair." % (sampled.shape,))
        if not np.all(np.isfinite(sampled)):
            raise ValueError(
                "Matrix element returned a non-finite weight; check the decay "
                "table expression for this channel.")
        # |M|^2 is non-negative by construction.  A negative value is a defect
        # of the tabulated expression, not a physical amplitude: clamp it to
        # zero and count it for the caller.
        negative = int(np.count_nonzero(sampled < 0.0))
        if negative:
            SAMPLER_STATS["negative_weight_points"] += negative
            sampled = np.where(sampled < 0.0, 0.0, sampled)
        values[inside] = sampled
        return values

    accepted = np.empty((Nevents, 2), dtype=float)
    filled, drawn = 0, 0
    while filled < Nevents:
        missing = Nevents - filled
        # Aim at ~1.3x the outstanding events given the efficiency.  Before any
        # candidate has been drawn that is the share of them the cover expects
        # to keep, which it knows from the area under it and the integral it
        # covers; a fixed guess instead would draw that many candidates per
        # event whatever the channel, and every one of them costs an evaluation
        # of the matrix element.  A batch that falls short is followed by
        # another sized by what this call has seen.
        efficiency = (max(filled / drawn, 1e-3) if drawn
                      else upper_bound.estimated_acceptance)
        batch = int(min(max(1.3 * missing / efficiency, 1024), 2_000_000))
        energy1, energy3, ceiling = upper_bound.draw_candidates(batch)
        inside = sampling_bounds.dalitz_region_mask(m, m1, m2, m3, energy1, energy3)
        values = _weights(energy1, energy3, inside)
        drawn += batch
        SAMPLER_STATS["candidates"] += batch
        exceeding = values > ceiling
        if np.any(exceeding):
            worst = int(np.argmax(values - ceiling))
            raise RuntimeError(
                "The matrix element of the channel with daughters %d, %d, %d "
                "reached %.17g at E1 = %.17g, E3 = %.17g, above the upper "
                "bound %.17g for that part of the Dalitz plane. The bound is "
                "worked out before sampling and cannot be exceeded, so either "
                "the matrix element is not the one the bound was built from or "
                "the bound in funcs/sampling_bounds.py is wrong."
                % (pdg1, pdg2, pdg3, values[worst], energy1[worst],
                   energy3[worst], ceiling[worst]))
        keep = np.random.random(batch) * ceiling < values
        take = min(int(keep.sum()), missing)
        accepted[filled:filled + take, 0] = energy1[keep][:take]
        accepted[filled:filled + take, 1] = energy3[keep][:take]
        filled += take
        if drawn > 2000 * Nevents + 10_000_000:
            raise RuntimeError(
                "Three-body accept-reject efficiency collapsed for the channel "
                "with daughters %d, %d, %d." % (pdg1, pdg2, pdg3))
    SAMPLER_STATS["accepted"] += Nevents
    return accepted


@njit
def tabPS3bodyCompiled(tabPSenergies, MASSM, MASS1, MASS2, MASS3, pdg1, pdg2, pdg3, charge1, charge2, charge3, stability1, stability2, stability3):
    """
    Compute the momentum components for a three-body decay event, given the energies and particle properties.

    Parameters:
    -----------
    tabPSenergies : np.ndarray
        Array of energies [E1, E3] for the decay products.
    MASSM : float
        Total mass of the decaying particle.
    MASS1, MASS2, MASS3 : float
        Masses of the decay products.
    pdg1, pdg2, pdg3 : int
        PDG codes of the decay products.
    charge1, charge2, charge3 : int
        Charges of the decay products.
    stability1, stability2, stability3 : bool
        Stability flags for the decay products.

    Returns:
    --------
    np.ndarray
        Array containing the momentum components and other properties of the decay products.
    """
    # Extract energies
    eprod1 = tabPSenergies[0]
    eprod3 = tabPSenergies[1]
    eprod2 = MASSM - eprod1 - eprod3

    # Generate random angles for momentum direction
    thetaRand = np.arccos(uniform(-1, 1))
    phiRand = uniform(-np.pi, np.pi)
    kappaRand = uniform(-np.pi, np.pi)

    # Rotate vectors to compute momentum components
    pxprod1 = rotateVectors.p1rotatedX_jit(eprod1, MASS1, thetaRand, phiRand)
    pyprod1 = rotateVectors.p1rotatedY_jit(eprod1, MASS1, thetaRand, phiRand)
    pzprod1 = rotateVectors.p1rotatedZ_jit(eprod1, MASS1, thetaRand, phiRand)

    pxprod2 = rotateVectors.p2rotatedX_jit(eprod1, eprod3, MASSM, MASS1, MASS2, MASS3, thetaRand, phiRand, kappaRand)
    pyprod2 = rotateVectors.p2rotatedY_jit(eprod1, eprod3, MASSM, MASS1, MASS2, MASS3, thetaRand, phiRand, kappaRand)
    pzprod2 = rotateVectors.p2rotatedZ_jit(eprod1, eprod3, MASSM, MASS1, MASS2, MASS3, thetaRand, phiRand, kappaRand)

    pxprod3 = rotateVectors.p3rotatedX_jit(eprod1, eprod3, MASSM, MASS1, MASS2, MASS3, thetaRand, phiRand, kappaRand)
    pyprod3 = rotateVectors.p3rotatedY_jit(eprod1, eprod3, MASSM, MASS1, MASS2, MASS3, thetaRand, phiRand, kappaRand)
    pzprod3 = rotateVectors.p3rotatedZ_jit(eprod1, eprod3, MASSM, MASS1, MASS2, MASS3, thetaRand, phiRand, kappaRand)

    # Return the momentum components and particle properties
    return np.array([
        pxprod1, pyprod1, pzprod1, eprod1, MASS1, pdg1, charge1, stability1,
        pxprod2, pyprod2, pzprod2, eprod2, MASS2, pdg2, charge2, stability2,
        pxprod3, pyprod3, pzprod3, eprod3, MASS3, pdg3, charge3, stability3
    ])

def decay_products(MASSM, Nevents, SpecificDecay):
    """
    Simulate the decay products of a three-body decay event.

    Parameters:
    -----------
    MASSM : float
        Total mass of the decaying particle.
    Nevents : int
        Number of decay events to simulate.
    SpecificDecay : tuple
        Contains properties of the specific decay: PDG codes, masses, charges, stability flags, and matrix element.

    Returns:
    --------
    np.ndarray
        Array containing the simulated decay products for each event.
    """
    pdg1, pdg2, pdg3, MASS1, MASS2, MASS3, charge1, charge2, charge3, stability1, stability2, stability3, Msquared3BodyLLP = SpecificDecay

    # The matrix element is passed on as it stands: the upper bound is cached
    # against it, and wrapping it in a fresh closure on every call would make
    # every call look like a new channel.
    tabE1E3true = block_random_energies(
        MASSM, MASS1, MASS2, MASS3, Nevents, Msquared3BodyLLP, pdg1, pdg2, pdg3)

    # Compute the momentum components and particle properties for each event
    result = np.array([
        tabPS3bodyCompiled(
            e, MASSM, MASS1, MASS2, MASS3, pdg1, pdg2, pdg3,
            charge1, charge2, charge3, stability1, stability2, stability3
        )
        for e in tabE1E3true
    ])
    
    return result
