import numpy as np
import numba as nb
from scipy.interpolate import RegularGridInterpolator


@nb.njit
def seed_random(seed):
    """Seed Numba's own RNG inside njit kernels.

    ``np.random.seed`` in the host does not reach the RNG numba compiles into
    ``@njit`` code, so every module with njit sampling exposes this entry point
    and the host seeds all of them from one card seed.
    """
    np.random.seed(seed)

"""
Indices referring to the meaning of the final table columns.
"""
indexpx1 = 0
indexpy1 = 1
indexpz1 = 2
indexE1 = 3
indexm1 = 4
indexpdg1 = 5
indexcharge1 = 6
indexstability1 = 7
LengthDataProduct = indexstability1 + 1

@nb.njit
def pvec_prod_rest(pProdRest1, pProdRest2, pProdRest3):
    """
    Combine the three momentum components of decay products into a single vector.
    """
    return np.array([pProdRest1, pProdRest2, pProdRest3])

@nb.njit
def n_vector_particles(thetaVals, phiVals, E1, E2, m1, m2, pdg1, pdg2, charge1, charge2, stability1, stability2):
    """
    Calculate the momentum components of two decay products based on random angles (theta, phi).
    """
    # Magnitude of momentum from the energy-mass relation.
    #
    # Exactly at threshold the true value of E1**2 - m1**2 is zero and the
    # double-precision evaluation may land a few ulp below it, so a clamp is
    # needed -- but ONLY for that.  A closed channel (daughters heavier than
    # the parent) misses by a relative amount of order 1e-2, which is ten
    # orders of magnitude outside this tolerance; it raises rather than
    # returning daughters at rest with an energy below their own rest mass.
    # The tolerance and its justification live in funcs/thresholds.py
    # (MOMENTUM_ROUNDOFF_REL); closed rows are zeroed before sampling by
    # funcs.thresholds.gate_rates.
    psq = E1**2 - m1**2
    if psq < -1e-12 * max(E1**2, 1.0):
        raise ValueError('Two-body channel is closed at this mass: the '
                         'daughters are heavier than the parent. Gate the row '
                         'with funcs.thresholds.gate_rates before sampling.')
    pmod = np.sqrt(max(psq, 0.0))

    # Momentum components for the first particle
    px1 = pmod * np.sin(thetaVals) * np.cos(phiVals)
    py1 = pmod * np.sin(thetaVals) * np.sin(phiVals)
    pz1 = pmod * np.cos(thetaVals)

    # Momentum components for the second particle (opposite direction due to momentum conservation)
    px2 = -px1
    py2 = -py1
    pz2 = -pz1

    # Return the properties of both decay products
    return np.array([
        [px1, py1, pz1, E1, m1, pdg1, charge1, stability1], 
        [px2, py2, pz2, E2, m2, pdg2, charge2, stability2]
    ])

@nb.njit
def simulate_decays(m, m1, m2, pdg1, pdg2, charge1, charge2, stability1, stability2):
    """
    Simulate a two-body decay of a particle with mass `m` into two products with masses `m1` and `m2`.
    Random angles are used to determine the momentum directions.
    """
    # Random polar and azimuthal angles
    thetaVals = np.arccos(np.random.uniform(-1, 1))
    phiVals = np.random.rand() * 2 * np.pi

    # Calculate the energies of the decay products in the rest frame
    E1 = (m**2 + m1**2 - m2**2) / (2 * m)
    E2 = (m**2 + m2**2 - m1**2) / (2 * m)

    # Generate the decay products with their respective kinematic properties
    daughters = n_vector_particles(thetaVals, phiVals, E1, E2, m1, m2, pdg1, pdg2, charge1, charge2, stability1, stability2)
    return daughters

def LLP_BrRatios(m, LLP_BrRatios):
    """
    Interpolate branching ratios for a given particle mass `m` from a pre-calculated table `LLP_BrRatios`.
    """
    mass_axis = LLP_BrRatios[0]
    channels = LLP_BrRatios.columns[1:]

    # Create interpolators for each branching ratio channel
    interpolators = np.asarray([RegularGridInterpolator((mass_axis,), LLP_BrRatios[channel].values) for channel in channels])

    # Return interpolated branching ratios at the given mass `m`
    return np.array([interpolator([m])[0] for interpolator in interpolators])

@nb.njit
def weighted_choice(values, weights):
    """
    Select a value from `values` based on the provided `weights`.
    """
    total_weight = np.sum(weights)
    cumulative_weights = np.cumsum(weights) / total_weight  # Normalize cumulative weights to [0, 1]

    # Generate a random number and find the corresponding value based on cumulative weights
    rand = np.random.rand()
    index = 0
    while index < len(cumulative_weights) and rand >= cumulative_weights[index]:
        index += 1

    return values[index]

@nb.njit('(float64, int64, float64, float64, int64, int64, int64, int64, int64, int64,)')
def decay_products(m, size, m1, m2, pdg1, pdg2, charge1, charge2, stability1, stability2):
    """
    Simulate multiple decay events for a particle of mass `m` into two products.
    The size parameter defines the number of decay events to generate.
    """
    # Threshold gate.  A row whose daughters do not fit inside the parent has
    # rate exactly zero and must never be sampled; if one arrives here the
    # caller skipped funcs.thresholds.gate_rates.  Strict: at threshold the
    # daughters would be produced at rest, which carries no phase space.
    if m <= m1 + m2:
        raise ValueError('Two-body channel is closed at this mass (m <= m1 + m2). '
                         'Threshold-aware rates must zero this row before sampling; '
                         'no particles are produced.')

    products = np.empty((size, 16), dtype=np.float64)  # Preallocate array for all decay products

    for i in nb.prange(size):  # Parallel loop over all decay events
        # Simulate the decay and extract the kinematic properties for the two decay products
        r = simulate_decays(m, m1, m2, pdg1, pdg2, charge1, charge2, stability1, stability2)
        products[i] = r.flatten()  # Flatten and store the results
    return products
