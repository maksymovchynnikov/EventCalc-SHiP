import numpy as np
from numba import jit, njit, prange
from numpy.random import random, uniform, choice
from . import rotateVectors
import time

@njit
def E2valEnergies(m, E1, E3):
    """
    Calculate the energy of the second particle in a three-body decay,
    given the total mass and the energies of the first and third particles.

    Parameters:
    -----------
    m : float
        Total mass of the decaying particle.
    E1 : float
        Energy of the first particle.
    E3 : float
        Energy of the third particle.

    Returns:
    --------
    float
        Energy of the second particle.
    """
    return m - E1 - E3

@njit
def block_random_energies_old(m, m1, m2, m3):
    """
    Generate random energies for the first and third decay products in a three-body decay.
    This function uses a while loop to ensure energy conservation and kinematic constraints.

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

    Returns:
    --------
    np.ndarray
        Array containing the generated energies [E1, E3].
    """
    while True:
        # Randomly sample energies for the third and first decay products
        E3r = np.random.uniform(m3, (m**2 + m3**2 - (m1 + m2)**2) / (2 * m))
        E1r = np.random.uniform(m1, (m**2 + m1**2 - (m2 + m3)**2) / (2 * m))
        E2v = m - E1r - E3r

        # Apply energy and kinematic constraints to ensure a valid decay
        if E2v > m2 and (E2v**2 - m2**2 - (E1r**2 - m1**2) - (E3r**2 - m3**2))**2 < 4 * (E1r**2 - m1**2) * (E3r**2 - m3**2):
            break
    return np.array([E1r, E3r])

def block_random_energies_vectorized(m, m1, m2, m3, n_events, success_rate=1.0):
    """
    Vectorized version of block_random_energies_old. This generates random energies for multiple events simultaneously.

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
    n_events : int
        Number of decay events to simulate.
    success_rate : float, optional
        Estimate of the success rate for generating valid decay configurations. Default is 1.0.

    Returns:
    --------
    tuple
        Arrays containing the generated energies for E1 and E3.
    """
    n_valid = 0
    n_missing = n_events
    E1r_valid = np.zeros(n_events)
    E3r_valid = np.zeros(n_events)

    while n_missing > 0:
        # Clip success_rate to avoid RAM issues
        success_rate = min(max(success_rate, 0.001), 1.0)
        
        # Generate random energy samples for E1 and E3
        E3r = np.random.uniform(m3, (m ** 2 + m3 ** 2 - (m1 + m2) ** 2) / (2 * m), int(1.2 * n_missing / success_rate))
        E1r = np.random.uniform(m1, (m ** 2 + m1 ** 2 - (m2 + m3) ** 2) / (2 * m), int(1.2 * n_missing / success_rate))
        E2v = m - E1r - E3r

        # Apply kinematic constraints to filter out invalid configurations
        term1 = (E2v ** 2 - m2 ** 2 - (E1r ** 2 - m1 ** 2) - (E3r ** 2 - m3 ** 2)) ** 2
        term2 = 4 * (E1r ** 2 - m1 ** 2) * (E3r ** 2 - m3 ** 2)
        is_valid = np.logical_and(E2v > m2, term1 < term2)
        current_n_valid = np.sum(is_valid)

        # Store valid energies and update counters
        n_new_to_add = min(current_n_valid, n_missing)
        E1r_valid[n_valid:n_valid+n_new_to_add] = E1r[is_valid][:n_new_to_add]
        E3r_valid[n_valid:n_valid+n_new_to_add] = E3r[is_valid][:n_new_to_add]
        success_rate = current_n_valid / len(is_valid)
        n_missing -= current_n_valid
        n_valid += current_n_valid

    return E1r_valid, E3r_valid

@njit
def block_random_energies_old1(m, m1, m2, m3, Nevents):
    """
    Generate random energies for multiple decay events using the old method.
    This version uses a loop to generate energies for each event.

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

    Returns:
    --------
    np.ndarray
        Array of generated energies for each event.
    """
    result = np.empty((Nevents, 2))  # Preallocate array with the right shape
    for i in prange(Nevents):
        result[i] = block_random_energies_old(m, m1, m2, m3)
    return result

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
    e1 = np.array(tabe1e3[:, 0])
    e3 = np.array(tabe1e3[:, 1])

    # Calculate the matrix element for each energy pair
    ME = distr(MASSM, e1, e3)
    return ME

def block_random_energies(m, m1, m2, m3, Nevents, distr):
    """
    Generate random energies and apply non-uniform weighting to simulate decay events.

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

    Returns:
    --------
    np.ndarray
        Array of weighted energy pairs [E1, E3].
    """
    # Generate random energies assuming a unit matrix element
    tabE1E3unweighted = np.array(block_random_energies_vectorized(m, m1, m2, m3, Nevents)).T

    # Calculate weights for the generated energies
    t = time.time()
    weights1 = np.abs(weights_non_uniform_comp(tabE1E3unweighted, m, m1, m2, m3, distr))

    # Ensure weights are non-negative
    weights1 = np.where(weights1 < 0, 0, weights1)

    # Select events according to the computed weights
    tabsel_indeces = choice(len(tabE1E3unweighted), size=Nevents, p=weights1/weights1.sum())
    
    return tabE1E3unweighted[tabsel_indeces]

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
    return np.array([pxprod1, pyprod1, pzprod1, eprod1, MASS1, pdg1, charge1, stability1,
                     pxprod2, pyprod2, pzprod2, eprod2, MASS2, pdg2, charge2, stability2,
                     pxprod3, pyprod3, pzprod3, eprod3, MASS3, pdg3, charge3, stability3])

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

    def distr(m, E1, E3):
        return Msquared3BodyLLP(m, E1, E3)

    # Generate energy pairs [E1, E3] for the decay products
    tabE1E3true = block_random_energies(MASSM, MASS1, MASS2, MASS3, Nevents, distr)

    # Compute the momentum components and particle properties for each event
    result = np.array([tabPS3bodyCompiled(e, MASSM, MASS1, MASS2, MASS3, pdg1, pdg2, pdg3, charge1, charge2, charge3, stability1, stability2, stability3)
                       for e in tabE1E3true])
    
    return result
