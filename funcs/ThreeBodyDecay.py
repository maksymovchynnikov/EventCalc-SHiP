import numpy as np
from numba import njit
from numpy.random import uniform, choice
from . import rotateVectors  # Assuming this module is defined elsewhere
import time

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
        current_success_rate = min(max(success_rate, 0.001), 1.0)
        
        # Determine the number of samples to generate
        sample_size = int(1.2 * n_missing / current_success_rate)
        
        # Generate random energy samples for E1 and E3
        E1r = np.random.uniform(m1, (m**2 + m1**2 - (m2 + m3)**2) / (2 * m), sample_size)
        E3r = np.random.uniform(m3, (m**2 + m3**2 - (m1 + m2)**2) / (2 * m), sample_size)
        E2v = m - E1r - E3r

        # Apply kinematic constraints to filter out invalid configurations
        term1 = (E2v**2 - m2**2 - (E1r**2 - m1**2) - (E3r**2 - m3**2))**2
        term2 = 4 * (E1r**2 - m1**2) * (E3r**2 - m3**2)
        is_valid = np.logical_and(E2v > m2, term1 < term2)
        current_n_valid = np.sum(is_valid)

        # Store valid energies and update counters
        n_new_to_add = min(current_n_valid, n_missing)
        E1r_valid[n_valid:n_valid+n_new_to_add] = E1r[is_valid][:n_new_to_add]
        E3r_valid[n_valid:n_valid+n_new_to_add] = E3r[is_valid][:n_new_to_add]
        success_rate = current_n_valid / len(is_valid)
        n_missing -= n_new_to_add
        n_valid += n_new_to_add

    return E1r_valid, E3r_valid

def block_random_energies_hadrons(m, m1, m2, m3, n_events, distr, pdg1, pdg2, pdg3):
    """
    Modified energy sampler that incorporates invariant mass threshold for parton pairs.

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
    distr : function
        Function that computes the matrix element for given energies.
    pdg1, pdg2, pdg3 : int
        PDG identifiers for the three decay products.

    Returns:
    --------
    tuple
        Arrays containing the generated energies for E1 and E3.
    """
    # Define parton PDG codes
    parton_pdgs = {1, 2, 3, 4, 21}
    
    # Identify partons among the decay products
    is_parton1 = abs(pdg1) in parton_pdgs
    is_parton2 = abs(pdg2) in parton_pdgs
    is_parton3 = abs(pdg3) in parton_pdgs
    
    # Count the number of partons
    n_partons = is_parton1 + is_parton2 + is_parton3
    
    if n_partons < 2:
        # Less than two partons: use the original vectorized sampler
        return block_random_energies_vectorized(m, m1, m2, m3, n_events)
    else:
        # Exactly two partons: apply invariant mass threshold
        # Identify the non-parton particle (assumed to be either pdg1 or pdg3)
        if not is_parton1:
            non_parton_index = 1  # Corresponds to E1
            m_non_parton = m1
        elif not is_parton3:
            non_parton_index = 3  # Corresponds to E3
            m_non_parton = m3
        else:
            # According to your note, pdg2 is not a non-parton, but handle gracefully
            raise ValueError("Unexpected scenario: non-parton is pdg2, which is not anticipated.")
        
        # Identify the parton PDGs
        if non_parton_index == 1:
            parton_pdgs_pair = (pdg2, pdg3)
        elif non_parton_index == 3:
            parton_pdgs_pair = (pdg1, pdg2)
        else:
            # This case should not occur as per user note
            parton_pdgs_pair = (pdg1, pdg3)
        
        # Define fictitious meson masses based on PDG codes
        def m_fictitious(pdg):
            abs_pdg = abs(pdg)
            if abs_pdg in {1, 2, 21}:
                return 0.1396
            elif abs_pdg == 3:
                return 0.496
            elif abs_pdg == 4:
                return 1.875
            else:
                # Default value for unexpected PDG codes
                return 0.0
        
        # Compute the invariant mass threshold
        mthr = m_fictitious(parton_pdgs_pair[0]) + m_fictitious(parton_pdgs_pair[1])
        
        # Compute the maximum allowed energy for the non-parton
        E_non_parton_max = (m**2 + m_non_parton**2 - mthr**2) / (2 * m)
        
        # Ensure that the upper bound does not exceed the kinematic limit without threshold
        if non_parton_index == 1:
            E_non_parton_limit = (m**2 + m1**2 - (m2 + m3)**2) / (2 * m)
        elif non_parton_index == 3:
            E_non_parton_limit = (m**2 + m3**2 - (m1 + m2)**2) / (2 * m)
        else:
            # This case should not occur as per user note
            E_non_parton_limit = E_non_parton_max
        
        # The upper bound is the minimum of the two
        E_non_parton_max = min(E_non_parton_max, E_non_parton_limit)
        
        # Initialize arrays to store valid energies
        n_valid = 0
        n_missing = n_events
        E1r_valid = np.zeros(n_events)
        E3r_valid = np.zeros(n_events)
        
        while n_missing > 0:
            # Estimate the number of samples to generate
            sample_size = int(1.2 * n_missing)
            
            if non_parton_index == 1:
                # Sample E1 with the new upper bound
                E1r = np.random.uniform(m1, E_non_parton_max, sample_size)
                E3r = np.random.uniform(m3, (m**2 + m3**2 - (m1 + m2)**2) / (2 * m), sample_size)
            elif non_parton_index == 3:
                # Sample E3 with the new upper bound
                E3r = np.random.uniform(m3, E_non_parton_max, sample_size)
                E1r = np.random.uniform(m1, (m**2 + m1**2 - (m2 + m3)**2) / (2 * m), sample_size)
            else:
                # This case should not occur
                raise ValueError("Unexpected non-parton index.")
            
            E2v = m - E1r - E3r
            
            # Apply kinematic constraints to filter out invalid configurations
            term1 = (E2v**2 - m2**2 - (E1r**2 - m1**2) - (E3r**2 - m3**2))**2
            term2 = 4 * (E1r**2 - m1**2) * (E3r**2 - m3**2)
            is_valid = np.logical_and(E2v > m2, term1 < term2)
            
            # Additional invariant mass threshold condition
            # As per the exact formula: E_non_parton <= (m^2 + m_non_parton^2 - mthr^2)/(2*m)
            # Since we've already sampled E_non_parton <= E_non_parton_max, which incorporates this,
            # there's no need for an additional condition here.
            # However, to be precise, ensure that the parton pair invariant mass >= mthr
            # Using the exact formula provided:
            # M_partons^2 = m^2 + m_non_parton^2 - 2 * m * E_non_parton >= mthr^2
            # Which simplifies to E_non_parton <= (m^2 + m_non_parton^2 - mthr^2)/(2*m)
            # This condition is already enforced in the sampling range
            
            current_n_valid = np.sum(is_valid)
            
            # Store valid energies and update counters
            n_new_to_add = min(current_n_valid, n_missing)
            E1r_valid[n_valid:n_valid+n_new_to_add] = E1r[is_valid][:n_new_to_add]
            E3r_valid[n_valid:n_valid+n_new_to_add] = E3r[is_valid][:n_new_to_add]
            n_missing -= n_new_to_add
            n_valid += n_new_to_add

        return E1r_valid, E3r_valid

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

    # Calculate the matrix element for each energy pair
    ME = distr(MASSM, e1, e3)
    return ME

def block_random_energies(m, m1, m2, m3, Nevents, distr, pdg1, pdg2, pdg3):
    """
    Generate random energies and apply non-uniform weighting to simulate decay events.
    This function incorporates invariant mass threshold for parton pairs when necessary.

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
        Array of weighted energy pairs [E1, E3].
    """
    # Generate energy pairs [E1, E3] with invariant mass constraints if necessary
    tabE1E3unweighted = np.array(
        block_random_energies_hadrons(m, m1, m2, m3, Nevents, distr, pdg1, pdg2, pdg3)
    ).T

    # Calculate weights for the generated energies
    weights1 = np.abs(weights_non_uniform_comp(tabE1E3unweighted, m, m1, m2, m3, distr))

    # Ensure weights are non-negative
    weights1 = np.where(weights1 < 0, 0, weights1)

    # Normalize weights to form a probability distribution
    weight_sum = weights1.sum()
    if weight_sum == 0:
        raise ValueError("All weights are zero. Check the distribution function and energy sampling.")
    probabilities = weights1 / weight_sum

    # Select events according to the computed weights
    tabsel_indices = choice(len(tabE1E3unweighted), size=Nevents, p=probabilities)

    return tabE1E3unweighted[tabsel_indices]

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

    def distr(m, E1, E3):
        return Msquared3BodyLLP(m, E1, E3)

    # Generate energy pairs [E1, E3] for the decay products with invariant mass constraints
    tabE1E3true = block_random_energies(MASSM, MASS1, MASS2, MASS3, Nevents, distr, pdg1, pdg2, pdg3)

    # Compute the momentum components and particle properties for each event
    result = np.array([
        tabPS3bodyCompiled(
            e, MASSM, MASS1, MASS2, MASS3, pdg1, pdg2, pdg3,
            charge1, charge2, charge3, stability1, stability2, stability3
        )
        for e in tabE1E3true
    ])
    
    return result

