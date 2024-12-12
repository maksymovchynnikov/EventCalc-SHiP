import numpy as np

def decay_products(mother_mass, SpecificDecay, N_events, N_candidates=100000):
    """
    Simulate four-body decay of a particle into four daughter particles.
    """
    # Masses, charges, and stability from PDG
    pdg1, pdg2, pdg3, pdg4, m1, m2, m3, m4, charge1, charge2, charge3, charge4, stability1, stability2, stability3, stability4 = SpecificDecay
    
    # Helper functions
    def lambda_func(a, b, c):
        return a**2 + b**2 + c**2 - 2*(a*b + a*c + b*c)
    
    def jacobian_4body(m, m1, m2, m3, m4, m234, m34):
        term1 = np.sqrt(lambda_func(m**2, m1**2, m234**2)) / (2 * m)
        term2 = np.sqrt(lambda_func(m234**2, m2**2, m34**2)) / (2 * m234)
        term3 = np.sqrt(lambda_func(m34**2, m3**2, m4**2)) / (2 * m34)
        return term1 * term2 * term3
    
    def generate_random_masses(N_events, N_candidates, m_parent, m1, m2, m3, m4):
        m234_min = m2 + m3 + m4
        m234_max = m_parent - m1
        m34_min = m3 + m4

        m234_candidates = np.random.uniform(m234_min, m234_max, N_candidates)
        m34_max_candidates = m234_candidates - m2

        valid = m34_max_candidates >= m34_min
        m234_candidates = m234_candidates[valid]
        m34_max_candidates = m34_max_candidates[valid]

        m34_candidates = np.random.uniform(m34_min, m34_max_candidates)

        # Include the missing factor in the weights
        weights = (m234_candidates - m2 - m3 - m4) * jacobian_4body(m_parent, m1, m2, m3, m4, m234_candidates, m34_candidates)
        valid_weights = weights > 0
        m234_candidates = m234_candidates[valid_weights]
        m34_candidates = m34_candidates[valid_weights]
        weights = weights[valid_weights]
        probabilities = weights / np.sum(weights)
    
        # Check that N_events does not exceed the number of available candidates
        if N_events > len(m234_candidates):
            raise ValueError("N_events exceeds the number of valid mass combinations. Increase N_candidates or reduce N_events.")
    
        indices = np.random.choice(len(m234_candidates), size=N_events, p=probabilities)

        return m234_candidates[indices], m34_candidates[indices]
    
    def two_body_decay_array(M, m1, m2):
        E1 = (M**2 + m1**2 - m2**2) / (2 * M)
        E2 = (M**2 - m1**2 + m2**2) / (2 * M)
        p = np.sqrt(np.maximum(0, E1**2 - m1**2))
        N = len(M)
        costheta = np.random.uniform(-1, 1, N)
        sintheta = np.sqrt(1 - costheta**2)
        phi = np.random.uniform(0, 2*np.pi, N)
        px = p * sintheta * np.cos(phi)
        py = p * sintheta * np.sin(phi)
        pz = p * costheta
        p1 = np.vstack((E1, px, py, pz)).T
        p2 = np.vstack((E2, -px, -py, -pz)).T
        return p1, p2, costheta
    
    def lorentz_vector_boost(p, beta):
        beta2 = np.sum(beta**2, axis=1)
        gamma = 1.0 / np.sqrt(1 - beta2)
        bp = beta[:,0]*p[:,1] + beta[:,1]*p[:,2] + beta[:,2]*p[:,3]
        gamma2 = (gamma - 1.0) / beta2
        gamma2 = np.nan_to_num(gamma2)
        p0 = gamma * (p[:,0] + bp)
        px = p[:,1] + gamma2 * bp * beta[:,0] + gamma * beta[:,0] * p[:,0]
        py = p[:,2] + gamma2 * bp * beta[:,1] + gamma * beta[:,1] * p[:,0]
        pz = p[:,3] + gamma2 * bp * beta[:,2] + gamma * beta[:,2] * p[:,0]
        return np.vstack((p0, px, py, pz)).T

    m_parent = mother_mass
    m234_selected, m34_selected = generate_random_masses(N_events, 100000, m_parent, m1, m2, m3, m4)
    
    M0 = np.full(N_events, m_parent)
    p1_0, p234_0, costheta_1 = two_body_decay_array(M0, m1, m234_selected)
    
    M234 = m234_selected
    p2_234, p34_234, _ = two_body_decay_array(M234, m2, m34_selected)
    
    M34 = m34_selected
    p3_34, p4_34, _ = two_body_decay_array(M34, m3, m4)
    
    E_34 = p34_234[:,0]
    beta_34 = np.vstack((p34_234[:,1], p34_234[:,2], p34_234[:,3])).T / E_34[:, None]
    p3_234 = lorentz_vector_boost(p3_34, beta_34)
    p4_234 = lorentz_vector_boost(p4_34, beta_34)
    
    E_234 = p234_0[:,0]
    beta_234 = np.vstack((p234_0[:,1], p234_0[:,2], p234_0[:,3])).T / E_234[:, None]
    p2_lab = lorentz_vector_boost(p2_234, beta_234)
    p3_lab = lorentz_vector_boost(p3_234, beta_234)
    p4_lab = lorentz_vector_boost(p4_234, beta_234)
    
    p1_lab = p1_0

    final_data = []
    particles = [p1_lab, p2_lab, p3_lab, p4_lab]
    pdgs = [pdg1, pdg2, pdg3, pdg4]
    masses = [m1, m2, m3, m4]
    charges = [charge1, charge2, charge3, charge4]
    stability = [stability1, stability2, stability3, stability4]

    for i in range(4):
        px, py, pz = particles[i][:, 1], particles[i][:, 2], particles[i][:, 3]
        energy = particles[i][:, 0]
        pdg = np.full(N_events, pdgs[i])  # Expand pdg to length of N_events
        charge = np.full(N_events, charges[i])  # Expand charge to length of N_events
        stab = np.full(N_events, stability[i])  # Expand stability to length of N_events
        mass = np.full(N_events, masses[i])  # Expand mass to length of N_events
        
        # Now all arrays have the same length and can be stacked
        final_data.append(np.vstack((px, py, pz, energy, mass, pdg, charge, stab)).T)

    final_phase_space = np.hstack(final_data)

    # Invariant mass check (sum of all four particles)
    total_energy = p1_lab[:,0] + p2_lab[:,0] + p3_lab[:,0] + p4_lab[:,0]
    total_px = p1_lab[:,1] + p2_lab[:,1] + p3_lab[:,1] + p4_lab[:,1]
    total_py = p1_lab[:,2] + p2_lab[:,2] + p3_lab[:,2] + p4_lab[:,2]
    total_pz = p1_lab[:,3] + p2_lab[:,3] + p3_lab[:,3] + p4_lab[:,3]

    invariant_mass_total = np.sqrt(np.maximum(0, total_energy**2 - (total_px**2 + total_py**2 + total_pz**2)))

    # Check that the invariant mass matches the mother particle mass within floating point precision
    tolerance = 1e-6
    if not np.allclose(invariant_mass_total, mother_mass, rtol=tolerance, atol=tolerance):
        num_failed = np.sum(~np.isclose(invariant_mass_total, mother_mass, rtol=tolerance, atol=tolerance))
        print(f"Warning: {num_failed} events failed the invariant mass check.")

    # return final_phase_space, costheta_1, p1_lab, p2_lab, p3_lab, p4_lab
    return final_phase_space