# boost.py

import numpy as np

# Constants for table indices
indexpx1 = 0
indexpy1 = 1
indexpz1 = 2
indexE1 = 3
indexm1 = 4
indexpdg1 = 5

def vvec_mother_lab(EmotherLab, pMotherLab1, pMotherLab2, pMotherLab3):
    """
    Calculate the velocity vector of the mother particle in the lab frame.
    """
    return np.array([pMotherLab1, pMotherLab2, pMotherLab3]) / EmotherLab

def gamma_factor(Energy, m):
    """
    Compute the Lorentz factor (gamma) given the energy and mass.
    """
    return Energy / m

def gamma_factor_mother_lab(EmotherLab, mMother):
    """
    Compute the Lorentz factor in the mother particle's lab frame.
    """
    gamma = gamma_factor(EmotherLab, mMother)
    v = np.sqrt(1 - 1 / gamma**2)
    return (gamma - 1) / (v**2)

def pvec_prod_lab(EmotherLab, mMother, pMotherLab1, pMotherLab2, pMotherLab3,
                  EProdRest, pProdRest1, pProdRest2, pProdRest3):
    """
    Transform the momentum vector of a decay product from the rest frame of the mother particle to the lab frame.
    """
    gamma = gamma_factor(EmotherLab, mMother)
    vvec = vvec_mother_lab(EmotherLab, pMotherLab1, pMotherLab2, pMotherLab3)
    gamma_factor_lab = gamma_factor_mother_lab(EmotherLab, mMother)

    pVecProdRest = np.array([pProdRest1, pProdRest2, pProdRest3])
    vdotp = np.dot(vvec, pVecProdRest)

    return pVecProdRest + gamma * vvec * EProdRest + gamma_factor_lab * vvec * vdotp

def E_prod_lab(EmotherLab, mMother, pMotherLab1, pMotherLab2, pMotherLab3,
               EProdRest, pProdRest1, pProdRest2, pProdRest3):
    """
    Compute the energy of a decay product in the lab frame.
    """
    gamma = gamma_factor(EmotherLab, mMother)
    vvec = vvec_mother_lab(EmotherLab, pMotherLab1, pMotherLab2, pMotherLab3)
    pVecProdRest = np.array([pProdRest1, pProdRest2, pProdRest3])
    return gamma * (EProdRest + np.dot(vvec, pVecProdRest))

def tab_boosted_decay_products(m, momentum, tabledaughters_array):
    """
    Compute the boosted decay products for multiple events.
    """
    num_events, num_columns = tabledaughters_array.shape
    num_particles = num_columns // 6  # Each particle has 6 attributes

    # Initialize list to collect boosted products
    boosted_products = []

    # Debugging: Print shape of input arrays
    #print(f"Number of decay events (tabledaughters_array): {num_events}")
    #print(f"Number of particles per event (including padding): {num_particles}")
    #print(f"Shape of tabledaughters_array: {tabledaughters_array.shape}")
    #print(f"Shape of momentum array: {momentum.shape}")

    # Convert momentum to NumPy array if it's not already
    if not isinstance(momentum, np.ndarray):
        momentum = np.array(momentum)
    
    # Debugging: Print number of momentum entries
    #print(f"Number of momentum entries: {momentum.shape[0]}")

    # Validate momentum array dimensions
    if momentum.ndim != 2 or momentum.shape[1] != 4:
        raise ValueError("Momentum should be a 2D array with shape (num_events, 4)")

    if momentum.shape[0] != num_events:
        print(f"Mismatch detected: {momentum.shape[0]} momentum entries vs {num_events} decay events.")
        raise ValueError("The number of momentum entries does not match the number of decay events.")

    # Extract mother energy and momentum components
    mother_E = momentum[:, 3]
    mother_px = momentum[:, 0]
    mother_py = momentum[:, 1]
    mother_pz = momentum[:, 2]

    # Calculate velocity vectors
    with np.errstate(divide='ignore', invalid='ignore'):
        vvec_x = mother_px / mother_E
        vvec_y = mother_py / mother_E
        vvec_z = mother_pz / mother_E

    # Calculate gamma and gamma_factor_lab
    gamma = mother_E / m
    v_squared = vvec_x**2 + vvec_y**2 + vvec_z**2
    # To prevent division by zero
    v_squared = np.where(v_squared == 0, 1e-12, v_squared)
    gamma_factor_lab = (gamma - 1) / v_squared

    # Define the pad value for boosted products
    padding = [0.0, 0.0, 0.0, 0.0, 0.0, -999]

    for i in range(num_events):
        boosted_event = []
        for j in range(num_particles):
            idx = j * 6
            pdgId = tabledaughters_array[i, idx + indexpdg1]
            if pdgId == -999:
                continue  # Skip placeholder particles

            EProdRest = tabledaughters_array[i, idx + indexE1]
            pProdRest1 = tabledaughters_array[i, idx + indexpx1]
            pProdRest2 = tabledaughters_array[i, idx + indexpy1]
            pProdRest3 = tabledaughters_array[i, idx + indexpz1]

            # Compute boosted energy and momentum
            E_lab = gamma[i] * (EProdRest + vvec_x[i] * pProdRest1 +
                                vvec_y[i] * pProdRest2 + vvec_z[i] * pProdRest3)
            p_lab = np.array([
                pProdRest1 + gamma[i] * vvec_x[i] * EProdRest + gamma_factor_lab[i] * vvec_x[i] * (vvec_x[i] * pProdRest1 + vvec_y[i] * pProdRest2 + vvec_z[i] * pProdRest3),
                pProdRest2 + gamma[i] * vvec_y[i] * EProdRest + gamma_factor_lab[i] * vvec_y[i] * (vvec_x[i] * pProdRest1 + vvec_y[i] * pProdRest2 + vvec_z[i] * pProdRest3),
                pProdRest3 + gamma[i] * vvec_z[i] * EProdRest + gamma_factor_lab[i] * vvec_z[i] * (vvec_x[i] * pProdRest1 + vvec_y[i] * pProdRest2 + vvec_z[i] * pProdRest3)
            ])

            boosted_daughter = [
                p_lab[0],  # px
                p_lab[1],  # py
                p_lab[2],  # pz
                E_lab,     # E
                tabledaughters_array[i, idx + indexm1],  # mass
                pdgId
            ]
            boosted_event.extend(boosted_daughter)

        # Pad boosted_event to have the same number of particles as max_n
        current_num_boosted = len(boosted_event) // 6
        m = num_particles - current_num_boosted
        if m > 0:
            event_padded = boosted_event + padding * m
            #print(f"Padded event {i+1} from {current_num_boosted} to {num_particles} particles.")
        else:
            event_padded = boosted_event
            #print(f"No padding needed for event {i+1} with {current_num_boosted} particles.")
        boosted_products.append(event_padded)
        #boosted_products.append(boosted_event)

        # Debugging: Print progress every 10 events
        #if (i + 1) % 10 == 0 or i + 1 == num_events:
            #print(f"Boosted {i + 1} / {num_events} events")

    # Convert to NumPy array
    boosted_products_array = np.array(boosted_products, dtype=np.float64)

    return boosted_products_array

