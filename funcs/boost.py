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
    """Boost and pack all real daughters, preserving scalar arithmetic order."""
    num_events, num_columns = tabledaughters_array.shape
    num_particles = num_columns // 6
    momentum = np.asarray(momentum)
    if momentum.ndim != 2 or momentum.shape[1] != 4:
        raise ValueError("Momentum should be a 2D array with shape (num_events, 4)")
    if momentum.shape[0] != num_events:
        raise ValueError("The number of momentum entries does not match the number of decay events.")
    with np.errstate(divide='ignore', invalid='ignore'):
        vvec_x = momentum[:, 0] / momentum[:, 3]
        vvec_y = momentum[:, 1] / momentum[:, 3]
        vvec_z = momentum[:, 2] / momentum[:, 3]
    gamma = momentum[:, 3] / m
    v_squared = vvec_x**2 + vvec_y**2 + vvec_z**2
    v_squared = np.where(v_squared == 0, 1e-12, v_squared)
    gamma_factor_lab = (gamma - 1) / v_squared

    # Keep the original operation order, including E + vx*px + vy*py + vz*pz.
    # Internal padding is compacted without changing particle order.
    daughters = tabledaughters_array.reshape(num_events, num_particles, 6)
    present = daughters[:, :, indexpdg1] != -999
    rows, slots = np.nonzero(present)
    destination = np.cumsum(present, axis=1)[rows, slots] - 1
    values = daughters[rows, slots]
    px, py, pz, energy = (values[:, k] for k in range(4))
    vx, vy, vz = vvec_x[rows], vvec_y[rows], vvec_z[rows]
    g, gf = gamma[rows], gamma_factor_lab[rows]
    dot = vx * px + vy * py + vz * pz
    boosted = values.copy()
    boosted[:, 0] = px + g * vx * energy + gf * vx * dot
    boosted[:, 1] = py + g * vy * energy + gf * vy * dot
    boosted[:, 2] = pz + g * vz * energy + gf * vz * dot
    boosted[:, 3] = g * (energy + vx * px + vy * py + vz * pz)
    output = np.zeros((num_events, num_particles, 6), dtype=np.float64)
    output[:, :, indexpdg1] = -999
    output[rows, destination] = boosted
    return output.reshape(num_events, num_columns)
