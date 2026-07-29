import numpy as np

# Masses in GeV
# Source: PDG Review of Particle Physics 2026, B+ mass.
M_B_PLUS = 5.27941
M_K_PLUS = 0.493677
M_PI_PLUS = 0.13957039

# Should these be included?
M_PI_0 = 0.1349768
M_K_0 = 0.497611
M_KSTAR_PLUS_892 = 0.89188
M_K2STAR_PLUS_1430 = 1.4273


M_KSTAR_892 = 0.89556
M_KSTAR_1410 = 1.414
M_KSTAR_1680 = 1.718
M_K0STAR_700 = 0.838
M_K0STAR_1430 = 1.425
M_K1_1270 = 1.256
M_K1_1400 = 1.403
M_K2STAR_1430 = 1.4324

BPLUS_TO_XA_CHANNELS = [
    {"name": "K+", "mass": M_K_PLUS, "pdg": 321, "scalar_csv_column": "K"},
    {"name": "K*(892)+", "mass": M_KSTAR_892, "pdg": 323, "scalar_csv_column": "K_star892"},
    {"name": "K*(1410)+", "mass": M_KSTAR_1410, "pdg": 100323, "scalar_csv_column": "K_star1410"},
    {"name": "K*(1680)+", "mass": M_KSTAR_1680, "pdg": 30323, "scalar_csv_column": "K_star1680"},
    {"name": "K0*(700)+", "mass": M_K0STAR_700, "pdg": None, "scalar_csv_column": "K_0star800"},
    {"name": "K0*(1430)+", "mass": M_K0STAR_1430, "pdg": 10321, "scalar_csv_column": "K_0star1430"},
    {"name": "K1(1270)+", "mass": M_K1_1270, "pdg": 10323, "scalar_csv_column": "K_11270"},
    {"name": "K1(1400)+", "mass": M_K1_1400, "pdg": 20323, "scalar_csv_column": "K_11400"},
    {"name": "K2*(1430)+", "mass": M_K2STAR_1430, "pdg": 325, "scalar_csv_column": "K_21430"},
    {"name": "pi+", "mass": M_PI_PLUS, "pdg": 211, "scalar_csv_column": "Pi"},
]


# SHiP decay-volume angular coverage, from ship_setup.py values.
from funcs.ship_setup import theta_max_dec_vol  # 0.04495960111270482, same as EventCalc

THETA_MAX_SHIP = float(theta_max_dec_vol)
THETA_MARGIN = 0.005  # Suggested margin from Maksym
THETA_MAX_TABLE = THETA_MAX_SHIP + THETA_MARGIN


# Natural-unit conversion
# Source: PDG Review of Particle Physics, physical constants.
HBAR_GEV_S = 6.582119569e-25

# B+ lifetime
# Source: HFLAV/PDG B+ lifetime average.
TAU_B_PLUS_S = 1.637e-12
TAU_B_PLUS_GEV_INV = TAU_B_PLUS_S / HBAR_GEV_S


# N_BB_PER_POT = sigma_bb / sigma_pp for SHiP, source: 1904.10447v4, Table 2. This includes cascade?
N_BB_PER_POT = 2.7e-7

# Source: 1902.06240v2, Table 3.
F_BPLUS = 0.417
F_BZERO = 0.418


# Lifetime 
ALPHA_EM = 1.0 / 137.035999177  # NIST/CODATA
SIN2_THETA_W = 0.23122  # For THETA_W(M_Z)
M_W = 80.3625  # PDG

LEPTON_MASSES = {
    "e": 0.00051099895000,
    "mu": 0.1056583755,
    "tau": 1.77686,
}

# hbar*c in GeV m. For Gamma in GeV, c*tau in metres is HBARC_GEV_M/Gamma.
HBARC_GEV_M = 1.973269804e-16  # NIST

ALPHA_SU2 = ALPHA_EM / SIN2_THETA_W
SU2_OPERATOR_FACTOR = ALPHA_SU2 / (4.0 * np.pi)
PHOTON_OPERATOR_FACTOR = ALPHA_EM / (4.0 * np.pi)


# BRANCHING
# Source:
#   URL: https://pdg.lbl.gov/
G_F = 1.1663788e-5
G2_EW_SQUARED = 4.0 * np.sqrt(2.0) * G_F * M_W**2

# Quark masses for x_q = m_q^2 / m_W^2
# Source:
#   URL: https://pdg.lbl.gov/
#   MAYBE BETTER SOURCES??
M_U = 0.00216
M_C = 1.65
M_T = 172.57


# CKM matrix
#   URL: https://pdg.lbl.gov/
# Wolfenstein quantities
LAMBDA_CKM = 0.22500
A_CKM = 0.826
RHOBAR = 0.159
ETABAR = 0.348
