from pathlib import Path

import numpy as np

ALP_DIR = Path(__file__).resolve().parent
EVENTCALC_DIR = ALP_DIR.parents[1]

MASSES_GEV = np.array([
    0.01, 0.02, 0.05, 0.075,
    0.10, 0.15, 0.20, 0.25,
    0.30, 0.40, 0.50, 0.75,
    1.00, 1.25, 1.50, 1.75,
    2.00, 2.50, 3.00, 3.50,
    4.00, 4.50, 4.75,
    5.00, 5.10,
])


COUPLING_NORMALIZATION_GEV_INV = 1.0
F_A_MATCHING_GEV = 1.0e3

EMAX_RELATIVE_THRESHOLD = 1e-7
EMAX_THETA_PADDING_BINS = 2

THETA_MARGIN = 0.005 # Suggested margin from Maksym

B0_TO_BPLUS_BR_FACTOR = 0.93 # From article, 1904.10447v4
# TODO: The global B0/B+ factor 0.93 is appropriate for the
# corresponding strange modes. Check separately whether
# B0 -> pi0 a requires an additional isospin factor 1/2
# relative to B+ -> pi+ a.




# Grid used for the angle-energy distribution.
N_THETA_FORWARD = 100

# Only used for the full-angle normalization test.
# These bins disappear when the final table is truncated.
N_THETA_TAIL = 250

# EventCalc convention for effectively empty bins.
DISTRIBUTION_FLOOR = 1e-90




DEBUG_MASSES_GEV = np.array([
    0.01, 0.02, 0.05, 0.075,
    0.10, 0.20, 0.50, 1.00,
])

# General run configuration.
RUN_MODE = "final"  # "debug" or "final"
seed = 1
rng = np.random.default_rng(seed)


B_MOMENTA_PATH = ALP_DIR / "exp1.txt"
SCALAR_TABLE_PATH = ALP_DIR / "Br-ratios-scalar.csv"

if RUN_MODE == "final":
    OUTPUT_ROOT = EVENTCALC_DIR / "Distributions" / "ALP-SU2L"
    OUTPUT_ROOT_PLOTS = ALP_DIR / "Final_plots"
else:
    OUTPUT_ROOT = ALP_DIR / "Tests"
    OUTPUT_ROOT_PLOTS = OUTPUT_ROOT / "plots"

