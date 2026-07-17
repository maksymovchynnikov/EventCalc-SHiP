from pathlib import Path

import numpy as np

ALP_DIR = Path(__file__).resolve().parent
EVENTCALC_DIR = ALP_DIR.parents[1]

MASSES_GEV = np.array([
    0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.0625, 0.075, 0.0875, 0.10,
    0.125, 0.15, 0.175, 0.20,0.225, 0.25, 0.275, 0.30, 0.35, 0.40,
    0.50, 0.625, 0.75, 0.875, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 
    2.50, 2.75, 3.00, 3.25, 3.50, 3.56, 3.75, 3.84, 3.87, 4.00, 
    4.03, 4.25, 4.38, 4.44, 4.50, 4.75, 4.78, 4.90, 5.00, 5.10,
], dtype=float)
assert len(MASSES_GEV) == 50
assert np.all(np.diff(MASSES_GEV) > 0)


COUPLING_NORMALIZATION_GEV_INV = 1.0
F_A_MATCHING_GEV = 1.0e3

EMAX_RELATIVE_THRESHOLD = 1e-7
EMAX_THETA_PADDING_BINS = 2

THETA_MARGIN = 0.005 # Suggested margin from Maksym

B0_TO_BPLUS_BR_FACTOR = 0.93 # From article, 1904.10447v4


# Grid used for the angle-energy distribution.
N_THETA_FORWARD = 100

# Only used for the full-angle normalization test.
# These bins disappear when the final table is truncated.
N_THETA_TAIL = 250

# EventCalc convention for effectively empty bins.
DISTRIBUTION_FLOOR = 1e-90




DEBUG_MASSES_GEV = np.array([
    0.05, 4.5,
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
    OUTPUT_ROOT = ALP_DIR / "debug_tests"
    OUTPUT_ROOT_PLOTS = OUTPUT_ROOT / "plots"

