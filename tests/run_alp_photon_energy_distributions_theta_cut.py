#!/usr/bin/env python3
"""
Event-level ALP-photon plot with EventCalc theta_ALP preselection.

This uses the same setup and comparison curves as
run_alp_photon_energy_distributions.py, but assigns zero weight to EventCalc
events with theta_ALP >= 0.029 rad before histogramming.
"""

from run_alp_photon_energy_distributions import DEFAULT_RESAMPLE_SIZE, DEFAULT_SEED, run


THETA_ALP_MAX_RAD = 0.029


if __name__ == "__main__":
    run(
        resample_size=DEFAULT_RESAMPLE_SIZE,
        seed=DEFAULT_SEED,
        eventcalc_theta_max=THETA_ALP_MAX_RAD,
        output_suffix="_theta_lt_29mrad",
    )
