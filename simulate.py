# simulate.py
import sys
import os
import time
import numpy as np
import math
import pandas as pd 

from funcs.ship_setup import (
    z_min, z_max, Delta_x_in, Delta_x_out, Delta_y_in, Delta_y_out, theta_max_dec_vol
)

print("\nSHiP setup (modify ship_setup.py if needed):\n")
print(f"z_min = {z_min} m, z_max = {z_max} m, "
      f"Delta_x_in = {Delta_x_in} m, Delta_x_out = {Delta_x_out} m, "
      f"Delta_y_in = {Delta_y_in} m, Delta_y_out = {Delta_y_out} m, "
      f"theta_max = {theta_max_dec_vol:.6f} rad\n")

from funcs import initLLP, decayProducts, boost, kinematics, mergeResults
from funcs.LLP_selection import (
    select_particle, prompt_uncertainty, prompt_mixing_pattern, 
    prompt_masses_and_c_taus, prompt_decay_channels, 
    resampleSize, particle_selection, mixing_pattern, 
    uncertainty, nEvents, N_pot
)
from funcs.plot_phenomenology import (
    plot_production_probability, plot_lifetime, plot_branching_ratios
)

# Initialize LLP
LLP = initLLP.LLP(
    mass=None, 
    particle_selection=particle_selection, 
    mixing_pattern=mixing_pattern, 
    uncertainty=uncertainty
)
selected_decay_indices = prompt_decay_channels(LLP.decayChannels)

print("\nGenerating LLP phenomenology plots...")

masses_plot = np.logspace(
    np.log10(LLP.m_min_tabulated), 
    np.log10(LLP.m_max_tabulated), 
    250
)
Yield_plot = np.array([LLP.get_total_yield(m) for m in masses_plot])
ctau_int_plot = np.array([LLP.get_ctau(m) for m in masses_plot])
Br_arr = [LLP.get_Br(m) for m in masses_plot]
Br_plot = np.vstack(Br_arr)
chosen_channels = [LLP.decayChannels[i] for i in selected_decay_indices]

plot_folder = f"plots/{LLP.LLP_name}/phenomenology"
if not os.path.exists(plot_folder):
    os.makedirs(plot_folder)

plot_production_probability(masses_plot, Yield_plot, LLP, plot_folder)
plot_lifetime(masses_plot, ctau_int_plot, LLP, plot_folder)
plot_branching_ratios(
    masses_plot, Br_plot, chosen_channels, selected_decay_indices, LLP, plot_folder
)

print("Phenomenology plots generated.")

masses, c_taus_list = prompt_masses_and_c_taus()
timing = False

ifExportEvents = True
min_events_threshold = 2  # Unified threshold

total_masses = len(masses)
print(f"\nTotal masses to process: {total_masses}")

for mass_idx, (mass, c_taus) in enumerate(zip(masses, c_taus_list), start=1):
    if not (LLP.m_min_tabulated < mass < LLP.m_max_tabulated):
        print(f"The current mass {mass} is outside the tabulated data range "
              f"({LLP.m_min_tabulated}, {LLP.m_max_tabulated}). Skipping...")
        continue

    print(f"\nProcessing mass {mass} GeV")
    LLP.set_mass(mass)
    LLP.compute_mass_dependent_properties()

    br_visible_val = sum(LLP.BrRatios_distr[idx] for idx in selected_decay_indices)

    if br_visible_val == 0:
        print("No decay events for these decay modes at this mass. Skipping...")
        continue

    if isinstance(c_taus, (list, tuple, np.ndarray)):
        c_tau_values = c_taus
    else:
        c_tau_values = [c_taus]

    total_c_taus = len(c_tau_values)
    print(f"  Total lifetimes (c_tau) to process for mass {mass} GeV: {total_c_taus}")

    for c_tau_idx, c_tau in enumerate(c_tau_values, start=1):
        print(f"  Processing c_tau {c_tau} m")

        LLP.set_c_tau(c_tau)

        coupling_squared = LLP.c_tau_int / c_tau
        if LLP.LLP_name != "Scalar-quartic":
            yield_times_coupling = LLP.Yield * coupling_squared
        else:
            yield_times_coupling = LLP.Yield * 0.01
        N_LLP_tot = N_pot * yield_times_coupling 

        if yield_times_coupling < 1e-21:
            print("    The overall yield of produced LLPs is effectively zero for this mass. Skipping...")
            continue

        val = math.exp(-z_min / (c_tau * 400 / LLP.mass))
        if val < 1e-21:
            print(f"For the given mass {LLP.mass} GeV and proper lifetime {c_tau} m, "
                  f"all LLPs decay before entering the decay volume. Skipping")
            continue

        t = time.time()

        kinematics_samples = kinematics.Grids(
            LLP.Distr, LLP.Energy_distr, nEvents, LLP.mass, LLP.c_tau_input
        )

        kinematics_samples.interpolate(timing)
        kinematics_samples.resample(resampleSize, timing)
        epsilon_polar = kinematics_samples.epsilon_polar
        kinematics_samples.true_samples(timing)
        momentum = kinematics_samples.get_momentum()

        finalEvents = len(momentum)
        epsilon_azimuthal = finalEvents / resampleSize

        motherParticleResults = kinematics_samples.get_kinematics()

        P_decay_data = motherParticleResults[:, 6]
        P_decay_averaged = np.mean(P_decay_data)

        N_ev_tot = (N_LLP_tot * epsilon_polar * epsilon_azimuthal 
                   * P_decay_averaged * br_visible_val)

        # Check threshold before computing decay products
        if N_ev_tot < min_events_threshold:
            print(f"    N_ev_tot = {N_ev_tot:.6e} < {min_events_threshold}, skipping decay computations...")
            mergeResults.save_total_only(
                LLP.LLP_name, LLP.mass, coupling_squared, c_tau, N_LLP_tot, epsilon_polar, 
                epsilon_azimuthal, P_decay_averaged, br_visible_val, N_ev_tot, uncertainty, 
                LLP.MixingPatternArray, LLP.decayChannels
            )
            print("    Only total results appended.")
            continue
        print(f"    N_ev_tot = {N_ev_tot:.6e} > {min_events_threshold}, proceeding to simulating phase space of decay products...")
        # If above threshold, proceed with decay computations
        unBoostedProducts, size_per_channel = decayProducts.simulateDecays_rest_frame(
            LLP.mass, LLP.PDGs, LLP.BrRatios_distr, finalEvents, LLP.Matrix_elements,
            selected_decay_indices, br_visible_val
        )

        boostedProducts = boost.tab_boosted_decay_products(
            LLP.mass, momentum, unBoostedProducts
        )

        print("    Exporting results...")

        t_export = time.time()
        # Removed redundant N_ev_tot check from mergeResults logic
        mergeResults.save(
            motherParticleResults, boostedProducts, LLP.LLP_name, LLP.mass,
            LLP.MixingPatternArray, LLP.c_tau_input, LLP.decayChannels, size_per_channel,
            finalEvents, epsilon_polar, epsilon_azimuthal, N_LLP_tot, coupling_squared,
            P_decay_averaged, N_ev_tot, br_visible_val, selected_decay_indices,
            uncertainty, ifExportEvents
        )
        print("    Total time spent on exporting: ", time.time() - t_export)

        print(
            f"LLP mass {mass} GeV ({mass_idx}/{total_masses}) and lifetime ctau {c_tau} m "
            f"({c_tau_idx}/{total_c_taus}) has been processed.\n"
            f"Sampled: {finalEvents:.6e}\n"
            f"Squared coupling: {coupling_squared:.6e}\n"
            f"Total number of produced LLPs: {N_LLP_tot:.6e}\n"
            f"Polar acceptance: {epsilon_polar:.6e}\n"
            f"Azimuthal acceptance: {epsilon_azimuthal:.6e}\n"
            f"Averaged decay probability: {P_decay_averaged:.6e}\n"
            f"Visible Br Ratio: {br_visible_val:.6e}\n"
            f"Total number of events: {N_ev_tot:.6e}\n\n"
        )

        print("    Done\n")

