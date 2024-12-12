# simulate.py
import sys
import os
import time
import numpy as np
import math

from funcs.ship_setup import (
    z_min, z_max, Delta_x_in, Delta_x_out, Delta_y_in, Delta_y_out, theta_max_dec_vol
)

# Print SHiP setup information
print("SHiP setup (modify ship_setup.py if needed):\n")
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

for mass, c_taus in zip(masses, c_taus_list):
    # Check if mass within tabulated range
    if not (LLP.m_min_tabulated < mass < LLP.m_max_tabulated):
        print(f"The current mass {mass} is outside the tabulated data range "
              f"({LLP.m_min_tabulated}, {LLP.m_max_tabulated}). Skipping...")
        continue

    print(f"\nProcessing mass {mass}")
    LLP.set_mass(mass)
    LLP.compute_mass_dependent_properties()

    br_visible_val = sum(LLP.BrRatios_distr[idx] for idx in selected_decay_indices)

    if br_visible_val == 0:
        print("No decay events for these decay modes at this mass. Skipping...")
        continue

    for c_tau in c_taus:
        print(f"  Processing c_tau {c_tau}")
        LLP.set_c_tau(c_tau)

        coupling_squared = LLP.c_tau_int / c_tau
        if LLP.LLP_name != "Scalar-quartic":
            yield_times_coupling = LLP.Yield * coupling_squared
        else:
            yield_times_coupling = LLP.Yield * 0.01
        N_LLP_tot = N_pot * yield_times_coupling 
        
        if yield_times_coupling < 1e-21:
            print("The overall yield of produced LLPs is effectively zero for this mass. Skipping...")
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

        unBoostedProducts, size_per_channel = decayProducts.simulateDecays_rest_frame(
            LLP.mass, LLP.PDGs, LLP.BrRatios_distr, finalEvents, LLP.Matrix_elements,
            selected_decay_indices, br_visible_val
        )
        boostedProducts = boost.tab_boosted_decay_products(
            LLP.mass, momentum, unBoostedProducts
        )

        motherParticleResults = kinematics_samples.get_kinematics()
        decayProductsResults = boostedProducts

        P_decay_data = motherParticleResults[:, 6]
        P_decay_averaged = np.mean(P_decay_data)

        N_ev_tot = (N_LLP_tot * epsilon_polar * epsilon_azimuthal 
                   * P_decay_averaged * br_visible_val)

        print(f"    Exporting results...")

        t_export = time.time()
        mergeResults.save(
            motherParticleResults, decayProductsResults, LLP.LLP_name, LLP.mass,
            LLP.MixingPatternArray, LLP.c_tau_input, LLP.decayChannels, size_per_channel,
            finalEvents, epsilon_polar, epsilon_azimuthal, N_LLP_tot, coupling_squared,
            P_decay_averaged, N_ev_tot, br_visible_val, selected_decay_indices
        )
        print("    Total time spent on exporting: ", time.time() - t_export)
        
        # Print all relevant information at the end of the iteration before marking as done
        print(
            f"Sampled {finalEvents:.6e} events inside SHiP volume. "
            f"Squared coupling: {coupling_squared:.6e}. "
            f"Total number of produced LLPs: {N_LLP_tot:.6e}. "
            f"Polar acceptance: {epsilon_polar:.6e}. "
            f"Azimuthal acceptance: {epsilon_azimuthal:.6e}. "
            f"Averaged decay probability: {P_decay_averaged:.6e}. "
            f"Visible Br Ratio: {br_visible_val:.6e}. " 
            f"Total number of events: {N_ev_tot:.6e}\n\n"
        )

        print(f"    Done\n")
        
# As before, branching ratios and decay widths are interpolated by reading their tabular data and
# using RegularGridInterpolator to provide continuous interpolation within the valid mass range.

