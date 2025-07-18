#!/usr/bin/env python3
# simulate.py – original workflow with θ_max_sim forwarding

import os, sys, time, math, numpy as np, pandas as pd
from funcs.ship_setup import (
    z_min, z_max, Delta_x_in, Delta_x_out, Delta_y_in, Delta_y_out,
    theta_max_dec_vol
)

# ---------------------------------------------------------------------
# Angle range used by the kinematics sampler
# ---------------------------------------------------------------------
theta_max_sim = theta_max_dec_vol          # **keep this**

print("\nSHiP setup (modify ship_setup.py if needed):\n")
print(f"z_min={z_min} m, z_max={z_max} m, "
      f"Δx_in={Delta_x_in} m, Δx_out={Delta_x_out} m, "
      f"Δy_in={Delta_y_in} m, Δy_out={Delta_y_out} m, "
      f"θ_max_dec_vol={theta_max_dec_vol:.6f} rad\n")

# --- local imports ----------------------------------------------------
from funcs import initLLP, decayProducts, boost, kinematics, mergeResults
from funcs.LLP_selection import (
    prompt_masses_and_c_taus, prompt_decay_channels, 
    resampleSize, particle_selection, mixing_pattern, 
    uncertainty, nEvents, N_pot, model, couplingSqr, 
    couplingSqr_list 
)

from funcs.plot_phenomenology import (
    plot_production_probability, plot_lifetime, plot_branching_ratios
)
from funcs.inelasticDMCalcs import (
    create_temp_particles, get_inelastic_dm_ctau
)
from funcs.inelasticDMTables import (
    _cleanup_idm_tables
)
from funcs.SDP_Decays import (
    SDP_Decay
)

# -------------------- LLP selection / plots ---------------------------
LLP = initLLP.LLP(
    mass=None, 
    particle_selection=particle_selection, 
    mixing_pattern=mixing_pattern, 
    uncertainty=uncertainty
)
selected_decay_indices = prompt_decay_channels(LLP.decayChannels)

masses, c_taus_list = prompt_masses_and_c_taus()

timing = False
total_masses        = len(masses)
min_events_threshold = 0.1
ifExportEvents       = True

total_masses = len(masses)
print(f"\nTotal masses to process: {total_masses}")

if LLP.LLP_name == "Short-Dark-photons":
    LLP, masses, selected_decay_indices = SDP_Decay(LLP, masses, selected_decay_indices)
    # compute ctaus manually for iDM
    ctau_matrix = []
    for m in masses:
        ctaus_for_m = [get_inelastic_dm_ctau(m, model, g) for g in couplingSqr_list]
        ctau_matrix.append(ctaus_for_m)
    c_taus_list = ctau_matrix
    print("\nComputed lifetimes for requested couplings:")
    for m, ctaus in zip(masses, c_taus_list):
        print(f"  m = {m:g} GeV:")
        for g, ct in zip(couplingSqr_list, ctaus):
            print(f"    ε² = {g:.3e}  →  cτ = {ct:.3e} m")

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

for mass_idx, (mass, c_taus) in enumerate(zip(masses, c_taus_list), start=1):
    if not (LLP.m_min_tabulated <= mass <= LLP.m_max_tabulated):
        print(f"Mass {mass} GeV outside tabulated range. Skipping.")
        continue

    print(f"\nProcessing mass {mass} GeV  ({mass_idx}/{total_masses})")
    LLP.set_mass(mass)

    if LLP.LLP_name == "Inelastic-DM":
        create_temp_particles(masses[mass_idx - 1], model) # creates a chi particle for idm to decay to

    LLP.compute_mass_dependent_properties()

    br_visible_val = sum(LLP.BrRatios_distr[i] for i in selected_decay_indices)

    if br_visible_val == 0:
        print("No visible decay channels at this mass. Skipping.")
        continue

    
    c_tau_values      = c_taus if isinstance(c_taus, (list, tuple, np.ndarray)) else [c_taus]
    total_c_taus      = len(c_tau_values)
    print(f"  Lifetimes to process: {total_c_taus}")

    for c_tau_idx, c_tau in enumerate(c_tau_values, 1):
        print(f"  Processing c_tau = {c_tau} m  ({c_tau_idx}/{total_c_taus})")

        LLP.set_c_tau(c_tau)

        if LLP.LLP_name == "Inelastic-DM":
            # We overrode c_tau_values from couplingSqr_list, so match indices.
            # When c_tau_values came from user lifetimes (fallback), fall back to scalar couplingSqr.
            if isinstance(c_taus, (list, tuple, np.ndarray)) and couplingSqr_list is not None:
                coupling_squared = couplingSqr_list[c_tau_idx - 1]
            else:
                coupling_squared = couplingSqr  # legacy single-coupling mode
        else:
            coupling_squared = (
                LLP.c_tau_int / c_tau if LLP.LLP_name != "Scalar-quartic" else 0.01
            )
        N_LLP_tot        = N_pot * LLP.Yield * coupling_squared
        if LLP.Yield * coupling_squared < 1e-21:
            print("    Negligible yield. Skipping.")
            continue

        t = time.time()


        if LLP.LLP_name == "Inelastic-DM":
            # use distribution produced for single mass only 
            distr_m  = LLP.Distr.loc[LLP.Distr[0] == mass].reset_index(drop=True)
            emax_m   = LLP.Energy_distr.loc[LLP.Energy_distr[0] == mass].reset_index(drop=True)
            # -------- kinematic sampling (with θ_max_sim) -----------------
            kin = kinematics.Grids(
                distr_m, emax_m,
                nEvents, LLP.mass, LLP.c_tau_input, 
                theta_max_sim=theta_max_sim
            )
        else:
            # -------- kinematic sampling (with θ_max_sim) -----------------
            kin = kinematics.Grids(
                LLP.Distr, LLP.Energy_distr,
                nEvents, LLP.mass, LLP.c_tau_input,
                theta_max_sim=theta_max_sim
            )

        kin.interpolate(False)
        kin.resample(resampleSize, False)
        #np.savetxt(f"angle-energy-{LLP.LLP_name}-{LLP.mass}.txt",
        #           np.column_stack((kin.get_theta(), kin.get_energy())),
        #           fmt='%.8e')
        kin.true_samples(False)
        momentum       = kin.get_momentum()

        finalEvents    = len(momentum)
        epsilon_polar  = kin.epsilon_polar
        epsilon_azimuthal = finalEvents / resampleSize

        motherParticleResults = kin.get_kinematics()      # <— fixed name
        P_decay_averaged      = motherParticleResults[:, 6].mean()

        N_ev_tot = (N_LLP_tot * epsilon_polar * epsilon_azimuthal *
                    P_decay_averaged * br_visible_val)

        # Check threshold before computing decay products
        if N_ev_tot < min_events_threshold:
            print(f"    N_ev_tot = {N_ev_tot:.6e} < {min_events_threshold}. "
                  "Skipping decay computation...")
            mergeResults.save_total_only(
                LLP.LLP_name, LLP.mass, coupling_squared, c_tau, N_LLP_tot, epsilon_polar, 
                epsilon_azimuthal, P_decay_averaged, br_visible_val, N_ev_tot, uncertainty, 
                model, LLP.MixingPatternArray, LLP.decayChannels
            )
            continue
        
        # ---- simulate visible decays -------------------------------
        unBoostedProducts, size_per_channel = decayProducts.simulateDecays_rest_frame(
            LLP.mass, LLP.PDGs, LLP.BrRatios_distr, finalEvents, LLP.Matrix_elements,
            selected_decay_indices, br_visible_val
        )

        boostedProducts = boost.tab_boosted_decay_products(
            LLP.mass, momentum, unBoostedProducts
        )

        # ---- save ----------------------------------------------------
        t0 = time.time()

        mergeResults.save(
            motherParticleResults, boostedProducts, LLP.LLP_name, LLP.mass,
            LLP.MixingPatternArray, LLP.c_tau_input, LLP.decayChannels, size_per_channel,
            finalEvents, epsilon_polar, epsilon_azimuthal, N_LLP_tot, coupling_squared,
            P_decay_averaged, N_ev_tot, br_visible_val, selected_decay_indices,
            uncertainty, model, ifExportEvents
        )
        print(f"    Exported in {time.time() - t0:.1f} s")

        print(
            f"LLP mass {mass} GeV ({mass_idx}/{total_masses}) "
            f"cτ {c_tau} m ({c_tau_idx}/{total_c_taus}) processed.\n"
            f"Sampled inside volume: {finalEvents:.6e}\n"
            f"Squared coupling:      {coupling_squared:.6e}\n"
            f"N_LLP_total:           {N_LLP_tot:.6e}\n"
            f"ε_polar:               {epsilon_polar:.6e}\n"
            f"ε_azimuthal:           {epsilon_azimuthal:.6e}\n"
            f"⟨P_decay⟩:             {P_decay_averaged:.6e}\n"
            f"Visible Br:            {br_visible_val:.6e}\n"
            f"N_events_tot:          {N_ev_tot:.6e}\n"
        )
    
if LLP.LLP_name == "Inelastic-DM":
    _cleanup_idm_tables() # gets rid of now redundant idm tables