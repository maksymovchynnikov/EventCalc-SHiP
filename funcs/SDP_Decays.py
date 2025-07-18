# SDP_Decays.py
import os
import numpy as np
import copy
import math

from funcs import initLLP, decayProducts, boost, kinematics
from funcs.LLP_selection import (
    prompt_decay_channels, 
    particle_selection, mixing_pattern, 
    uncertainty, model, couplingSqr 
)

from funcs.inelasticDMCalcs import (
    create_temp_particles, compute_masses
)

from funcs.inelasticDMTables import (
    extract_chipr_tables
)

# c_tau is arbitrary, as particles are sampled right away
# but should not be small as this will cause unphysical 
# energy value samples
C_TAU = 70

RESAMPLE = 100000 # number of decays to resample
N_POINTS = RESAMPLE * 10 # number of grid points in kinematics

timing = False

def SDP_Decay(LLP, masses, selected_decay_indices):

    # the tabulated masses are actually for the inelastic dark matter, so save for later
    masses_DM = copy.deepcopy(masses)
    masses = [compute_masses(mass, model)[1] for mass in masses] # replace with dark photon mass

    mass_remove_ind = [] # keep track of masses we want to remove

    for mass_idx, mass in enumerate(masses, start=1):

        if not (LLP.m_min_tabulated < mass < LLP.m_max_tabulated):
            print(f"The current mass {masses_DM[mass_idx - 1]} is too large, results in dark photon of"
                  f"mass {mass} outside range {LLP.m_min_tabulated}-{LLP.m_max_tabulated}GeV")
            mass_remove_ind.append(mass_idx - 1) # remove this mass later
            continue

        create_temp_particles(masses_DM[mass_idx - 1], model) # create particles for SDP to decay into for given mass
        
        print(f"\nProcessing SDP mass {mass} GeV")
        LLP.set_mass(mass)

        LLP.compute_mass_dependent_properties()

        br_visible_val = sum(LLP.BrRatios_distr[idx] for idx in selected_decay_indices)

        LLP.set_c_tau(C_TAU)

        kinematics_samples = kinematics.Grids(
            LLP.Distr, LLP.Energy_distr, N_POINTS, LLP.mass, LLP.c_tau_input, theta_max_sim=math.pi/2
        )

        kinematics_samples.interpolate(timing)
        kinematics_samples.resample(RESAMPLE, timing)
        #np.savetxt(f"angle-energy-{LLP.LLP_name}-{LLP.mass}.txt",
        #           np.column_stack((kinematics_samples.get_theta(), kinematics_samples.get_energy())),
        #           fmt='%.8e')
        kinematics_samples.get_instant_decay_momentum(timing)
        momentum = kinematics_samples.get_momentum()

        finalEvents = len(momentum)

        unBoostedProducts, size_per_channel = decayProducts.simulateDecays_rest_frame(
            LLP.mass, LLP.PDGs, LLP.BrRatios_distr, finalEvents, LLP.Matrix_elements,
            selected_decay_indices, br_visible_val, process_in_pythia=False
        )

        boostedProducts = boost.tab_boosted_decay_products(
            LLP.mass, momentum, unBoostedProducts
        )

        print(f"Extracting distirbutions for iDM mass {masses_DM[mass_idx - 1]} GeV...")

        # create tables that will be used for iDM decays
        extract_chipr_tables(boostedProducts)

    # now create inelastic DM to simulate decays of

    try: 
        # if no products have been boosted we exit the simulation
        boostedProducts 
    except NameError:
        exit()


    particle_selection['LLP_name'] = "Inelastic-DM"
    particle_selection['particle_path'] = os.path.join("./Distributions", "Inelastic-DM")
    particle_selection['model'] = model
    particle_selection['couplingSqr'] = couplingSqr

    masses = []
    for i, val in enumerate(masses_DM):
        if i not in (mass_remove_ind):
            masses.append(val)

    LLP = initLLP.LLP(
        mass=None,
        particle_selection=particle_selection,
        mixing_pattern=mixing_pattern,
        uncertainty=uncertainty
    )

    print(f"\nFinished compiling data tables for iDM.")
    
    selected_decay_indices = prompt_decay_channels(LLP.decayChannels)

    return LLP, masses, selected_decay_indices