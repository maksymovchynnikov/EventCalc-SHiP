# funcs/LLP_selection.py
import os
import sys
import numpy as np

try:
    resampleSize = int(input("\nEnter the number of events to simulate: "))
    if resampleSize <= 0:
        raise ValueError("The number of events must be a positive integer.")
except ValueError as e:
    raise ValueError(f"Invalid input for the number of events: {e}")

nEvents = resampleSize * 10
N_pot = 6e20

def select_particle():
    main_folder = "./Distributions"
    folders = np.array(os.listdir(main_folder))
                
    print("\nParticle Selector\n")
    for i, folder in enumerate(folders):
        print(f"{i + 1}. {folder}")

    try:
        selected_particle = int(input("Select particle: ")) - 1
        particle_distr_folder = folders[selected_particle]
    except (IndexError, ValueError):
        raise ValueError("Invalid selection. Please select a valid particle.")

    particle_path = os.path.join(main_folder, particle_distr_folder)
    LLP_name = particle_distr_folder.replace("_", " ")

    return {'particle_path': particle_path, 'LLP_name': LLP_name}

particle_selection = select_particle()

def prompt_uncertainty():
    # Always defined. If not Dark-photons, return None.
    if particle_selection['LLP_name'] != "Dark-photons":
        return None
    print("\nWhich variation of the dark photon flux within the uncertainty to select?")
    print("1. lower")
    print("2. central")
    print("3. upper")

    try:
        selected_uncertainty = int(input("Select uncertainty level (1-3): "))
        if selected_uncertainty == 1:
            uncertainty = "lower"
        elif selected_uncertainty == 2:
            uncertainty = "central"
        elif selected_uncertainty == 3:
            uncertainty = "upper"
        else:
            raise ValueError("Invalid selection.")
    except ValueError as e:
        raise ValueError(f"Invalid input for uncertainty level: {e}")
    return uncertainty

uncertainty = prompt_uncertainty()

def prompt_mixing_pattern():
    # Always defined. If not HNL, return None.
    if particle_selection['LLP_name'] != "HNL":
        return None
    try:
        mixing_input = input("\nEnter xi_e, xi_mu, xi_tau: (Ue2, Umu2, Utau2) = U2(xi_e,xi_mu,xi_tau), summing to 1, separated by spaces: ").strip().split()
        if len(mixing_input) != 3:
            raise ValueError("Please enter exactly three numerical values separated by spaces.")
        
        Ue2, Umu2, Utau2 = map(float, mixing_input)
        sumMixingPattern = Ue2 + Umu2 + Utau2
        if sumMixingPattern != 1:
            print("The entered pattern is not normalized by 1. Renormalizing...")
            Ue2 /= sumMixingPattern
            Umu2 /= sumMixingPattern
            Utau2 /= sumMixingPattern

        MixingPatternArray = np.array([Ue2, Umu2, Utau2])
        return MixingPatternArray

    except ValueError as e:
        raise ValueError(f"Invalid input. Please enter three numerical values separated by spaces: {e}")

mixing_pattern = prompt_mixing_pattern()

def prompt_masses_and_c_taus():
    try:
        masses_input = input("\nEnter LLP masses in GeV (separated by spaces): ").split()
        masses = [float(m.rstrip('.')) for m in masses_input]

        ifSameLifetimes = True
        c_taus_list = []
        if ifSameLifetimes:
            c_taus_input = input("Enter lifetimes c*tau in m for all masses (separated by spaces): ")
            c_taus = [float(tau) for tau in c_taus_input.replace(',', ' ').split()]
            c_taus_list = [c_taus] * len(masses)
        
        return masses, c_taus_list
    except ValueError:
        raise ValueError("Invalid input for masses or c*taus. Please enter numerical values.")

def prompt_decay_channels(decayChannels):
    print("\nSelect the decay modes:")
    print("0. All")
    for i, channel in enumerate(decayChannels):
        print(f"{i + 1}. {channel}")
    
    user_input = input("Enter the numbers of the decay channels to select (separated by spaces): ")
    try:
        selected_indices = [int(x) for x in user_input.strip().split()]
        if not selected_indices:
            raise ValueError("No selection made.")
        if 0 in selected_indices:
            return list(range(len(decayChannels)))
        else:
            selected_indices = [x - 1 for x in selected_indices]
            for idx in selected_indices:
                if idx < 0 or idx >= len(decayChannels):
                    raise ValueError(f"Invalid index {idx + 1}.")
            return selected_indices
    except ValueError as e:
        raise ValueError(f"Invalid input for decay channel selection: {e}")

