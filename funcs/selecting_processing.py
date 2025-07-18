# selecting_processing.py
import os
import numpy as np
import re
import sys

def parse_filenames(directory):
    """
    Parses filenames in the given directory and its subdirectories to extract LLP names, masses, lifetimes, mixing patterns, uncertainty and model choices.
    Returns a dictionary llp_dict[llp_name][(mass, lifetime)][mixing_patterns_or_uncertainty_or_model] = filepath
    """
    llp_dict = {}  # LLP_name: { (mass, lifetime): { mixing_patterns_or_uncertainty: filepath } }

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('_data.dat'):
                filepath = os.path.relpath(os.path.join(root, filename), directory)
                # Extract LLP_name from the parent directory of 'eventData'
                rel_path = os.path.relpath(root, directory)
                llp_name = os.path.basename(os.path.dirname(rel_path))
                # Parse filename to extract mass, lifetime, mixing patterns or uncertainty
                base_name = filename[:-len('_data.dat')]
                tokens = base_name.split('_')
                
                if llp_name == "HNL":
                    # Expected pattern: HNL_mass_c_tau_mixing1_mixing2_mixing3_data.dat
                    if len(tokens) < 3:
                        print(f"Filename {filename} does not have enough tokens for HNL.")
                        continue
                    try:
                        mass = float(tokens[1])
                        lifetime = float(tokens[2])
                        mixing_patterns = tuple(float(tok) for tok in tokens[3:6]) if len(tokens) >= 6 else None
                    except ValueError:
                        print(f"Invalid numerical values in filename {filename}. Skipping.")
                        continue
                    # Store in llp_dict
                    if llp_name not in llp_dict:
                        llp_dict[llp_name] = {}
                    mass_lifetime = (mass, lifetime)
                    if mass_lifetime not in llp_dict[llp_name]:
                        llp_dict[llp_name][mass_lifetime] = {}
                    llp_dict[llp_name][mass_lifetime][mixing_patterns] = filepath
                elif llp_name == "Dark-photons":
                    # Expected pattern: Dark-photons_mass_c_tau_uncertainty_data.dat
                    if len(tokens) < 4:
                        print(f"Filename {filename} does not have enough tokens for Dark-photons.")
                        continue
                    try:
                        mass = float(tokens[1])
                        lifetime = float(tokens[2])
                        uncertainty_choice = tokens[3]  # 'lower', 'central', 'upper'
                        if uncertainty_choice not in ['lower', 'central', 'upper']:
                            print(f"Invalid uncertainty choice '{uncertainty_choice}' in filename {filename}. Skipping.")
                            continue
                    except ValueError:
                        print(f"Invalid numerical values in filename {filename}. Skipping.")
                        continue
                    # Store in llp_dict
                    if llp_name not in llp_dict:
                        llp_dict[llp_name] = {}
                    mass_lifetime = (mass, lifetime)
                    if mass_lifetime not in llp_dict[llp_name]:
                        llp_dict[llp_name][mass_lifetime] = {}
                    llp_dict[llp_name][mass_lifetime][uncertainty_choice] = filepath

                elif llp_name == "Inelastic-DM":
                    # Expected pattern: Inelastic-DM_mass_c_tau_uncertainty_modelNum_data.dat
                    if len(tokens) < 5:
                        print(f"[WARN] {filename}: not enogh tokens for Inelastic-DM")
                        continue
                    try:
                        mass = float(tokens[1])
                        lifetime = float(tokens[2])
                        uncertainty_choice = tokens[3] # 'lower', 'central', 'upper'
                        model_num = int(tokens[4].strip("Model")) # 1, 2, 3 or 4
                        if uncertainty_choice not in ['lower', 'central', 'upper']:
                            print(f"Invalid uncertainty choice '{uncertainty_choice}' in filename {filename}. Skipping.")
                            continue
                        elif model_num not in [1, 2, 3, 4]:
                            print(f"Invalid model number '{model_num}' in filename {filename}. Skipping.")
                    except ValueError:
                        print(f"Invalid numerical values in filename {filename}. Skipping.")
                        continue
                    # Store in llp_dict
                    if llp_name not in llp_dict:
                        llp_dict[llp_name] = {}
                    mass_lifetime = (mass, lifetime)
                    if mass_lifetime not in llp_dict[llp_name]:
                        llp_dict[llp_name][mass_lifetime] = {}
                    llp_dict[llp_name][mass_lifetime][(uncertainty_choice, model_num)] = filepath
                else:
                    # For other LLPs, expected pattern: LLP_name_mass_c_tau_data.dat
                    if len(tokens) < 3:
                        print(f"Filename {filename} does not have enough tokens for LLP '{llp_name}'. Skipping.")
                        continue
                    try:
                        mass = float(tokens[1])
                        lifetime = float(tokens[2])
                        mixing_patterns_or_uncertainty_or_model = None  # No additional identifiers
                    except ValueError:
                        print(f"Invalid numerical values in filename {filename}. Skipping.")
                        continue
                    # Store in llp_dict
                    if llp_name not in llp_dict:
                        llp_dict[llp_name] = {}
                    mass_lifetime = (mass, lifetime)
                    if mass_lifetime not in llp_dict[llp_name]:
                        llp_dict[llp_name][mass_lifetime] = {}
                    llp_dict[llp_name][mass_lifetime][mixing_patterns_or_uncertainty_or_model] = filepath
            else:
                continue  # Skip files not ending with '_data.dat'
    return llp_dict

def user_selection(llp_dict):
    """
    Allows the user to select an LLP, mass-lifetime combination, and mixing patterns or uncertainty choices.
    Returns the selected filepath, the selected LLP name, mass, lifetime, and mixing patterns or uncertainty.
    """
    print("Available LLPs:")
    llp_names_list = sorted(llp_dict.keys())
    for i, llp_name in enumerate(llp_names_list):
        print(f"{i+1}. {llp_name}")

    # Ask user to choose an LLP
    while True:
        try:
            choice = int(input("Choose an LLP by typing the number: "))
            if 1 <= choice <= len(llp_names_list):
                break
            else:
                print(f"Please enter a number between 1 and {len(llp_names_list)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    selected_llp = llp_names_list[choice - 1]
    print(f"Selected LLP: {selected_llp}")

    # Get available mass-lifetime combinations
    mass_lifetime_list = sorted(llp_dict[selected_llp].keys())
    print(f"Available mass-lifetime combinations for {selected_llp}:")
    for i, (mass, lifetime) in enumerate(mass_lifetime_list):
        print(f"{i+1}. mass={mass:.2e} GeV, lifetime={lifetime:.2e} s")

    # Ask user to choose a mass-lifetime
    while True:
        try:
            mass_lifetime_choice = int(input("Choose a mass-lifetime combination by typing the number: "))
            if 1 <= mass_lifetime_choice <= len(mass_lifetime_list):
                break
            else:
                print(f"Please enter a number between 1 and {len(mass_lifetime_list)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")
    selected_mass_lifetime = mass_lifetime_list[mass_lifetime_choice - 1]
    selected_mass, selected_lifetime = selected_mass_lifetime
    print(f"Selected mass: {selected_mass:.2e} GeV, lifetime: {selected_lifetime:.2e} s")

    # Get mixing patterns or uncertainty choices
    sub_dict = llp_dict[selected_llp][selected_mass_lifetime]
    options_list = sorted(sub_dict.keys(), key=lambda x: (x is None, x))
    if selected_llp == "HNL":
        if any(options_list):
            print(f"Available mixing patterns for {selected_llp} with mass {selected_mass:.2e} GeV and lifetime {selected_lifetime:.2e} s:")
            for i, mixing_patterns in enumerate(options_list):
                if mixing_patterns is None:
                    print(f"{i+1}. None")
                else:
                    print(f"{i+1}. {mixing_patterns}")
            # Ask user to choose a mixing pattern
            while True:
                try:
                    mixing_choice = int(input("Choose a mixing pattern by typing the number: "))
                    if 1 <= mixing_choice <= len(options_list):
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(options_list)}.")
                except ValueError:
                    print("Invalid input. Please enter a valid number.")
            selected_mixing_patterns = options_list[mixing_choice - 1]
            print(f"Selected mixing pattern: {selected_mixing_patterns}")
        else:
            selected_mixing_patterns = None
    elif selected_llp == "Dark-photons":
        if any(options_list):
            print(f"Available uncertainty choices for {selected_llp} with mass {selected_mass:.2e} GeV and lifetime {selected_lifetime:.2e} s:")
            for i, uncertainty_choice in enumerate(options_list):
                print(f"{i+1}. {uncertainty_choice}")
            # Ask user to choose an uncertainty choice
            while True:
                try:
                    uncertainty_choice_num = int(input("Choose an uncertainty choice by typing the number: "))
                    if 1 <= uncertainty_choice_num <= len(options_list):
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(options_list)}.")
                except ValueError:
                    print("Invalid input. Please enter a valid number.")
            selected_uncertainty = options_list[uncertainty_choice_num - 1]
            print(f"Selected uncertainty choice: {selected_uncertainty}")
            selected_mixing_patterns = selected_uncertainty
        else:
            selected_mixing_patterns = None
    elif selected_llp == "Inelastic-DM":
        if any(options_list):
            print(f"Available uncertainty and model choices for {selected_llp} with mass {selected_mass:.2e} GeV and lifetime {selected_lifetime:.2e} s:")
            for i, (uncertainty_choice, model) in enumerate(options_list):
                print(f"{i + 1}. {uncertainty_choice}, {model}")
            # Ask user to choose an uncertainty choice
            while True:
                try:
                    uncertainty_model_choice_num = int(input("Choose an uncertainty and model choice by typing the number: "))
                    if 1 <= uncertainty_model_choice_num <= len(options_list):
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(options_list)}.")
                except ValueError:
                    print("Invalid input. Please enter a valid number.")
            selected_uncertainty_model = options_list[uncertainty_model_choice_num - 1]
            print(f"Selected uncertainty choice: {selected_uncertainty_model}")
            selected_mixing_patterns = selected_uncertainty_model
        else:
            selected_mixing_patterns = None
            
    else:
        selected_mixing_patterns = None  # For other LLPs

    # Find the file matching the selection
    selected_filepath = sub_dict[selected_mixing_patterns]
    print(f"Selected file: {selected_filepath}")

    return selected_filepath, selected_llp, selected_mass, selected_lifetime, selected_mixing_patterns

def read_file(filepath):
    """
    Reads the file at the given filepath.
    Returns finalEvents, epsilon_polar, epsilon_azimuthal, br_visible_val, coupling_squared, channels
    """
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        # Updated pattern to include Squared coupling
        pattern = (
            r'Sampled\s+(?P<finalEvents>[\d\.\+\-eE]+)\s+events inside SHiP volume\. '
            r'Squared coupling:\s+(?P<coupling_squared>[\d\.\+\-eE]+)\. '
            r'Total number of produced LLPs:\s+(?P<N_LLP_tot>[\d\.\+\-eE]+)\. '
            r'Polar acceptance:\s+(?P<epsilon_polar>[\d\.\+\-eE]+)\. '
            r'Azimuthal acceptance:\s+(?P<epsilon_azimuthal>[\d\.\+\-eE]+)\. '
            r'Averaged decay probability:\s+(?P<P_decay_averaged>[\d\.\+\-eE]+)\. '
            r'Visible Br Ratio:\s+(?P<br_visible_val>[\d\.\+\-eE]+)\. '
            r'Total number of events:\s+(?P<N_ev_tot>[\d\.\+\-eE]+)'
        )
        match = re.match(pattern, first_line)
        if match:
            finalEvents = float(match.group('finalEvents'))
            coupling_squared = float(match.group('coupling_squared'))
            N_LLP_tot = float(match.group('N_LLP_tot'))
            epsilon_polar = float(match.group('epsilon_polar'))
            epsilon_azimuthal = float(match.group('epsilon_azimuthal'))
            P_decay_averaged = float(match.group('P_decay_averaged'))
            br_visible_val = float(match.group('br_visible_val'))
            N_ev_tot = float(match.group('N_ev_tot'))
            print(f"finalEvents: {finalEvents}, coupling_squared: {coupling_squared}, N_LLP_tot: {N_LLP_tot}, epsilon_polar: {epsilon_polar}, "
                  f"epsilon_azimuthal: {epsilon_azimuthal}, P_decay_averaged: {P_decay_averaged}, "
                  f"br_visible_val: {br_visible_val}, N_ev_tot: {N_ev_tot}")
        else:
            print("Error: First line does not match expected format.")
            sys.exit(1)

        # Skip any empty lines
        while True:
            line = f.readline()
            if not line:
                break
            if line.strip() != '':
                break
        # Now process the rest of the file
        # Extract channels and sample_points
        channels = {}
        current_channel = None
        current_channel_size = 0
        current_data = []
        # If the line we just read is a channel header, process it
        if line.strip().startswith('#<process='):
            match = re.match(
                r'#<process=(?P<channel>.*?);\s*sample_points=(?P<channel_size>[\d\.\+\-eE]+)>', line.strip())
            if match:
                current_channel = match.group('channel')
                current_channel_size = int(float(match.group('channel_size')))
                current_data = []
            else:
                print(f"Error: Could not parse channel line: {line}")
        else:
            print("Error: Expected channel header after first line.")
            sys.exit(1)

        # Continue reading the file
        for line in f:
            line = line.strip()
            if line.startswith('#<process='):
                # This is a new channel
                match = re.match(
                    r'#<process=(?P<channel>.*?);\s*sample_points=(?P<channel_size>[\d\.\+\-eE]+)>', line)
                if match:
                    if current_channel is not None:
                        # Save the data of the previous channel
                        channels[current_channel] = {
                            'size': current_channel_size, 'data': current_data}
                    current_channel = match.group('channel')
                    current_channel_size = int(float(match.group('channel_size')))
                    current_data = []
                else:
                    print(f"Error: Could not parse channel line: {line}")
            elif line == '':
                # Empty line, skip
                continue
            else:
                # This is data
                current_data.append(line)

        # After the loop, save the last channel's data
        if current_channel is not None:
            channels[current_channel] = {
                'size': current_channel_size, 'data': current_data}

    return finalEvents, epsilon_polar, epsilon_azimuthal, br_visible_val, coupling_squared, channels

