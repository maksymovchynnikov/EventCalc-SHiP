import os
import numpy as np
import re
import sys

def parse_filenames(directory):
    """
    Parses filenames in the given directory and its subdirectories to extract LLP names, masses, lifetimes, and mixing patterns.
    Returns a dictionary llp_dict[llp_name][(mass, lifetime)][mixing_patterns] = filepath
    """
    llp_dict = {}  # LLP_name: { (mass, lifetime): { mixing_patterns: filepath } }

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('_decayProducts.dat'):
                filepath = os.path.relpath(os.path.join(root, filename), directory)
                # Extract LLP_name from the parent directory of 'eventData'
                rel_path = os.path.relpath(root, directory)
                llp_name = os.path.basename(os.path.dirname(rel_path))
                # Parse filename to extract mass, lifetime, and mixing patterns
                base_name = filename[:-len('_decayProducts.dat')]
                tokens = base_name.split('_')
                # Identify indices of tokens that can be converted to float
                float_token_indices = []
                for i, token in enumerate(tokens):
                    try:
                        float(token)
                        float_token_indices.append(i)
                    except ValueError:
                        continue
                if len(float_token_indices) >= 2:
                    mass_index = float_token_indices[0]
                    # mass and lifetime
                    mass = tokens[mass_index]
                    lifetime = tokens[mass_index + 1]
                    # Mixing patterns, if any
                    mixing_pattern_tokens = tokens[mass_index + 2:]
                    mixing_patterns = []
                    for token in mixing_pattern_tokens:
                        try:
                            mixing_patterns.append(float(token))
                        except ValueError:
                            print(f"Invalid mixing pattern token: {token} in filename {filename}")
                    # Convert mass and lifetime to float
                    mass = float(mass)
                    lifetime = float(lifetime)
                    # Store in llp_dict
                    if llp_name not in llp_dict:
                        llp_dict[llp_name] = {}
                    mass_lifetime = (mass, lifetime)
                    if mass_lifetime not in llp_dict[llp_name]:
                        llp_dict[llp_name][mass_lifetime] = {}
                    mixing_patterns_tuple = tuple(mixing_patterns) if mixing_patterns else None
                    llp_dict[llp_name][mass_lifetime][mixing_patterns_tuple] = filepath
                else:
                    print(f"Filename {filename} does not have enough float tokens to extract mass and lifetime.")
            else:
                continue  # Skip files not ending with '_decayProducts.dat'
    return llp_dict

def user_selection(llp_dict):
    """
    Allows the user to select an LLP, mass-lifetime combination, and mixing patterns.
    Returns the selected filepath, the selected LLP name, mass, lifetime, and mixing patterns.
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

    # Get mixing patterns
    mixing_patterns_dict = llp_dict[selected_llp][selected_mass_lifetime]
    mixing_patterns_list = sorted(mixing_patterns_dict.keys(), key=lambda x: (x is None, x))
    if any(mixing_patterns_list):
        print(f"Available mixing patterns for {selected_llp} with mass {selected_mass:.2e} GeV and lifetime {selected_lifetime:.2e} s:")
        for i, mixing_patterns in enumerate(mixing_patterns_list):
            if mixing_patterns is None:
                print(f"{i+1}. None")
            else:
                print(f"{i+1}. {mixing_patterns}")
        # Ask user to choose a mixing pattern
        while True:
            try:
                mixing_choice = int(input("Choose a mixing pattern by typing the number: "))
                if 1 <= mixing_choice <= len(mixing_patterns_list):
                    break
                else:
                    print(f"Please enter a number between 1 and {len(mixing_patterns_list)}.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        selected_mixing_patterns = mixing_patterns_list[mixing_choice - 1]
        print(f"Selected mixing pattern: {selected_mixing_patterns}")
    else:
        selected_mixing_patterns = None

    # Find the file matching the selection
    selected_filepath = mixing_patterns_dict[selected_mixing_patterns]
    print(f"Selected file: {selected_filepath}")

    return selected_filepath, selected_llp, selected_mass, selected_lifetime, selected_mixing_patterns

def read_file(filepath):
    """
    Reads the file at the given filepath.
    Returns finalEvents, epsilon_polar, epsilon_azimuthal, br_visible_val, channels
    """
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        # Expected format:
        # Sampled {finalEvents} events inside SHiP volume. Total number of produced LLPs: {N_LLP_tot}. Polar acceptance: {epsilon_polar}. Azimuthal acceptance: {epsilon_azimuthal}. Averaged decay probability: {P_decay_averaged}. Visible Br Ratio: {br_visible_val:.6e}. Total number of events: {N_ev_tot}
        pattern = (
            r'Sampled\s+(?P<finalEvents>[\d\.\+\-eE]+)\s+events inside SHiP volume\. '
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
            N_LLP_tot = float(match.group('N_LLP_tot'))
            epsilon_polar = float(match.group('epsilon_polar'))
            epsilon_azimuthal = float(match.group('epsilon_azimuthal'))
            P_decay_averaged = float(match.group('P_decay_averaged'))
            br_visible_val = float(match.group('br_visible_val'))
            N_ev_tot = float(match.group('N_ev_tot'))
            print(f"finalEvents: {finalEvents}, N_LLP_tot: {N_LLP_tot}, epsilon_polar: {epsilon_polar}, "
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

    return finalEvents, epsilon_polar, epsilon_azimuthal, br_visible_val, channels

