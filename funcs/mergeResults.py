# mergeResults.py
import os
import numpy as np

def save(
    motherParticleResults, 
    decayProductsResults, 
    LLP_name, 
    mass, 
    MixingPatternArray, 
    c_tau, 
    decayChannels, 
    size_per_channel, 
    finalEvents, 
    epsilon_polar, 
    epsilon_azimuthal, 
    N_LLP_tot, 
    coupling_squared, 
    P_decay_averaged, 
    N_ev_tot, 
    br_visible_val, 
    selected_decay_indices, 
    uncertainty,
    ifExportEvents  # Added parameter
):
    """
    Saves simulation results to data files.

    Parameters:
    - motherParticleResults (np.ndarray): Array containing mother particle kinematics.
    - decayProductsResults (np.ndarray): Array containing decay products kinematics.
    - LLP_name (str): Name of the Long-Lived Particle (LLP).
    - mass (float): Mass of the LLP.
    - MixingPatternArray (np.ndarray or None): Array representing the mixing pattern.
    - c_tau (float): Proper lifetime of the LLP.
    - decayChannels (list): List of possible decay channels.
    - size_per_channel (list): Number of events per decay channel.
    - finalEvents (int): Total number of final events.
    - epsilon_polar (float): Polar acceptance.
    - epsilon_azimuthal (float): Azimuthal acceptance.
    - N_LLP_tot (float): Total number of produced LLPs.
    - coupling_squared (float): Squared coupling constant.
    - P_decay_averaged (float): Averaged decay probability.
    - N_ev_tot (float): Total number of events.
    - br_visible_val (float): Visible Branching Ratio.
    - selected_decay_indices (list): Indices of selected decay channels.
    - uncertainty (float or None): Uncertainty parameter.
    - ifExportEvents (bool): Flag to determine whether to export event data files.
    """

    # Combine mother particle results and decay products results
    # Assuming both arrays have the same number of rows
    results = np.concatenate((motherParticleResults, decayProductsResults), axis=1)
    
    # Create base output directory
    base_output_dir = os.path.join('.', 'outputs', LLP_name)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create subdirectories for event data and total files
    eventData_dir = os.path.join(base_output_dir, 'eventData')
    total_dir = os.path.join(base_output_dir, 'total')
    os.makedirs(eventData_dir, exist_ok=True)
    os.makedirs(total_dir, exist_ok=True)
    
    # Create the output file name for data.dat based on LLP type and parameters
    if LLP_name == "HNL":
        if MixingPatternArray is not None and isinstance(MixingPatternArray, np.ndarray):
            mixing_str = '_'.join([f"{mp:.3e}" for mp in MixingPatternArray])
            outputfileName = os.path.join(
                eventData_dir, 
                f'{LLP_name}_{mass:.3e}_{c_tau:.3e}_{mixing_str}_data.dat'
            )
        else:
            outputfileName = os.path.join(
                eventData_dir, 
                f'{LLP_name}_{mass:.3e}_{c_tau:.3e}_data.dat'
            )
    elif LLP_name == "Dark-photons":
        if uncertainty is not None:
            outputfileName = os.path.join(
                eventData_dir, 
                f'{LLP_name}_{mass:.3e}_{c_tau:.3e}_{uncertainty}_data.dat'
            )
        else:
            outputfileName = os.path.join(
                eventData_dir, 
                f'{LLP_name}_{mass:.3e}_{c_tau:.3e}_data.dat'
            )
    else:
        outputfileName = os.path.join(
            eventData_dir, 
            f'{LLP_name}_{mass:.3e}_{c_tau:.3e}_data.dat'
        )
    
    # Conditional check: Only write data.dat if ifExportEvents is True and N_ev_tot >= 0.1
    if ifExportEvents and N_ev_tot >= 0.1:
        # Write the results to the data.dat file
        with open(outputfileName, 'w') as f:
            # Write the header line with relevant information
            header = (
                f"Sampled {finalEvents:.6e} events inside SHiP volume. "
                f"Squared coupling: {coupling_squared:.6e}. "
                f"Total number of produced LLPs: {N_LLP_tot:.6e}. "
                f"Polar acceptance: {epsilon_polar:.6e}. "
                f"Azimuthal acceptance: {epsilon_azimuthal:.6e}. "
                f"Averaged decay probability: {P_decay_averaged:.6e}. "
                f"Visible Br Ratio: {br_visible_val:.6e}. " 
                f"Total number of events: {N_ev_tot:.6e}\n\n"
            )
            f.write(header)
        
            start_row = 0  # Initialize the starting row index
            for idx_in_selected, i in enumerate(selected_decay_indices):
                channel = decayChannels[i]
                channel_size = size_per_channel[idx_in_selected]
        
                # Skip channels with size 0
                if channel_size == 0:
                    continue
        
                end_row = start_row + channel_size
                data = results[start_row:end_row, :]  # Extract the relevant rows
                
                # Write the header for the channel
                channel_header = f"#<process={channel}; sample_points={channel_size}>\n\n"
                f.write(channel_header)
                
                # Write the data with formatted numbers
                for row in data:
                    row_str = ' '.join("{:.6e}".format(x) for x in row)
                    f.write(f"{row_str}\n")
                
                # Add a blank line between channels
                f.write("\n")
                
                # Update the starting row index for the next channel
                start_row = end_row
    else:
        if not ifExportEvents:
            print("Exporting events is disabled. The events file has not been recorded.")
        else:
            print("The total number of events < 0.1. The events file has not been recorded.")
    
    # Construct the total file name based on LLP_name and parameters
    if LLP_name == "HNL":
        if MixingPatternArray is not None and isinstance(MixingPatternArray, np.ndarray):
            mixing_str = '_'.join([f"{mp:.3e}" for mp in MixingPatternArray])
            total_filename = f"{LLP_name}_{mixing_str}_total.txt"
        else:
            total_filename = f"{LLP_name}_total.txt"
    elif LLP_name == "Dark-photons":
        if uncertainty is not None:
            total_filename = f"{LLP_name}_{uncertainty}_total.txt"
        else:
            total_filename = f"{LLP_name}_total.txt"
    elif "Scalar" in LLP_name:
        total_filename = f"{LLP_name}_total.txt"
    else:
        total_filename = f"{LLP_name}_total.txt"  # Default case
    
    total_file_path = os.path.join(total_dir, total_filename)
    
    # Check if the total file exists; if not, create it and write the header
    if not os.path.exists(total_file_path):
        with open(total_file_path, 'w') as total_file:
            # Write the header line with formatted column names
            total_file.write(
                'mass coupling_squared c_tau N_LLP_tot epsilon_polar '
                'epsilon_azimuthal P_decay_averaged Br_visible N_ev_tot\n'
            )
    
    # Append the data to the total file with formatted numbers
    with open(total_file_path, 'a') as total_file:
        # Prepare the data string with each number formatted
        data_values = [
            mass, 
            coupling_squared, 
            c_tau, 
            N_LLP_tot, 
            epsilon_polar, 
            epsilon_azimuthal, 
            P_decay_averaged, 
            br_visible_val, 
            N_ev_tot
        ]
        data_string = ' '.join("{:.6e}".format(x) for x in data_values) + "\n"
        total_file.write(data_string)

