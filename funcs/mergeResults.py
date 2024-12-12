# mergeResults.py
import os
import numpy as np

def save(motherParticleResults, decayProductsResults, LLP_name, mass, MixingPatternArray, c_tau, decayChannels, size_per_channel, finalEvents, epsilon_polar, epsilon_azimuthal, N_LLP_tot, coupling_squared, P_decay_averaged, N_ev_tot, br_visible_val, selected_decay_indices, uncertainty):
    # Format of results
    # px_mother, py_mother, pz_mother, energy_mother, m_mother, PDG_mother, P_decay_mother, x_mother, y_mother, z_mother, px1, py1, pz1, e1, MASS1, pdg1, charge1, stability1, px2, py2, pz2, e2, MASS2, pdg2, charge2, stability2, px3, py3, pz3, e3, MASS3, pdg3, charge3, stability3
    results = np.concatenate((motherParticleResults, decayProductsResults), axis=1)
    
    # Create base output directory
    base_output_dir = os.path.join('./outputs', LLP_name)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Create subdirectories for event data and total files
    eventData_dir = os.path.join(base_output_dir, 'eventData')
    total_dir = os.path.join(base_output_dir, 'total')
    os.makedirs(eventData_dir, exist_ok=True)
    os.makedirs(total_dir, exist_ok=True)
    
    # Create the output file name for data.dat
    if LLP_name == "HNL":
        if MixingPatternArray is not None and isinstance(MixingPatternArray, np.ndarray):
            outputfileName = os.path.join(
                eventData_dir, 
                f'{LLP_name}_{mass:.3e}_{c_tau:.3e}_' +
                f'{MixingPatternArray[0]:.3e}_{MixingPatternArray[1]:.3e}_{MixingPatternArray[2]:.3e}_data.dat'
            )
        else:
            outputfileName = os.path.join(eventData_dir, f'{LLP_name}_{mass:.3e}_{c_tau:.3e}_data.dat')
    elif LLP_name == "Dark-photons":
        if uncertainty is not None:
            outputfileName = os.path.join(
                eventData_dir, 
                f'{LLP_name}_{mass:.3e}_{c_tau:.3e}_{uncertainty}_data.dat'
            )
        else:
            outputfileName = os.path.join(eventData_dir, f'{LLP_name}_{mass:.3e}_{c_tau:.3e}_data.dat')
    else:
        outputfileName = os.path.join(eventData_dir, f'{LLP_name}_{mass:.3e}_{c_tau:.3e}_data.dat')
    
    # Write the results to the data.dat file
    with open(outputfileName, 'w') as f:
        # Write the new header line with Squared coupling
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
    
    # Construct the total file name based on LLP_name
    if LLP_name == "HNL":
        if MixingPatternArray is not None and isinstance(MixingPatternArray, np.ndarray):
            total_filename = f"{LLP_name}_{MixingPatternArray[0]:.3e}_{MixingPatternArray[1]:.3e}_{MixingPatternArray[2]:.3e}_total.txt"
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
            total_file.write('mass coupling_squared c_tau N_LLP_tot epsilon_polar epsilon_azimuthal P_decay_averaged Br_visible N_ev_tot\n')
    
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

