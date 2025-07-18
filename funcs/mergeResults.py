# mergeResults.py
import os
import numpy as np
import pandas as pd

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
    model,
    ifExportEvents
):
    """
    Saves simulation results to data files.

    NOTE: The redundant check for N_ev_tot >= 2 is removed. 
    The threshold is fully handled in simulate.py.
    """

    results = np.concatenate((motherParticleResults, decayProductsResults), axis=1)
    df_results = pd.DataFrame(results)

    base_output_dir = os.path.join('.', 'outputs', LLP_name)
    os.makedirs(base_output_dir, exist_ok=True)
    
    eventData_dir = os.path.join(base_output_dir, 'eventData')
    total_dir = os.path.join(base_output_dir, 'total')
    os.makedirs(eventData_dir, exist_ok=True)
    os.makedirs(total_dir, exist_ok=True)
    
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
    elif LLP_name == "Inelastic-DM":
        if uncertainty is not None and model is not None:
            outputfileName = os.path.join(
                eventData_dir,
                f'{LLP_name}_{mass:.3e}_{c_tau:.3e}_{uncertainty}_Model{model}_data.dat'
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
    
    # Since we already checked N_ev_tot < min_events_threshold in simulate.py, 
    # here we only check ifExportEvents.
    if ifExportEvents:
        with open(outputfileName, 'w') as f:
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

            start_row = 0
            for idx_in_selected, i in enumerate(selected_decay_indices):
                channel = decayChannels[i]
                channel_size = size_per_channel[idx_in_selected]
                if channel_size == 0:
                    continue

                end_row = start_row + channel_size
                channel_data = df_results.iloc[start_row:end_row]
                channel_header = f"#<process={channel}; sample_points={channel_size}>\n\n"
                f.write(channel_header)
                
                data_str = channel_data.to_csv(sep=' ', index=False, header=False)
                data_str = data_str.rstrip('\n')
                f.write(data_str)
                f.write("\n\n")

                start_row = end_row
    else:
        print("Exporting events is disabled. The events file has not been recorded.")
    
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
    elif LLP_name == "Inelastic-DM":
        if uncertainty is not None:
            total_filename = f"{LLP_name}_{uncertainty}_Model{model}_total.txt"
    elif "Scalar" in LLP_name:
        total_filename = f"{LLP_name}_total.txt"
    else:
        total_filename = f"{LLP_name}_total.txt"
    
    total_file_path = os.path.join(total_dir, total_filename)
    
    if not os.path.exists(total_file_path):
        with open(total_file_path, 'w') as total_file:
            total_file.write(
                'mass coupling_squared c_tau N_LLP_tot epsilon_polar '
                'epsilon_azimuthal P_decay_averaged Br_visible N_ev_tot\n'
            )
    
    with open(total_file_path, 'a') as total_file:
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
        data_string = ' '.join("{:.9e}".format(x) for x in data_values) + "\n"
        total_file.write(data_string)


def save_total_only(
    LLP_name, 
    mass,
    coupling_squared,
    c_tau,
    N_LLP_tot,
    epsilon_polar,
    epsilon_azimuthal,
    P_decay_averaged,
    br_visible_val,
    N_ev_tot,
    uncertainty,
    model,
    MixingPatternArray,
    decayChannels
):
    base_output_dir = os.path.join('.', 'outputs', LLP_name)
    os.makedirs(base_output_dir, exist_ok=True)
    total_dir = os.path.join(base_output_dir, 'total')
    os.makedirs(total_dir, exist_ok=True)
    
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
    elif LLP_name == "Inelastic-DM":
        if uncertainty is not None and model is not None:
            total_filename = f"{LLP_name}_{uncertainty}_Model{model}_total.txt"
    elif "Scalar" in LLP_name:
        total_filename = f"{LLP_name}_total.txt"
    else:
        total_filename = f"{LLP_name}_total.txt"
    
    total_file_path = os.path.join(total_dir, total_filename)
    
    if not os.path.exists(total_file_path):
        with open(total_file_path, 'w') as total_file:
            total_file.write(
                'mass coupling_squared c_tau N_LLP_tot epsilon_polar '
                'epsilon_azimuthal P_decay_averaged Br_visible N_ev_tot\n'
            )
    
    with open(total_file_path, 'a') as total_file:
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
        data_string = ' '.join("{:.9e}".format(x) for x in data_values) + "\n"
        total_file.write(data_string)

