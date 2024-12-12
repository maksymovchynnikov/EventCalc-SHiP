# total-plots.py
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import re

def scan_outputs_folders(outputs_dir):
    """
    Scans the 'outputs' directory for non-empty folders and returns their names.
    """
    try:
        folders = [f for f in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, f))]
        non_empty_folders = []
        for folder in folders:
            folder_path = os.path.join(outputs_dir, folder)
            if any(os.scandir(folder_path)):
                non_empty_folders.append(folder)
        return non_empty_folders
    except FileNotFoundError:
        print(f"Error: The directory '{outputs_dir}' does not exist.")
        sys.exit(1)

def select_option(options, prompt):
    """
    Displays a list of options to the user and prompts for a selection.
    Returns the selected option.
    """
    if not options:
        print("No options available for selection.")
        sys.exit(1)
    
    print(prompt)
    for idx, option in enumerate(options, start=1):
        print(f"{idx}. {option}")
    
    while True:
        try:
            choice = int(input("Enter the number corresponding to your choice: "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                print(f"Please enter a number between 1 and {len(options)}.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def extract_files(total_dir, selected_llp):
    """
    Extracts relevant files from the 'total' directory based on LLP type.
    For HNL: HNL_mix1_mix2_mix3_total.txt
    For Dark-photons: Dark-photons_uncertainty_total.txt
    For Other LLPs: LLP_name_total.txt
    Returns a list of tuples (filename, identifier)
    """
    try:
        files = os.listdir(total_dir)
    except FileNotFoundError:
        print(f"Error: The directory '{total_dir}' does not exist.")
        sys.exit(1)
    
    extracted_files = []
    
    if selected_llp == "HNL":
        # Pattern: HNL_mix1_mix2_mix3_total.txt
        pattern = re.compile(r"HNL_([\d\.\+eE-]+)_([\d\.\+eE-]+)_([\d\.\+eE-]+)_total\.txt$")
        for file in files:
            match = pattern.match(file)
            if match:
                mix1 = float(match.group(1))
                mix2 = float(match.group(2))
                mix3 = float(match.group(3))
                # Format mixing pattern without scientific notation, truncated to three decimals
                identifier = f"[{mix1:.3f}, {mix2:.3f}, {mix3:.3f}]"
                extracted_files.append((file, identifier))
    elif selected_llp == "Dark-photons":
        # Pattern: Dark-photons_uncertainty_total.txt
        pattern = re.compile(r"Dark-photons_(lower|central|upper)_total\.txt$")
        for file in files:
            match = pattern.match(file)
            if match:
                uncertainty = match.group(1)
                identifier = f"uncertainty={uncertainty}"
                extracted_files.append((file, identifier))
    else:
        # For other LLPs, Pattern: LLP_name_total.txt
        pattern = re.compile(rf"{re.escape(selected_llp)}_total\.txt$")
        for file in files:
            match = pattern.match(file)
            if match:
                identifier = ""  # No additional identifier
                extracted_files.append((file, identifier))
    
    return extracted_files

def plot_acceptances(data, save_path, title, selected_lifetimes, selected_styles, selected_labels):
    """
    Plots epsilon_polar, epsilon_geom, and ctau*epsilon_geom*P_decay for selected lifetimes.
    """
    plt.figure(figsize=(10, 7))
    
    for lifetime, style, label in zip(selected_lifetimes, selected_styles, selected_labels):
        subset = data[data['c_tau'] == lifetime].sort_values(by='mass').reset_index(drop=True)
        mass = subset['mass'].to_numpy()
        epsilon_polar = subset['epsilon_polar'].to_numpy()
        epsilon_geom = epsilon_polar * subset['epsilon_azimuthal'].to_numpy()
        epsilon_geom_P_decay = subset['c_tau'].to_numpy() * epsilon_geom * subset['P_decay_averaged'].to_numpy()
        
        plt.plot(mass, epsilon_polar, linestyle=style, label=f'ε_polar ({label})', linewidth=2)
        plt.plot(mass, epsilon_geom, linestyle=style, label=f'ε_geom ({label})', linewidth=2)
        plt.plot(mass, epsilon_geom_P_decay, linestyle=style, label=f'cτ⋅ε_geom⋅⟨P_decay⟩ ({label})', linewidth=2)
    
    plt.xlabel(r'$m_{\mathrm{LLP}}$ [GeV]', fontsize=14)
    plt.ylabel('Fraction', fontsize=14)
    plt.yscale('log')
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_coupling_vs_events(data, save_path, title):
    """
    Plots N_events vs coupling_squared for different masses.
    """
    unique_masses = sorted(data['mass'].unique())
    plt.figure(figsize=(10, 7))
    for mass in unique_masses:
        subset = data[data['mass'] == mass].sort_values(by='coupling_squared').reset_index(drop=True)
        plt.loglog(subset['coupling_squared'], subset['N_ev_tot'], marker='o', label=rf'$m_{{\mathrm{{LLP}}}} = {mass:.3f}$ GeV')
    
    plt.xlabel(r'$\mathrm{coupling}^{2}$', fontsize=14)
    plt.ylabel(r'$N_{\mathrm{events}}$', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.show()

def plot_lifetime_vs_events(data, save_path, title):
    """
    Plots N_events vs c_tau for different masses.
    """
    unique_masses = sorted(data['mass'].unique())
    plt.figure(figsize=(10, 7))
    for mass in unique_masses:
        subset = data[data['mass'] == mass].sort_values(by='c_tau').reset_index(drop=True)
        plt.loglog(subset['c_tau'], subset['N_ev_tot'], marker='o', label=rf'$m_{{\mathrm{{LLP}}}} = {mass:.3f}$ GeV')
    
    plt.xlabel(r'$c\tau_{\mathrm{LLP}}$ [m]', fontsize=14)
    plt.ylabel(r'$N_{\mathrm{events}}$', fontsize=14)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, format='pdf')
    plt.show()

def main():
    basedir = os.getcwd()
    outputs_dir = os.path.join(basedir, 'outputs')
    llp_folders = scan_outputs_folders(outputs_dir)
    selected_llp = select_option(llp_folders, "Select the LLP:")
    
    total_dir = os.path.join(outputs_dir, selected_llp, 'total')
    extracted_files = extract_files(total_dir, selected_llp)
    
    if selected_llp == "HNL":
        if not extracted_files:
            print("No mixing pattern files found for HNL.")
            sys.exit(1)
        print(f"Available HNL files:")
        for i, (_, identifier) in enumerate(extracted_files, start=1):
            print(f"{i}. Mixing pattern={identifier}")
        # Ask user to choose a file
        while True:
            try:
                choice = int(input("Choose a file by typing the number: "))
                if 1 <= choice <= len(extracted_files):
                    break
                else:
                    print(f"Please enter a number between 1 and {len(extracted_files)}.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        selected_file, identifier = extracted_files[choice - 1]
        selected_filepath = os.path.join(total_dir, selected_file)
        mix_label = identifier  # Contains mixing pattern
    elif selected_llp == "Dark-photons":
        if not extracted_files:
            print("No uncertainty choice files found for Dark-photons.")
            sys.exit(1)
        print(f"Available Dark-photons files:")
        for i, (_, identifier) in enumerate(extracted_files, start=1):
            print(f"{i}. {identifier}")
        # Ask user to choose a file
        while True:
            try:
                choice = int(input("Choose an uncertainty choice file by typing the number: "))
                if 1 <= choice <= len(extracted_files):
                    break
                else:
                    print(f"Please enter a number between 1 and {len(extracted_files)}.")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        selected_file, identifier = extracted_files[choice - 1]

        selected_filepath = os.path.join(total_dir, selected_file)
        mix_label = identifier  # Contains uncertainty
    else:
        # For other LLPs, select the first file
        if not extracted_files:
            print(f"No total files found for LLP '{selected_llp}'.")
            sys.exit(1)
        selected_file, identifier = extracted_files[0]
        selected_filepath = os.path.join(total_dir, selected_file)
        mix_label = ""  # No additional identifier
    
    print(f"\nSelected file: {selected_filepath}\n")
    
    # Read the selected data file
    try:
        data = pd.read_csv(selected_filepath, delim_whitespace=True, header=0)
    except Exception as e:
        print(f"Error reading file {selected_filepath}: {e}")
        sys.exit(1)
    
    required_columns = ['mass', 'coupling_squared', 'c_tau', 'N_LLP_tot', 
                        'epsilon_polar', 'epsilon_azimuthal', 
                        'P_decay_averaged', 'Br_visible', 'N_ev_tot']
    if not all(col in data.columns for col in required_columns):
        print("Error: Missing columns in data file.")
        sys.exit(1)
    
    # For acceptances plot, ask user to select up to 2 different lifetimes
    unique_lifetimes = sorted(data['c_tau'].unique())
    print("Plotting averaged quantities as a function of mass for fixed lifetimes.")
    print("Select up to 2 lifetimes by typing their numbers separated by space (e.g., 1 2):")
    for i, lifetime in enumerate(unique_lifetimes, start=1):
        print(f"{i}. {lifetime:.2f} m")
    
    selected_lifetimes = []
    selected_styles = []
    selected_labels = []
    
    max_selection = min(2, len(unique_lifetimes))
    if max_selection == 0:
        print("No lifetimes available for plotting.")
        sys.exit(1)
    
    while True:
        try:
            selection = input("Select up to 2 lifetimes by typing their numbers separated by space (e.g., 1 2): ")
            choices = list(map(int, selection.strip().split()))
            if 1 <= len(choices) <= max_selection and all(1 <= choice <= len(unique_lifetimes) for choice in choices):
                selected_lifetimes = [unique_lifetimes[choice - 1] for choice in choices]
                break
            else:
                print(f"Please enter between 1 and {max_selection} valid numbers.")
        except ValueError:
            print("Invalid input. Please enter valid numbers separated by space.")
    
    # Assign styles and labels
    styles = ['solid', 'dashed']
    for idx, lifetime in enumerate(selected_lifetimes):
        selected_styles.append(styles[idx % len(styles)])
        if selected_llp == "HNL":
            # Label with mixing pattern
            selected_labels.append(f"{mix_label}")
        elif selected_llp == "Dark-photons":
            # Label with uncertainty choice
            selected_labels.append(f"{mix_label}")
        else:
            # Label with LLP name only
            selected_labels.append(f"{selected_llp}")
    
    # Prepare plot directory
    plot_dir = os.path.join(basedir, 'plots', selected_llp)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Plot acceptances
    acceptance_plot_path = os.path.join(plot_dir, 'acceptance_plot.png')
    if selected_llp in ["HNL", "Dark-photons"]:
        plot_title = f"{selected_llp} Acceptances ({mix_label})"
    else:
        plot_title = f"{selected_llp} Acceptances"
    plot_acceptances(data, acceptance_plot_path, plot_title, selected_lifetimes, selected_styles, selected_labels)
    
    # Plot coupling vs events
    coupling_vs_events_plot_path = os.path.join(plot_dir, 'coupling_vs_events.pdf')
    if selected_llp in ["HNL", "Dark-photons"]:
        coupling_plot_title = rf"$N_{{\mathrm{{events}}}}$ for {selected_llp} ({mix_label}) as a function of $\mathrm{{coupling}}^{{2}}$"
    else:
        coupling_plot_title = rf"$N_{{\mathrm{{events}}}}$ for {selected_llp} as a function of $\mathrm{{coupling}}^{{2}}$"
    plot_coupling_vs_events(data, coupling_vs_events_plot_path, coupling_plot_title)
    
    # Plot lifetime vs events
    lifetime_vs_events_plot_path = os.path.join(plot_dir, 'lifetime_vs_events.pdf')
    if selected_llp in ["HNL", "Dark-photons"]:
        lifetime_plot_title = rf"$N_{{\mathrm{{events}}}}$ for {selected_llp} ({mix_label}) as a function of $c\tau_{{\mathrm{{LLP}}}}$"
    else:
        lifetime_plot_title = rf"$N_{{\mathrm{{events}}}}$ for {selected_llp} as a function of $c\tau_{{\mathrm{{LLP}}}}$"
    plot_lifetime_vs_events(data, lifetime_vs_events_plot_path, lifetime_plot_title)

if __name__ == "__main__":
    main()


