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

def extract_mixing_patterns(total_dir):
    """
    Extracts mixing patterns from filenames in the 'total' directory for HNL.
    Returns a list of tuples (filename, [XX, YY, ZZ]).
    """
    pattern = re.compile(r"HNL_([\d\.\+eE-]+)_([\d\.\+eE-]+)_([\d\.\+eE-]+)_total\.txt$")
    mixing_files = []
    
    try:
        files = os.listdir(total_dir)
    except FileNotFoundError:
        print(f"Error: The directory '{total_dir}' does not exist.")
        sys.exit(1)
    
    for file in files:
        match = pattern.match(file)
        if match:
            xx, yy, zz = match.groups()
            try:
                xx_float = float(xx)
                yy_float = float(yy)
                zz_float = float(zz)
                mixing_pattern = f"[{xx_float:.2f}, {yy_float:.2f}, {zz_float:.2f}]"
                mixing_files.append((file, mixing_pattern))
            except ValueError:
                print(f"Warning: Unable to parse mixing pattern from file '{file}'. Skipping.")
    
    return mixing_files

def plot_acceptances(data, save_path, title):
    data_sorted = data.sort_values(by='mass').reset_index(drop=True)
    try:
        mass = data_sorted['mass'].to_numpy()
        epsilon_polar = data_sorted['epsilon_polar'].to_numpy()
        epsilon_geom = epsilon_polar * data_sorted['epsilon_azimuthal'].to_numpy()
        epsilon_geom_P_decay = data_sorted['c_tau'].to_numpy() * epsilon_geom * data_sorted['P_decay_averaged'].to_numpy()
    except KeyError as e:
        print(f"Error: Missing expected column: {e}")
        sys.exit(1)
    
    plt.figure(figsize=(10, 7))
    plt.plot(mass, epsilon_polar, label=r'$\epsilon_{\mathrm{polar}}$', linewidth=2, marker='o')
    plt.plot(mass, epsilon_geom, label=r'$\epsilon_{\mathrm{geom}}$', linewidth=2, marker='s')
    plt.plot(mass, epsilon_geom_P_decay, label=r'$c\tau \langle \epsilon_{\mathrm{geom}} \cdot P_{\mathrm{decay}} \rangle$', linewidth=2, marker='^')
    plt.xlabel(r'$m_{\mathrm{LLP}}$ [GeV]', fontsize=14)
    plt.ylabel('Fraction', fontsize=14)
    plt.yscale('log')
    plt.title(title, fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_coupling_vs_events(data, save_path):
    unique_masses = data['mass'].unique()
    plt.figure(figsize=(10, 7))
    for mass in unique_masses:
        subset = data[data['mass'] == mass]
        plt.loglog(subset['coupling_squared'], subset['N_ev_tot'], marker='o', label=rf'$m_{{\mathrm{{LLP}}}} = {mass} \ \mathrm{{GeV}}$')
    
    plt.xlabel(r'$\mathrm{coupling}^2$', fontsize=14)
    plt.ylabel(r'$N_{\mathrm{events}}$', fontsize=14)
    plt.title(r'$N_{\mathrm{events}}$ for different masses', fontsize=16)
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
    if selected_llp != "HNL":
        data_file_path = os.path.join(outputs_dir, selected_llp, 'total', f"{selected_llp}_total.txt")
    else:
        total_dir = os.path.join(outputs_dir, selected_llp, 'total')
        mixing_files = extract_mixing_patterns(total_dir)
        selected_file = select_option([f for f, _ in mixing_files], "Select the mixing pattern file:")
        data_file_path = os.path.join(total_dir, selected_file)
    
    data = pd.read_csv(data_file_path, delim_whitespace=True, header=0)
    required_columns = ['mass', 'coupling_squared', 'c_tau', 'N_LLP_tot', 'epsilon_polar', 'epsilon_azimuthal', 'P_decay_averaged', 'Br_visible', 'N_ev_tot']
    if not all(col in data.columns for col in required_columns):
        print("Error: Missing columns in data file.")
        sys.exit(1)
    
    plot_dir = os.path.join(basedir, 'plots', selected_llp)
    os.makedirs(plot_dir, exist_ok=True)
    acceptance_plot_path = os.path.join(plot_dir, 'acceptance_plot.png')
    plot_acceptances(data, acceptance_plot_path, f"{selected_llp} Acceptances")
    
    coupling_vs_events_plot_path = os.path.join(plot_dir, 'coupling_vs_events.pdf')
    plot_coupling_vs_events(data, coupling_vs_events_plot_path)

if __name__ == "__main__":
    main()

