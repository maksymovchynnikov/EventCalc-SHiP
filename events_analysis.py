import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from funcs.ship_setup import plot_decay_volume, z_min, z_max, y_max, x_max  # Ensure these are correctly defined
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from funcs.selecting_processing import parse_filenames, user_selection, read_file  # Ensure these are correctly defined

def plot_channels(channels, finalEvents, output_path, llp_name, mass, lifetime):
    """
    Plots histogram for channels and adds LLP information text.
    """
    channel_names = list(channels.keys())
    channel_sizes = [channels[ch]['size'] for ch in channel_names]

    plt.figure(figsize=(12, 7))
    plt.bar(channel_names, channel_sizes, color='skyblue', edgecolor='black')
    plt.title(f"$N_{{\\mathrm{{entries}}}}$ = {finalEvents:.0f}", fontsize=14)
    plt.xlabel("Channel", fontsize=12)
    plt.ylabel("Number of events", fontsize=12)
    plt.xticks(rotation=45, ha='right')

    # Add LLP information text in the top right corner
    textstr = f"llp: {llp_name}\nmass: {mass} GeV\nlifetime: {lifetime} s"
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="white", 
                       edgecolor="black", 
                       alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "channels.pdf"), bbox_inches='tight')
    plt.close()

def extract_quantities(channels):
    """
    Extracts required quantities from the data lines.
    Returns a dictionary with the extracted quantities.
    """
    # Initialize dictionaries for quantities
    quantities = {
        'px_mother': [],
        'py_mother': [],
        'pz_mother': [],
        'energy_mother': [],
        'm_mother': [],
        'PDG_mother': [],
        'P_decay_mother': [],
        'x_mother': [],
        'y_mother': [],
        'z_mother': [],
        'decay_products_counts': [],  # Total decay products per event
        'charged_decay_products_counts': [],  # Charged decay products per event
        'nu_counts': [],  # Neutrino counts per event
        'decay_products_per_event_counts': defaultdict(list),  # Counts of dedicated products per event
        'ifAllPoint_counts': defaultdict(float),  # Weighted counts of events where all decay products point to detectors per channel
        'sum_P_decay_mother_per_channel': defaultdict(float),  # Sum of P_decay_mother per channel
        'ifAllPoint_ratios': {},  # Ratio per channel
        'final_states_per_channel': defaultdict(lambda: defaultdict(int)),  # Final states classifier per channel
    }

    # Define specific decay product types for detailed counting (particles and antiparticles separately)
    detailed_final_state_particles = [
        'e-', 'e+', 'mu-', 'mu+', 'pi-', 'pi+', 'K-', 'K+', 'K_L',
        'p', 'bar[p]', 'n', 'bar[n]', 'nu', 'gamma'
    ]

    # Define aggregated categories for multiplicity plots
    aggregated_final_state_categories = {
        'e': ['e-', 'e+'],
        'mu': ['mu-', 'mu+'],
        'pi': ['pi-', 'pi+'],
        'kcharged': ['K-', 'K+'],
        'k_l': ['K_L'],
        'p': ['p', 'bar[p]'],
        'n': ['n', 'bar[n]'],
        'nu': ['nu'],
        'gamma': ['gamma']
    }

    # Pre-define PDG codes mapping to final state particles (distinguish particles and antiparticles)
    pdg_to_particle = {
        22: 'gamma',
        11: 'e-',    # Electron
        -11: 'e+',   # Positron
        13: 'mu-',    # Muon-
        -13: 'mu+',   # Muon+
        211: 'pi+',   # Pion+
        -211: 'pi-',   # Pion-
        321: 'K+',    # Kaon+
        -321: 'K-',    # Kaon-
        130: 'K_L',
        2212: 'p',     # Proton
        -2212: 'bar[p]', # Anti-Proton
        2112: 'n',     # Neutron
        -2112: 'bar[n]', # Anti-Neutron
        12: 'nu',      # Neutrino
        -12: 'nu',     # Anti-Neutrino
        14: 'nu',
        -14: 'nu',
        16: 'nu',
        -16: 'nu'
    }

    # Temporary Debugging: Print channel and event information
    for channel_idx, (channel, channel_data) in enumerate(channels.items(), start=1):
        print(f"Channel {channel_idx}: {channel}")
        data_lines = channel_data['data']
        for event_idx, data_line in enumerate(data_lines, start=1):
            # Split the data line into numbers
            numbers = list(map(float, data_line.strip().split()))
            # First 10 numbers are px_mother, py_mother, pz_mother, energy_mother, m_mother, PDG_mother, P_decay_mother, x_mother, y_mother, z_mother
            if len(numbers) < 10:
                print(f"Error: Data line has less than 10 numbers: {data_line}")
                continue
            px_mother = numbers[0]
            py_mother = numbers[1]
            pz_mother = numbers[2]
            energy_mother = numbers[3]
            m_mother = numbers[4]
            PDG_mother = int(numbers[5])
            P_decay_mother = numbers[6]
            x_mother = numbers[7]
            y_mother = numbers[8]
            z_mother = numbers[9]

            # Extract PDG codes for decay products
            decay_products = numbers[10:]
            num_decay_products = len(decay_products) // 6
            pdg_list = [int(decay_products[6*i + 5]) for i in range(num_decay_products)]

            # Temporary Debugging: Print event PDG list
            print(f"  Event {event_idx}: pdg list = {pdg_list}")

            # Append mother particle information
            quantities['px_mother'].append(px_mother)
            quantities['py_mother'].append(py_mother)
            quantities['pz_mother'].append(pz_mother)
            quantities['energy_mother'].append(energy_mother)
            quantities['m_mother'].append(m_mother)
            quantities['PDG_mother'].append(PDG_mother)
            quantities['P_decay_mother'].append(P_decay_mother)
            quantities['x_mother'].append(x_mother)
            quantities['y_mother'].append(y_mother)
            quantities['z_mother'].append(z_mother)

            # Accumulate the sum of P_decay_mother for this channel
            quantities['sum_P_decay_mother_per_channel'][channel] += P_decay_mother

            # Initialize per-event counts
            decay_products_count = 0
            charged_decay_products_count = 0
            nu_count = 0
            final_state_counts = {ptype: 0 for ptype in detailed_final_state_particles}

            # Flag to check if all decay products point to the detector
            all_point = True

            # Process each decay product
            for i in range(num_decay_products):
                base_idx = i * 6
                px = decay_products[base_idx]
                py = decay_products[base_idx + 1]
                pz = decay_products[base_idx + 2]
                e = decay_products[base_idx + 3]
                mass = decay_products[base_idx + 4]
                pdg = int(decay_products[6*i + 5])

                # Check for pdg = -999.0
                if pdg == -999:
                    continue

                # Check for neutrinos
                if pdg in [12, -12, 14, -14, 16, -16]:
                    nu_count +=1
                    final_state_counts['nu'] +=1  # Include neutrinos in final_state_counts
                    continue  # Neutrinos are not counted as decay products or charged decay products

                # Map PDG to particle
                particle = pdg_to_particle.get(pdg, 'other')

                # Update final_state_counts
                if particle in final_state_counts:
                    final_state_counts[particle] +=1
                else:
                    # Ignore 'other' or undefined particles
                    pass

                # Update decay_products_count
                decay_products_count += 1

                # Update charged_decay_products_count
                if particle not in ['gamma', 'K_L', 'n', 'bar[n]', 'nu']:
                    charged_decay_products_count += 1

                # Calculate projections to z_max m plane
                if pz == 0:
                    # Avoid division by zero; set flag and continue
                    all_point = False
                    continue  # Do not break; process remaining decay products

                x_proj = x_mother + (z_max - z_mother) * px / pz
                y_proj = y_mother + (z_max - z_mother) * py / pz

                if not (-y_max(z_max) < y_proj < y_max(z_max) and -x_max(z_max) < x_proj < x_max(z_max)):
                    all_point = False
                    # Do not break; process remaining decay products
                    continue

            # Append counts to lists
            quantities['decay_products_counts'].append(decay_products_count)
            quantities['charged_decay_products_counts'].append(charged_decay_products_count)
            quantities['nu_counts'].append(nu_count)

            # Append per-event counts for each decay product type
            for ptype in detailed_final_state_particles:
                quantities['decay_products_per_event_counts'][ptype].append(final_state_counts[ptype])

            # Update final_states_per_channel with the final state tuple
            # Only include non-zero counts to reduce file size
            state_tuple = tuple(final_state_counts[ptype] for ptype in detailed_final_state_particles)
            quantities['final_states_per_channel'][channel][state_tuple] += 1

            # Update ifAllPoint_counts with weighted count
            if all_point:
                quantities['ifAllPoint_counts'][channel] += P_decay_mother

    # After processing all events, compute the ratios
    for channel in channels.keys():
        sum_P_decay = quantities['sum_P_decay_mother_per_channel'][channel]
        if sum_P_decay > 0:
            ratio = quantities['ifAllPoint_counts'][channel] / sum_P_decay
        else:
            ratio = 0
        quantities['ifAllPoint_ratios'][channel] = ratio

    return quantities

def plot_histograms(quantities, channels, output_path, llp_name, mass, lifetime):
    """
    Plots the required histograms and saves them in the output_path directory.
    All histograms are normalized to represent probability densities.
    Adds LLP information text to each plot.
    """
    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Prepare the text string for plots
    textstr = f"llp: {llp_name}\nmass: {mass} GeV\nlifetime: {lifetime} s"

    # Convert relevant quantities to numpy arrays for easier handling
    energy_mother = np.array(quantities['energy_mother'])
    P_decay_mother = np.array(quantities['P_decay_mother'])
    z_mother = np.array(quantities['z_mother'])
    x_mother = np.array(quantities['x_mother'])
    y_mother = np.array(quantities['y_mother'])
    nu_counts = np.array(quantities['nu_counts'])

    # Energy of mother particle (unweighted)
    plt.figure(figsize=(10, 6))
    plt.hist(energy_mother, bins=50, color='skyblue', edgecolor='black', density=True)
    plt.yscale('log')  # Preserving original y-axis scaling
    plt.xlabel("$E_{\\mathrm{LLP}}$ [GeV]", fontsize=14)
    plt.ylabel("probability density", fontsize=14)
    plt.title("llp energy distribution (unweighted)", fontsize=16)
    # Add LLP information text
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="white", 
                       edgecolor="black", 
                       alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "energy_mother_unweighted.pdf"), bbox_inches='tight')
    plt.close()

    # Energy of mother particle (weighted by P_decay_mother)
    plt.figure(figsize=(10, 6))
    plt.hist(energy_mother, bins=50, weights=P_decay_mother, color='salmon', edgecolor='black', density=True)
    plt.yscale('log')  # Preserving original y-axis scaling
    plt.xlabel("$E_{\\mathrm{LLP}}$ [GeV]", fontsize=14)
    plt.ylabel("probability density", fontsize=14)
    plt.title("llp energy distribution (weighted by $P_{\\mathrm{decay}}$)", fontsize=16)
    # Add LLP information text
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="white", 
                       edgecolor="black", 
                       alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "energy_mother_weighted.pdf"), bbox_inches='tight')
    plt.close()

    # P_decay of mother particle
    plt.figure(figsize=(10, 6))
    plt.hist(P_decay_mother, bins=50, color='lightgreen', edgecolor='black', density=True)
    plt.xscale('log')  # Preserving original x-axis scaling
    plt.yscale('log')  # Preserving original y-axis scaling
    plt.xlabel("$P_{\\mathrm{decay,LLP}}$", fontsize=14)
    plt.ylabel("probability density", fontsize=14)
    plt.title("llp decay probability distribution", fontsize=16)
    # Add LLP information text
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="white", 
                       edgecolor="black", 
                       alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "P_decay_mother.pdf"), bbox_inches='tight')
    plt.close()

    # z_mother weighted by P_decay_mother
    plt.figure(figsize=(10, 6))
    plt.hist(z_mother, bins=50, weights=P_decay_mother, color='violet', edgecolor='black', density=True)
    plt.yscale('log')
    plt.xlabel("$z_{\\mathrm{decay,LLP}}$ [m]", fontsize=14)
    plt.ylabel("probability density", fontsize=14)
    plt.title("llp decay positions (weighted by $P_{\\mathrm{decay}}$)", fontsize=16)
    # Add LLP information text
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="white", 
                       edgecolor="black", 
                       alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "z_mother_weighted.pdf"), bbox_inches='tight')
    plt.close()

    # z_mother unweighted
    plt.figure(figsize=(10, 6))
    plt.hist(z_mother, bins=50, color='cyan', edgecolor='black', density=True)
    plt.yscale('log')
    plt.xlabel("$z_{\\mathrm{decay,LLP}}$ [m]", fontsize=14)
    plt.ylabel("probability density", fontsize=14)
    plt.title("llp decay positions (unweighted)", fontsize=16)
    # Add LLP information text
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="white", 
                       edgecolor="black", 
                       alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "decay_positions_unweighted.pdf"), bbox_inches='tight')
    plt.close()

    # Merged histogram of total and charged decay products
    plt.figure(figsize=(10, 6))
    max_count = max(
        max(quantities['decay_products_counts']) if quantities['decay_products_counts'] else 0, 
        max(quantities['charged_decay_products_counts']) if quantities['charged_decay_products_counts'] else 0
    )
    bins = range(1, int(max_count) + 2)
    plt.hist(quantities['decay_products_counts'], bins=bins, alpha=0.5, label='all decay products', color='blue', edgecolor='black', density=True)
    plt.hist(quantities['charged_decay_products_counts'], bins=bins, alpha=0.5, label='charged decay products', color='yellow', edgecolor='black', density=True)
    plt.xlabel("number of decay products", fontsize=14)
    plt.ylabel("probability density", fontsize=14)
    plt.title("decay products multiplicity", fontsize=16)
    plt.legend(fontsize=12)
    # Add LLP information text
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="white", 
                       edgecolor="black", 
                       alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "decay_products_counts_merged.pdf"), bbox_inches='tight')
    plt.close()

    # Histograms of counts per event for each decay product type (combined where necessary)
    decay_products_types = quantities['decay_products_per_event_counts'].keys()
    for ptype in decay_products_types:
        if ptype == 'nu':
            continue  # Neutrinos are handled separately
        counts = quantities['decay_products_per_event_counts'][ptype]
        if counts:
            plt.figure(figsize=(10, 6))
            max_count = max(counts)
            bins = range(0, int(max_count) + 2)  # +2 to include max_count
            plt.hist(counts, bins=bins, align='left', edgecolor='black', color='lightcoral', density=True)
            # Define LaTeX labels
            if ptype == 'e-':
                xlabel = r"$e^{-}$ multiplicity"
            elif ptype == 'e+':
                xlabel = r"$e^{+}$ multiplicity"
            elif ptype == 'mu-':
                xlabel = r"$\mu^{-}$ multiplicity"
            elif ptype == 'mu+':
                xlabel = r"$\mu^{+}$ multiplicity"
            elif ptype == 'pi-':
                xlabel = r"$\pi^{-}$ multiplicity"
            elif ptype == 'pi+':
                xlabel = r"$\pi^{+}$ multiplicity"
            elif ptype == 'k-':
                xlabel = r"$K^{-}$ multiplicity"
            elif ptype == 'k+':
                xlabel = r"$K^{+}$ multiplicity"
            elif ptype == 'K_L':
                xlabel = r"$K_{L}$ multiplicity"
            elif ptype == 'p':
                xlabel = r"$p$ multiplicity"
            elif ptype == 'bar[p]':
                xlabel = r"$\bar{p}$ multiplicity"
            elif ptype == 'n':
                xlabel = r"$n$ multiplicity"
            elif ptype == 'bar[n]':
                xlabel = r"$\bar{n}$ multiplicity"
            elif ptype == 'gamma':
                xlabel = r"$\gamma$ multiplicity"
            else:
                xlabel = f"{ptype} multiplicity"

            plt.xlabel(xlabel, fontsize=14)
            plt.ylabel("probability density", fontsize=14)
            plt.title(f"multiplicity of {ptype} per event", fontsize=16)
            plt.xticks(bins)
            # Add LLP information text
            plt.text(0.95, 0.95, textstr,
                     horizontalalignment='right',
                     verticalalignment='top',
                     transform=plt.gca().transAxes,
                     fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", 
                               facecolor="white", 
                               edgecolor="black", 
                               alpha=0.8))
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"decay_products_counts_{ptype}.pdf"), bbox_inches='tight')
            plt.close()

    # ===========================
    # Neutrino Counts per Event
    # ===========================
    if 'nu' in quantities['decay_products_per_event_counts']:
        nu_counts_list = quantities['nu_counts']
        if len(nu_counts_list) > 0:
            plt.figure(figsize=(10, 6))
            max_nu = max(nu_counts_list)
            bins = range(0, int(max_nu) + 2)  # +2 to include max_nu
            plt.hist(nu_counts_list, bins=bins, align='left', edgecolor='black', color='green', density=True)
            plt.xlabel("number of neutrinos per event", fontsize=14)
            plt.ylabel("probability density", fontsize=14)
            plt.title("neutrino multiplicity per event", fontsize=16)
            plt.xticks(bins)
            # Add LLP information text
            plt.text(0.95, 0.95, textstr,
                     horizontalalignment='right',
                     verticalalignment='top',
                     transform=plt.gca().transAxes,
                     fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", 
                               facecolor="white", 
                               edgecolor="black", 
                               alpha=0.8))
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, "decay_products_counts_nu.pdf"), bbox_inches='tight')
            plt.close()

    # 3D scatter plot of (x_mother, y_mother, z_mother) unweighted
    # Limit to maximum 10k points
    max_points = 10000
    total_points = len(x_mother)
    if total_points > max_points:
        np.random.seed(42)  # For reproducibility
        indices_unw = np.random.choice(total_points, max_points, replace=False)
        x_plot_unw = x_mother[indices_unw]
        y_plot_unw = y_mother[indices_unw]
        z_plot_unw = z_mother[indices_unw]
    else:
        x_plot_unw = x_mother
        y_plot_unw = y_mother
        z_plot_unw = z_mother

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(x_plot_unw, y_plot_unw, z_plot_unw, s=1, alpha=0.5, c='blue')
    
    plot_decay_volume(ax)  # Use ship_setup.py's plot_decay_volume
    
    ax.set_xlabel(r"$x_{\mathrm{mother}}$ [m]", fontsize=12)
    ax.set_ylabel(r"$y_{\mathrm{mother}}$ [m]", fontsize=12)
    ax.set_zlabel(r"$z_{\mathrm{mother}}$ [m]", fontsize=12)
    
    plt.title("decay positions of llp (unweighted)", fontsize=16)
    
    # Add LLP information text
    ax.text2D(0.95, 0.95, textstr,
              horizontalalignment='right',
              verticalalignment='top',
              transform=ax.transAxes,
              fontsize=12,
              bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor="white", 
                        edgecolor="black", 
                        alpha=0.8))
    
    # Create an invisible "dummy" text to add padding to the right
    right_padding = ax.text2D(1.2, 0.5, '', transform=ax.transAxes)  # 1.2 extends right side

    # Save figure with extra space on the right using bbox_extra_artists
    plt.savefig(os.path.join(output_path, "decay_positions_unweighted.pdf"), 
                bbox_inches='tight', 
                bbox_extra_artists=[right_padding], 
                pad_inches=0.5)
    
    plt.close()

    # 3D scatter plot of (x_mother, y_mother, z_mother) weighted by P_decay_mother
    # Select N_entries/10 events using P_decay_mother as weights
    N_selected = len(x_mother) // 10
    max_selected = 10000
    if N_selected > max_selected:
        N_selected = max_selected

    if N_selected > len(x_mother):
        N_selected = len(x_mother)

    if N_selected > 0 and P_decay_mother.sum() > 0:
        # Normalize the decay probabilities
        probabilities = P_decay_mother / P_decay_mother.sum()
        np.random.seed(24)  # Different seed for variety
        try:
            indices_w = np.random.choice(len(x_mother), size=N_selected, replace=False, p=probabilities)
        except ValueError as e:
            print(f"Error during weighted sampling: {e}")
            indices_w = np.random.choice(len(x_mother), size=N_selected, replace=False)

        x_plot_w = x_mother[indices_w]
        y_plot_w = y_mother[indices_w]
        z_plot_w = z_mother[indices_w]
    else:
        x_plot_w = np.array([])
        y_plot_w = np.array([])
        z_plot_w = np.array([])

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    if N_selected > 0:
        ax.scatter(x_plot_w, y_plot_w, z_plot_w, s=1, alpha=0.5, c='red', label=f'selected {N_selected} decays')
    
    plot_decay_volume(ax)  # Use ship_setup.py's plot_decay_volume
    
    ax.set_xlabel(r"$x_{\mathrm{mother}}$ [m]", fontsize=12)
    ax.set_ylabel(r"$y_{\mathrm{mother}}$ [m]", fontsize=12)
    ax.set_zlabel(r"$z_{\mathrm{mother}}$ [m]", fontsize=12)
    
    plt.title("decay positions of llp (weighted by $P_{\\mathrm{decay}}$)", fontsize=16)
    
    # Set z-axis limits
    ax.set_zlim(z_min, z_max + 5)
    
    if N_selected > 0:
        plt.legend(fontsize=12)
    
    # Add LLP information text
    ax.text2D(0.95, 0.95, textstr,
              horizontalalignment='right',
              verticalalignment='top',
              transform=ax.transAxes,
              fontsize=12,
              bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor="white", 
                        edgecolor="black", 
                        alpha=0.8))
    
    # Create an invisible "dummy" text to add padding to the right
    right_padding = ax.text2D(1.2, 0.5, '', transform=ax.transAxes)  # 1.2 means extend right side

    # Save figure with extra space on the right using bbox_extra_artists
    plt.savefig(os.path.join(output_path, "decay_positions_weighted.pdf"), 
                bbox_inches='tight', 
                bbox_extra_artists=[right_padding], 
                pad_inches=0.5)
    plt.close()

    # ===========================
    # Combined 2D Point Plots
    # ===========================
    z_min_decay = z_min + 1  # Adjust as needed
    mask_z = z_mother < z_min_decay

    # Unweighted 2D scatter plot of x and y decay coordinates for z_mother < z_min_decay
    plt.figure(figsize=(10, 8))
    plt.scatter(x_mother[mask_z], y_mother[mask_z], s=1, alpha=0.5, c='blue', label='decays')
    plt.xlabel(r"$x_{\mathrm{mother}}$ [m]", fontsize=14)
    plt.ylabel(r"$y_{\mathrm{mother}}$ [m]", fontsize=14)
    plt.title(f"decay positions (z < {z_min_decay:.0f} m) unweighted", fontsize=16)
    plt.legend(fontsize=12)
    # Add LLP information text
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="white", 
                       edgecolor="black", 
                       alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"decay_positions_xy_unweighted_z_less_{int(z_min_decay)}.pdf"), bbox_inches='tight')
    plt.close()

    # Weighted 2D scatter plot of x and y decay coordinates for z_mother < z_min_decay
    # Select N_entries/10 events using P_decay_mother as weights within the mask
    x_masked = x_mother[mask_z]
    y_masked = y_mother[mask_z]
    P_decay_masked = P_decay_mother[mask_z]
    total_masked = len(x_masked)
    N_selected_xy = total_masked // 10
    if N_selected_xy > max_selected:
        N_selected_xy = max_selected
    if N_selected_xy > total_masked:
        N_selected_xy = total_masked

    if N_selected_xy > 0 and P_decay_masked.sum() > 0:
        probabilities_xy = P_decay_masked / P_decay_masked.sum()
        np.random.seed(100)  # Different seed for variety
        try:
            indices_xy = np.random.choice(total_masked, size=N_selected_xy, replace=False, p=probabilities_xy)
        except ValueError as e:
            print(f"Error during weighted sampling for 2D plot: {e}")
            indices_xy = np.random.choice(total_masked, size=N_selected_xy, replace=False)

        x_plot_xy_w = x_masked[indices_xy]
        y_plot_xy_w = y_masked[indices_xy]
    else:
        x_plot_xy_w = np.array([])
        y_plot_xy_w = np.array([])

    plt.figure(figsize=(10, 8))
    if N_selected_xy > 0:
        plt.scatter(x_plot_xy_w, y_plot_xy_w, s=1, alpha=0.5, c='red', label=f'selected {N_selected_xy} decays')
    plt.xlabel(r"$x_{\mathrm{mother}}$ [m]", fontsize=14)
    plt.ylabel(r"$y_{\mathrm{mother}}$ [m]", fontsize=14)
    plt.title(f"decay positions (z < {z_min_decay:.0f} m) weighted by $P_{{\\mathrm{{decay}}}}$", fontsize=16)
    if N_selected_xy > 0:
        plt.legend(fontsize=12)
    # Add LLP information text
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="white", 
                       edgecolor="black", 
                       alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"decay_positions_xy_weighted_z_less_{int(z_min_decay)}.pdf"), bbox_inches='tight')
    plt.close()

    # ===========================
    # Combined Histogram: Channel vs ifAllPoint/N_events
    # ===========================
    plt.figure(figsize=(12, 8))
    channel_names = list(channels.keys())
    ratios = [quantities['ifAllPoint_ratios'].get(ch, 0) for ch in channel_names]
    plt.bar(channel_names, ratios, color='green', edgecolor='black')
    plt.title("fraction of events where all non-v decay products point to detector", fontsize=16)
    plt.xlabel("Channel", fontsize=14)
    plt.ylabel("fraction", fontsize=14)
    plt.ylim(0, 1.05)  # Since it's a ratio
    plt.xticks(rotation=45, ha='right')
    # Add LLP information text
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=12,
             bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="white", 
                       edgecolor="black", 
                       alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "channels_ifAllPoint_ratio.pdf"), bbox_inches='tight')
    plt.close()

def main():
    # Directory containing the files
    directory = 'outputs'

    # Hardcoded export option
    ifExportData = True  # Set to True to export the data table

    # Step 1: Parse filenames
    llp_dict = parse_filenames(directory)

    if not llp_dict:
        print("No llp files found in the specified directory.")
        sys.exit(1)

    # Step 2: User selection
    selected_file, selected_llp, selected_mass, selected_lifetime, selected_mixing_patterns = user_selection(llp_dict)

    # Set plots_directory to 'plots/selected_llp'
    plots_directory = os.path.join('plots', selected_llp)

    # Get the output filename without extension
    output_filename = os.path.splitext(os.path.basename(selected_file))[0]
    output_path = os.path.join(plots_directory, output_filename)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Step 3: Read file
    filepath = os.path.join(directory, selected_file)
    finalEvents, coupling_squared, epsilon_polar, epsilon_azimuthal, br_visible_val, channels = read_file(filepath)

    # Step 4: Plot channels with LLP info
    plot_channels(channels, finalEvents, output_path, selected_llp, selected_mass, selected_lifetime)

    # Step 5: Extract quantities (with debugging)
    quantities = extract_quantities(channels)

    # Step 6: Export data table if option is True
    if ifExportData:
        # Convert lists to numpy arrays
        energy_mother = np.array(quantities['energy_mother'])
        P_decay_mother = np.array(quantities['P_decay_mother'])
        z_mother = np.array(quantities['z_mother'])

        # Stack the columns: P_decay, energy, z_mother
        data_table = np.column_stack((P_decay_mother, energy_mother, z_mother))

        # Save the data table to a single text file with space delimiter and no header
        np.savetxt(os.path.join(output_path, 'data_table.txt'), data_table, fmt='%.6e', delimiter=' ')

        print(f"data table with p_decay, energy, z_mother has been exported to '{output_path}/data_table.txt'.")

    # Step 7: Plot histograms with LLP info
    plot_histograms(quantities, channels, output_path, selected_llp, selected_mass, selected_lifetime)

    # Step 8: Write final_states.txt
    final_states_path = os.path.join(output_path, 'final_states.txt')
    # Define detailed final state particles (particles and antiparticles)
    detailed_final_state_particles = [
        'e-', 'e+', 'mu-', 'mu+', 'pi-', 'pi+', 'K-', 'K+', 'K_L',
        'p', 'bar[p]', 'n', 'bar[n]', 'nu', 'gamma'
    ]

    with open(final_states_path, 'w') as f:
        # Write header with N_occurences as the first column
        header = 'N_occurences ' + ' '.join([f'N_{ptype}' for ptype in detailed_final_state_particles])
        f.write('channel ' + header + '\n')

        for channel, state_counter in quantities['final_states_per_channel'].items():
            if not state_counter:
                continue  # Skip channels with no final states
            # Write channel name once
            f.write(f"{channel}\n")
            # Write all state counts with N_occurences
            for state, count in state_counter.items():
                # state is a tuple of counts in the order of detailed_final_state_particles
                state_counts = ' '.join(map(str, state))
                f.write(f"{count} {state_counts}\n")

    print(f"final states have been exported to '{final_states_path}'.")

if __name__ == '__main__':
    main()

