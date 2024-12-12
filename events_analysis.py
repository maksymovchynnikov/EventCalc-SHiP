import os
import numpy as np
import re
import sys
import matplotlib.pyplot as plt
from funcs.ship_setup import plot_decay_volume, z_min, z_max, y_max, x_max  # Updated import
from collections import Counter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from funcs.selecting_processing import parse_filenames, user_selection, read_file  # Importing common functions
from matplotlib.lines import Line2D

def plot_channels(channels, finalEvents, output_path, llp_name, mass, lifetime):
    """
    Plots histogram for channels and adds LLP information text.
    """
    channel_names = list(channels.keys())
    channel_sizes = [channels[ch]['size'] for ch in channel_names]

    plt.figure(figsize=(10, 6))
    plt.bar(channel_names, channel_sizes, color='skyblue', edgecolor='black')
    plt.title(f"$N_{{\\mathrm{{entries}}}}$ = {finalEvents:.0f}")
    plt.xlabel("Channel")
    plt.ylabel("Number of events")
    plt.xticks(rotation=45)

    # Add LLP information text in the top right corner
    textstr = f"LLP: {llp_name}\nMass: {mass} GeV\nLifetime: {lifetime} s"
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="white", 
                       edgecolor="black", 
                       alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "channels.pdf"), bbox_inches='tight')
    plt.close()

def extract_quantities(channels):
    """
    Extracts required quantities from the data lines.
    Returns a dictionary with the extracted quantities.
    """
    # Initialize lists for quantities
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
        'decay_products_counts': [],  # total decay products per event
        'charged_decay_products_counts': [],  # charged decay products per event
        'decay_products_per_event_counts': {},  # counts of dedicated products per event
        'ifAllPoint_counts': Counter(),  # Weighted counts of events where all decay products point to detectors per channel
        'sum_P_decay_mother_per_channel': Counter(),  # Sum of P_decay_mother per channel
        'ifAllPoint_ratios': {},  # Ratio per channel
    }

    # Initialize per-event counts for dedicated decay products
    decay_products_types = ['e', 'mu', 'pi±', 'K±', 'K_L', 'gamma', 'p', 'n', 'nu', 'other']
    quantities['decay_products_per_event_counts'] = {ptype: [] for ptype in decay_products_types}

    for channel, channel_data in channels.items():
        data_lines = channel_data['data']
        for data_line in data_lines:
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
            PDG_mother = numbers[5]
            P_decay_mother = numbers[6]
            x_mother = numbers[7]
            y_mother = numbers[8]
            z_mother = numbers[9]

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

            # Now, count the number of decay products
            decay_products_count = 0
            charged_decay_products_count = 0

            # Initialize per-event Counter for dedicated products
            event_decay_products_counter = Counter()
            # The rest of the numbers are grouped into sets of 6: px, py, pz, e, mass, pdg
            decay_products = numbers[10:]
            all_point = True  # Initialize flag for ifAllPoint
            for i in range(0, len(decay_products), 6):
                # Each decay product has 6 numbers
                if i+6 > len(decay_products):
                    print(f"Error: Decay product data incomplete in line: {data_line}")
                    break
                px = decay_products[i]
                py = decay_products[i+1]
                pz = decay_products[i+2]
                e = decay_products[i+3]
                mass = decay_products[i+4]
                pdg = decay_products[i+5]
                if pdg == -999.0:
                    # Do not count
                    continue
                else:
                    # Exclude neutrinos
                    if int(abs(pdg)) in [12, 14, 16]:
                        continue
                    decay_products_count += 1
                    if pdg not in [22., 130., 310., 2112., -2112.]:
                        # These are neutral particles, so count as charged
                        charged_decay_products_count += 1
                    # Count dedicated products
                    pdg_int = int(pdg)
                    if pdg_int in [11, -11]:
                        event_decay_products_counter['e'] += 1
                    elif pdg_int in [13, -13]:
                        event_decay_products_counter['mu'] += 1
                    elif pdg_int in [211, -211]:
                        event_decay_products_counter['pi±'] += 1
                    elif pdg_int in [321, -321]:
                        event_decay_products_counter['K±'] += 1
                    elif pdg_int == 130:
                        event_decay_products_counter['K_L'] += 1
                    elif pdg_int == 22:
                        event_decay_products_counter['gamma'] += 1
                    elif pdg_int == 2212:
                        event_decay_products_counter['p'] += 1
                    elif pdg_int == 2112:
                        event_decay_products_counter['n'] += 1
                    elif pdg_int in [12, -12, 14, -14, 16, -16]:
                        event_decay_products_counter['nu'] +=1
                    else:
                        event_decay_products_counter['other'] +=1

                    # Calculate projections to z_max m plane
                    if pz == 0:
                        # Avoid division by zero; consider this event as not satisfying the condition
                        all_point = False
                        break
                    x_proj = x_mother + (z_max - z_mother) * px / pz
                    y_proj = y_mother + (z_max - z_mother) * py / pz

                    if not (-y_max(z_max) < y_proj < y_max(z_max) and -x_max(z_max) < x_proj < x_max(z_max)):
                        all_point = False
                        # No need to check further decay products for this event
                        break

            quantities['decay_products_counts'].append(decay_products_count)
            quantities['charged_decay_products_counts'].append(charged_decay_products_count)

            # For each particle type, append the count in this event
            for ptype in decay_products_types:
                count = event_decay_products_counter.get(ptype, 0)
                quantities['decay_products_per_event_counts'][ptype].append(count)

            # Update ifAllPoint_counts with weighted count
            if all_point:
                quantities['ifAllPoint_counts'][channel] += P_decay_mother  # Modified line

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
    import os
    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Prepare the text string for plots
    textstr = f"LLP: {llp_name}\nMass: {mass} GeV\nLifetime: {lifetime} s"

    # Convert relevant quantities to numpy arrays for easier handling
    energy_mother = np.array(quantities['energy_mother'])
    P_decay_mother = np.array(quantities['P_decay_mother'])
    z_mother = np.array(quantities['z_mother'])
    x_mother = np.array(quantities['x_mother'])
    y_mother = np.array(quantities['y_mother'])

    # Energy of mother particle (unweighted)
    plt.figure()
    plt.hist(energy_mother, bins=50, color='skyblue', edgecolor='black', density=True)
    plt.yscale('log')  # Preserving original y-axis scaling
    plt.xlabel("$E_{\\mathrm{LLP}}$ [GeV]")
    plt.ylabel("Probability density")
    plt.title("LLP Energy Distribution (Unweighted)")
    # Add LLP information text
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="white", 
                       edgecolor="black", 
                       alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "energy_mother_unweighted.pdf"), bbox_inches='tight')
    plt.close()

    # Energy of mother particle (weighted by P_decay)
    plt.figure()
    plt.hist(energy_mother, bins=50, weights=P_decay_mother, color='salmon', edgecolor='black', density=True)
    plt.yscale('log')  # Preserving original y-axis scaling
    plt.xlabel("$E_{\\mathrm{LLP}}$ [GeV]")
    plt.ylabel("Probability density")
    plt.title("LLP energy distribution (weighted by $P_{\\mathrm{decay}}$)")
    # Add LLP information text
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="white", 
                       edgecolor="black", 
                       alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "energy_mother_weighted.pdf"), bbox_inches='tight')
    plt.close()

    # P_decay of mother particle
    plt.figure()
    plt.hist(P_decay_mother, bins=50, color='lightgreen', edgecolor='black', density=True)
    plt.xscale('log')  # Preserving original x-axis scaling
    plt.yscale('log')  # Preserving original y-axis scaling
    plt.xlabel("$P_{\\mathrm{decay,LLP}}$")
    plt.ylabel("Probability density")
    plt.title("LLP decay probability distribution")
    # Add LLP information text
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="white", 
                       edgecolor="black", 
                       alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "P_decay_mother.pdf"), bbox_inches='tight')
    plt.close()

    # z_mother weighted by P_decay_mother
    plt.figure()
    plt.hist(z_mother, bins=50, weights=P_decay_mother, color='violet', edgecolor='black', density=True)
    plt.yscale('log')
    plt.xlabel("$z_{\\mathrm{decay,LLP}}$ [m]")
    plt.ylabel("Probability density")
    plt.title("LLP decay positions (weighted by $P_{\\mathrm{decay}}$)")
    # Add LLP information text
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="white", 
                       edgecolor="black", 
                       alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "z_mother_weighted.pdf"), bbox_inches='tight')
    plt.close()

    # z_mother unweighted
    plt.figure()
    plt.hist(z_mother, bins=50, color='cyan', edgecolor='black', density=True)
    plt.yscale('log')
    plt.xlabel("$z_{\\mathrm{decay,LLP}}$ [m]")
    plt.ylabel("Probability density")
    plt.title("LLP Decay Positions (Unweighted)")
    # Add LLP information text
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="white", 
                       edgecolor="black", 
                       alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "decay_positions_unweighted.pdf"), bbox_inches='tight')
    plt.close()

    # Merged histogram of total and charged decay products
    plt.figure()
    max_count = max(max(quantities['decay_products_counts']), max(quantities['charged_decay_products_counts']))
    bins = range(1, int(max_count) + 2)
    plt.hist(quantities['decay_products_counts'], bins=bins, alpha=0.5, label='All decay products', color='blue', edgecolor='black', density=True)
    plt.hist(quantities['charged_decay_products_counts'], bins=bins, alpha=0.5, label='Charged decay products', color='yellow', edgecolor='black', density=True)
    plt.xlabel("Number of decay products")
    plt.ylabel("Probability density")
    plt.title("Decay products multiplicity")
    plt.legend()
    # Add LLP information text
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="white", 
                       edgecolor="black", 
                       alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "decay_products_counts_merged.pdf"), bbox_inches='tight')
    plt.close()

    # Histograms of counts per event for each decay product type
    decay_products_types = quantities['decay_products_per_event_counts'].keys()
    for ptype in decay_products_types:
        counts = quantities['decay_products_per_event_counts'][ptype]
        if counts:
            plt.figure()
            max_count = max(counts)
            bins = range(int(max_count) + 2)  # +2 to include max_count
            plt.hist(counts, bins=bins, align='left', edgecolor='black', color='lightcoral', density=True)
            plt.xlabel(f"Number of {ptype} per event")
            plt.ylabel("Probability density")
            plt.title(f"Amount of {ptype} per event")
            plt.xticks(bins)
            # Add LLP information text
            plt.text(0.95, 0.95, textstr,
                     horizontalalignment='right',
                     verticalalignment='top',
                     transform=plt.gca().transAxes,
                     fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.3", 
                               facecolor="white", 
                               edgecolor="black", 
                               alpha=0.5))
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"decay_products_counts_{ptype}.pdf"), bbox_inches='tight')
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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(x_plot_unw, y_plot_unw, z_plot_unw, s=1, alpha=0.5, c='blue')
    
    plot_decay_volume(ax)  # Use ship_setup.py's plot_decay_volume
    
    ax.set_xlabel(r"$x_{\mathrm{mother}}$")
    ax.set_ylabel(r"$y_{\mathrm{mother}}$")
    ax.set_zlabel(r"$z_{\mathrm{mother}}$")
    
    plt.title("Decay positions of LLP (unweighted)")
    
    # Add LLP information text
    ax.text2D(0.95, 0.95, textstr,
              horizontalalignment='right',
              verticalalignment='top',
              transform=ax.transAxes,
              fontsize=10,
              bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor="white", 
                        edgecolor="black", 
                        alpha=0.5))
    
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

    if N_selected > 0:
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

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if N_selected > 0:
        ax.scatter(x_plot_w, y_plot_w, z_plot_w, s=1, alpha=0.5, c='red', label=f'Selected {N_selected} Decays')
    
    plot_decay_volume(ax)  # Use ship_setup.py's plot_decay_volume
    
    ax.set_xlabel(r"$x_{\mathrm{mother}}$")
    ax.set_ylabel(r"$y_{\mathrm{mother}}$")
    ax.set_zlabel(r"$z_{\mathrm{mother}}$")
    
    plt.title("Decay positions of LLP (weighted by $P_{\\mathrm{decay}}$)")

    # Set z-axis limits
    ax.set_zlim(z_min, z_max + 5)
    
    if N_selected > 0:
        plt.legend()
    
    # Add LLP information text
    ax.text2D(0.95, 0.95, textstr,
              horizontalalignment='right',
              verticalalignment='top',
              transform=ax.transAxes,
              fontsize=10,
              bbox=dict(boxstyle="round,pad=0.3", 
                        facecolor="white", 
                        edgecolor="black", 
                        alpha=0.5))
    
    # Create an invisible "dummy" text to add padding to the right
    right_padding = ax.text2D(1.2, 0.5, '', transform=ax.transAxes)  # 1.2 means extend right side

    # Save figure with extra space on the right using bbox_extra_artists
    plt.savefig(os.path.join(output_path, "decay_positions_weighted.pdf"), 
                bbox_inches='tight', 
                bbox_extra_artists=[right_padding], 
                pad_inches=0.5)
    plt.close()


    # ===========================
    # New 2D Point Plots Addition
    # ===========================

    z_min_decay = z_min+1  # This was previously hard-coded as 33 m
    mask_z = z_mother < z_min_decay

    # Unweighted 2D scatter plot of x and y decay coordinates for z_mother < z_min_decay
    plt.figure(figsize=(8,6))
    plt.scatter(x_mother[mask_z], y_mother[mask_z], s=1, alpha=0.5, c='blue')
    plot_decay_volume(ax)  # Use ship_setup.py's plot_decay_volume
    plt.xlabel(r"$x_{\mathrm{mother}}$ [m]")
    plt.ylabel(r"$y_{\mathrm{mother}}$ [m]")
    plt.title("Decay positions (z < 33 m) unweighted")
    # Add LLP information text
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="white", 
                       edgecolor="black", 
                       alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "decay_positions_xy_unweighted_z_less_33.pdf"), bbox_inches='tight')
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

    if N_selected_xy > 0:
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

    plt.figure(figsize=(8,6))
    if N_selected_xy > 0:
        plt.scatter(x_plot_xy_w, y_plot_xy_w, s=1, alpha=0.5, c='red', label=f'Selected {N_selected_xy} Decays')
    plt.xlabel(r"$x_{\mathrm{mother}}$ [m]")
    plt.ylabel(r"$y_{\mathrm{mother}}$ [m]")
    plt.title("Decay Positions (z < 33 m) Weighted by $P_{\\mathrm{decay}}$")
    if N_selected_xy > 0:
        plt.legend()
    # Add LLP information text
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="white", 
                       edgecolor="black", 
                       alpha=0.5))
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "decay_positions_xy_weighted_z_less_33.pdf"), bbox_inches='tight')
    plt.close()

    # ===========================
    # New Histogram: Channel vs ifAllPoint/N_events
    # ===========================

    # Extract ifAllPoint_ratios and plot them
    ifAllPoint_ratios = quantities.get('ifAllPoint_ratios', {})
    channel_names = list(channels.keys())
    ratios = [ifAllPoint_ratios.get(ch, 0) for ch in channel_names]

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(channel_names, ratios, color='green', edgecolor='black')
    plt.title("If all non-v decay products point to detector")
    plt.xlabel("Channel")
    plt.ylabel("Fraction")
    plt.ylim(0, 1.05)  # Since it's a ratio
    plt.xticks(rotation=45)
    # Add LLP information text
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", 
                       facecolor="white", 
                       edgecolor="black", 
                       alpha=0.5))
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
        print("No LLP files found in the specified directory.")
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
    finalEvents, epsilon_polar, epsilon_azimuthal, br_visible_val, channels = read_file(filepath)

    # Step 4: Plot channels with LLP info
    plot_channels(channels, finalEvents, output_path, selected_llp, selected_mass, selected_lifetime)

    # Step 5: Extract quantities
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

        print(f"Data table with P_decay, energy, z_mother has been exported to '{output_path}/data_table.txt'.")

    # Step 7: Plot histograms with LLP info
    plot_histograms(quantities, channels, output_path, selected_llp, selected_mass, selected_lifetime)

if __name__ == '__main__':
    main()

