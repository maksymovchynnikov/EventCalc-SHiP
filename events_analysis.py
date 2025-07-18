import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from funcs.ship_setup import plot_decay_volume, z_min, z_max, y_max, x_max  # Ensure these are correctly defined
from collections import defaultdict
from mpl_toolkits.mplot3d import Axes3D
from funcs.selecting_processing import parse_filenames, user_selection, read_file  # Ensure these are correctly defined

def plot_channels(channels, finalEvents, output_path, LLP_name, mass, lifetime):
    """
    Plots histogram for channels and adds LLP information text.
    """
    channel_names = list(channels.keys())
    channel_sizes = [channels[ch]['size'] for ch in channel_names]

    plt.figure(figsize=(16, 9))
    plt.bar(channel_names, channel_sizes, color='skyblue', edgecolor='black')
    plt.title(f"$N_{{\\mathrm{{entries}}}}$ = {finalEvents:.0f}", fontsize=35)
    plt.ylabel("Number of events", fontsize=34)
    plt.xticks(rotation=45, ha='right', fontsize=39)
    plt.yticks(fontsize=39)
    plt.tick_params(axis='both', which='both', labelsize=39, width=2, length=10)

    textstr = f"LLP: {LLP_name}\nMass: {mass:.2f} GeV\nLifetime: {lifetime:.2e} s"
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=24,
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor="white",
                       edgecolor="black",
                       alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "channels.pdf"), bbox_inches='tight')
    plt.close()

def extract_quantities(channels, ifDisplaypdgs=False):
    """
    Extracts required quantities from the data lines.
    Returns a dictionary with the extracted quantities.

    Important Points:
    - The first 10 values are mother particle info.
    - Each decay product has 6 values: px, py, pz, E, mass, PDG.
    - Invariant masses are computed from 4-vectors of the decay products.

    Modified according to the new requirements:
    - We do not break early when a particle doesn't point.
    - If a detectable decay product points to the detector, we add its 4-momentum to a reconstructible sum.
    - If at least two pointing detectable decay products exist, we compute reconstructed invariant mass weighted by P_decay_mother.

    Restoring the functionality for ifAllPoint_ratio:
    - For each event, if ALL detectable decay products (non-neutrino, non -999) point to the detector,
      we add P_decay_mother to ifAllPoint_counts for that channel. If at least one does not point, we don't add it.
    """

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
        'decay_products_counts': [],
        'charged_decay_products_counts': [],
        'nu_counts': [],
        'decay_products_per_event_counts': defaultdict(list),
        'ifAllPoint_counts': defaultdict(float),
        'sum_P_decay_mother_per_channel': defaultdict(float),
        'ifAllPoint_ratios': {},
        'final_states_per_channel': defaultdict(lambda: defaultdict(int)),
        'invariant_mass_all_per_channel': defaultdict(list),
        'invariant_mass_detectable_per_channel': defaultdict(list),
        'reconstructed_invariant_mass_per_channel': defaultdict(lambda: {'mass': [], 'weight': []}),
        'reconstructed_invariant_mass_all': {'mass': [], 'weight': []}
    }

    detailed_final_state_particles = [
        'e-', 'e+', 'mu-', 'mu+', 'pi-', 'pi+', 'K-', 'K+', 'K_L',
        'p', 'bar[p]', 'n', 'bar[n]', 'nu', 'gamma', 'chi'
    ]

    pdg_to_particle = {
        22: 'gamma',
        11: 'e-',
        -11: 'e+',
        13: 'mu-',
        -13: 'mu+',
        211: 'pi+',
        -211: 'pi-',
        321: 'K+',
        -321: 'K-',
        130: 'K_L',
        2212: 'p',
        -2212: 'bar[p]',
        2112: 'n',
        -2112: 'bar[n]',
        12: 'nu',
        -12: 'nu',
        14: 'nu',
        -14: 'nu',
        16: 'nu',
        -16: 'nu',
        1000022: 'chi'
    }

    for channel_idx, (channel, channel_data) in enumerate(channels.items(), start=1):
        if ifDisplaypdgs:
            print(f"Channel {channel_idx}: {channel}")
        data_lines = channel_data['data']

        for event_idx, data_line in enumerate(data_lines, start=1):
            parts = data_line.strip().split()
            try:
                numbers = list(map(float, parts))
            except ValueError:
                continue

            if len(numbers) < 10:
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

            decay_products = numbers[10:]
            if len(decay_products) % 6 != 0:
                continue

            num_decay_products = len(decay_products)//6

            pdg_list = [int(decay_products[6*i + 5]) for i in range(num_decay_products)]
            if ifDisplaypdgs:
                print(f"  Event {event_idx}: pdg list = {pdg_list}")

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

            quantities['sum_P_decay_mother_per_channel'][channel] += P_decay_mother

            decay_products_count = 0
            charged_decay_products_count = 0
            nu_count = 0
            final_state_counts = {ptype: 0 for ptype in detailed_final_state_particles}

            # Restoring all_point logic:
            # Start with all_point = True
            all_point = True

            pointing_detectables = 0
            total_px_pointing = 0.0
            total_py_pointing = 0.0
            total_pz_pointing = 0.0
            total_e_pointing = 0.0

            # For all products (including neutrinos)
            total_decay_energy_all = 0.0
            total_decay_px_all = 0.0
            total_decay_py_all = 0.0
            total_decay_pz_all = 0.0

            # For detectable decay products only
            total_decay_energy_detectable = 0.0
            total_decay_px_detectable = 0.0
            total_decay_py_detectable = 0.0
            total_decay_pz_detectable = 0.0

            for i in range(num_decay_products):
                base = i*6
                px = decay_products[base]
                py = decay_products[base + 1]
                pz_dp = decay_products[base + 2]
                e = decay_products[base + 3]
                mass_dp = decay_products[base + 4]
                pdg = int(decay_products[base + 5])

                # All products sum
                if pdg != -999:
                    total_decay_energy_all += e
                    total_decay_px_all += px
                    total_decay_py_all += py
                    total_decay_pz_all += pz_dp

                if pdg in [12, -12, 14, -14, 16, -16]:
                    # Neutrino
                    nu_count += 1
                    final_state_counts['nu'] += 1
                elif pdg == -999:
                    # Placeholder, ignore
                    pass
                else:
                    # Detectable decay product
                    total_decay_energy_detectable += e
                    total_decay_px_detectable += px
                    total_decay_py_detectable += py
                    total_decay_pz_detectable += pz_dp

                    particle = pdg_to_particle.get(pdg, 'other')
                    if particle in final_state_counts:
                        final_state_counts[particle] += 1

                    decay_products_count += 1
                    # Charged: exclude gamma, K_L, n, bar[n], nu
                    if particle not in ['gamma', 'K_L', 'n', 'bar[n]', 'nu', 'chi']:
                        charged_decay_products_count += 1

                    # Check if this detectable decay product points to the detector
                    if pz_dp != 0:
                        x_proj = x_mother + (z_max - z_mother)*px/pz_dp
                        y_proj = y_mother + (z_max - z_mother)*py/pz_dp
                        if (-y_max(z_max)<y_proj<y_max(z_max) and -x_max(z_max)<x_proj<x_max(z_max)):
                            pointing_detectables += 1
                            total_px_pointing += px
                            total_py_pointing += py
                            total_pz_pointing += pz_dp
                            total_e_pointing += e
                        else:
                            # This detectable decay product does not point, set all_point = False
                            all_point = False
                    else:
                        # pz_dp=0 means no pointing
                        all_point = False

            quantities['decay_products_counts'].append(decay_products_count)
            quantities['charged_decay_products_counts'].append(charged_decay_products_count)
            quantities['nu_counts'].append(nu_count)

            for ptype in detailed_final_state_particles:
                quantities['decay_products_per_event_counts'][ptype].append(final_state_counts[ptype])

            state_tuple = tuple(final_state_counts[ptype] for ptype in detailed_final_state_particles)
            quantities['final_states_per_channel'][channel][state_tuple] += 1

            # Invariant mass for all products
            if num_decay_products > 0:
                invariant_mass_sq_all = total_decay_energy_all**2 - (total_decay_px_all**2 + total_decay_py_all**2 + total_decay_pz_all**2)
                if invariant_mass_sq_all < 0:
                    continue
                invariant_mass_all = np.sqrt(invariant_mass_sq_all)
            else:
                invariant_mass_all = 0.0

            # Invariant mass for detectable decay products
            if num_decay_products > 0:
                invariant_mass_sq_detectable = total_decay_energy_detectable**2 - (total_decay_px_detectable**2 + total_decay_py_detectable**2 + total_decay_pz_detectable**2)
                if invariant_mass_sq_detectable < 0:
                    continue
                invariant_mass_detectable = np.sqrt(invariant_mass_sq_detectable)
            else:
                invariant_mass_detectable = 0.0

            # Reconstructible invariant mass for pointing detectable decay products
            if pointing_detectables >= 2:
                inv_mass_sq_recon = total_e_pointing**2 - (total_px_pointing**2 + total_py_pointing**2 + total_pz_pointing**2)
                if inv_mass_sq_recon >= 0:
                    reconstructed_mass = np.sqrt(inv_mass_sq_recon)
                    quantities['reconstructed_invariant_mass_per_channel'][channel]['mass'].append(reconstructed_mass)
                    quantities['reconstructed_invariant_mass_per_channel'][channel]['weight'].append(P_decay_mother)
                    quantities['reconstructed_invariant_mass_all']['mass'].append(reconstructed_mass)
                    quantities['reconstructed_invariant_mass_all']['weight'].append(P_decay_mother)

            if ifDisplaypdgs:
                print(f"Event {event_idx}: p_x,mother = {px_mother}, p_x,products,total = {total_decay_px_all}, "
                      f"p_y,mother = {py_mother}, p_y,products,total = {total_decay_py_all}, "
                      f"p_z,mother = {pz_mother}, p_z,products,total = {total_decay_pz_all}, "
                      f"E_mother = {energy_mother}, E_products,total = {total_decay_energy_all}")
                print(f"m_mother = {m_mother}, m_inv,products = {invariant_mass_all}")

            quantities['invariant_mass_all_per_channel'][channel].append(invariant_mass_all)
            quantities['invariant_mass_detectable_per_channel'][channel].append(invariant_mass_detectable)

            # Restore adding P_decay_mother if ALL detectable decay products point
            if all_point:
                quantities['ifAllPoint_counts'][channel] += P_decay_mother

    for channel in channels.keys():
        sum_P_decay = quantities['sum_P_decay_mother_per_channel'][channel]
        if sum_P_decay > 0:
            ratio = quantities['ifAllPoint_counts'][channel]/sum_P_decay
        else:
            ratio = 0
        quantities['ifAllPoint_ratios'][channel] = ratio

    return quantities

def plot_histograms(quantities, channels, output_path, LLP_name, mass, lifetime):
    """
    Plots the required histograms and saves them in the output_path directory.
    All histograms are normalized to represent probability densities.
    Adds LLP information text to each plot.
    """

    multiplicities_path = os.path.join(output_path, "multiplicities")
    if not os.path.exists(multiplicities_path):
        os.makedirs(multiplicities_path)

    invariant_mass_path = os.path.join(output_path, "invariant-mass")
    if not os.path.exists(invariant_mass_path):
        os.makedirs(invariant_mass_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    textstr = f"LLP: {LLP_name}\nMass: {mass:.2f} GeV\nLifetime: {lifetime:.2e} s"

    energy_mother = np.array(quantities['energy_mother'])
    P_decay_mother = np.array(quantities['P_decay_mother'])
    z_mother = np.array(quantities['z_mother'])
    x_mother = np.array(quantities['x_mother'])
    y_mother = np.array(quantities['y_mother'])
    nu_counts = np.array(quantities['nu_counts'])

    # Determine mother mass for setting plot range
    if len(quantities['m_mother']) > 0:
        mother_mass = quantities['m_mother'][0]
    else:
        mother_mass = mass  # fallback if no events

    # Energy Histograms Unweighted
    plt.figure(figsize=(18, 12))
    plt.hist(energy_mother, bins=50, color='skyblue', edgecolor='black', density=True)
    plt.yscale('log')
    plt.xlabel("$E_{\\mathrm{LLP}}$ [GeV]", fontsize=48)
    plt.ylabel("Probability density", fontsize=48)
    plt.title("LLP energy distribution (unweighted)", fontsize=39)
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right', verticalalignment='top',
             transform=plt.gca().transAxes, fontsize=36,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="black", alpha=0.8))
    plt.tick_params(axis='both', which='both', labelsize=50, width=2, length=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "energy_mother_unweighted.pdf"), bbox_inches='tight')
    plt.close()

    # Energy Histograms Weighted
    plt.figure(figsize=(18, 12))
    plt.hist(energy_mother, bins=50, weights=P_decay_mother, color='salmon', edgecolor='black', density=True)
    plt.yscale('log')
    plt.xlabel("$E_{\\mathrm{LLP}}$ [GeV]", fontsize=48)
    plt.ylabel("Probability density", fontsize=48)
    plt.title("LLP energy distribution (weighted by $P_{\\mathrm{decay}}$)", fontsize=39)
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right', verticalalignment='top',
             transform=plt.gca().transAxes, fontsize=36,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="black", alpha=0.8))
    plt.tick_params(axis='both', which='both', labelsize=50, width=2, length=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "energy_mother_weighted.pdf"), bbox_inches='tight')
    plt.close()

    # P_decay_mother Histogram
    plt.figure(figsize=(18, 12))
    plt.hist(P_decay_mother, bins=50, color='lightgreen', edgecolor='black', density=True)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("$P_{\\mathrm{decay,LLP}}$", fontsize=59)
    plt.ylabel("Probability density", fontsize=59)
    plt.title("LLP decay probability distribution", fontsize=39)
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right', verticalalignment='top',
             transform=plt.gca().transAxes, fontsize=36,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="black", alpha=0.8))
    ax = plt.gca()
    ax.tick_params(axis='both', which='both', labelsize=48, width=2, length=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "P_decay_mother.pdf"), bbox_inches='tight')
    plt.close()

    # z_mother Weighted Histogram
    plt.figure(figsize=(18, 12))
    plt.hist(z_mother, bins=50, weights=P_decay_mother, color='violet', edgecolor='black', density=True)
    plt.yscale('log')
    plt.xlabel("z [m]", fontsize=39)
    plt.ylabel("Probability density", fontsize=39)
    plt.title("LLP decay positions (weighted by $P_{\\mathrm{decay}}$)", fontsize=30)
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right', verticalalignment='top',
             transform=plt.gca().transAxes, fontsize=36,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="black", alpha=0.8))
    ax = plt.gca()
    ax.tick_params(axis='both', which='both', labelsize=41, width=2, length=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "z_mother_weighted.pdf"), bbox_inches='tight')
    plt.close()

    # z_mother Unweighted Histogram
    plt.figure(figsize=(18, 12))
    plt.hist(z_mother, bins=50, color='cyan', edgecolor='black', density=True)
    plt.yscale('log')
    plt.xlabel("z [m]", fontsize=36)
    plt.ylabel("Probability density", fontsize=36)
    plt.title("LLP decay positions (unweighted)", fontsize=30)
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right', verticalalignment='top',
             transform=plt.gca().transAxes, fontsize=36,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="black", alpha=0.8))
    plt.tick_params(axis='both', which='both', labelsize=50, width=2, length=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "decay_positions_unweighted.pdf"), bbox_inches='tight')
    plt.close()

    # Decay products multiplicity histograms merged
    plt.figure(figsize=(18, 12))
    max_count_all = max(quantities['decay_products_counts']) if quantities['decay_products_counts'] else 0
    max_count_charged = max(quantities['charged_decay_products_counts']) if quantities['charged_decay_products_counts'] else 0
    max_count = max(max_count_all, max_count_charged)
    bins = np.arange(-0.5, int(max_count) + 1.5, 1)
    plt.hist(quantities['decay_products_counts'], bins=bins, alpha=0.5, label='All decay products', color='blue', edgecolor='black', density=True)
    plt.hist(quantities['charged_decay_products_counts'], bins=bins, alpha=0.5, label='Charged decay products', color='yellow', edgecolor='black', density=True)
    plt.ylabel("Probability density", fontsize=48)
    plt.title("Decay products multiplicity", fontsize=39)
    plt.legend(loc='best', fontsize=24)
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right', verticalalignment='top',
             transform=plt.gca().transAxes, fontsize=36,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="black", alpha=0.8))
    if max_count <=10:
        tick_positions = np.arange(0, 11, 1)
    else:
        tick_positions = np.arange(0, int(max_count) + 1, 2)
    plt.xticks(tick_positions, fontsize=50)
    plt.yticks(fontsize=50)
    plt.tick_params(axis='both', which='both', labelsize=50, width=2, length=10)
    plt.tight_layout()
    plt.savefig(os.path.join(multiplicities_path, "decay_products_counts_merged.pdf"), bbox_inches='tight')
    plt.close()

    # Combined Decay Products Types
    combine_mapping = {
        'e': ['e-', 'e+'],
        'mu': ['mu-', 'mu+'],
        'pi': ['pi-', 'pi+'],
        'k': ['K-', 'K+'],
        'K_L': ['K_L'],
        'p': ['p', 'bar[p]'],
        'n': ['n', 'bar[n]'],
        'gamma': ['gamma'],
        'chi': ['chi']
    }

    if len(quantities['decay_products_per_event_counts']['e-']) > 0:
        combined_counts = defaultdict(list)
        for combined_ptype, constituent_ptypes in combine_mapping.items():
            for event_idx in range(len(quantities['decay_products_per_event_counts']['e-'])):
                total = 0
                for ptype in constituent_ptypes:
                    total += quantities['decay_products_per_event_counts'][ptype][event_idx]
                combined_counts[combined_ptype].append(total)

        for ptype, counts in combined_counts.items():
            if counts:
                plt.figure(figsize=(18, 12))
                max_count = max(counts)
                bins = np.arange(-0.5, int(max_count) + 1.5, 1)
                plt.hist(counts, bins=bins, align='mid', edgecolor='black', color='lightcoral', density=True)
                if ptype == 'e':
                    xlabel = r"$e^{\pm}$ multiplicity"
                elif ptype == 'mu':
                    xlabel = r"$\mu^{\pm}$ multiplicity"
                elif ptype == 'pi':
                    xlabel = r"$\pi^{\pm}$ multiplicity"
                elif ptype == 'k':
                    xlabel = r"$K^{\pm}$ multiplicity"
                elif ptype == 'K_L':
                    xlabel = r"$K_{L}$ multiplicity"
                elif ptype == 'p':
                    xlabel = r"$p^{\pm}$ multiplicity"
                elif ptype == 'n':
                    xlabel = r"$n^{\pm}$ multiplicity"
                elif ptype == 'gamma':
                    xlabel = r"$\gamma$ multiplicity"
                elif ptype == 'chi':
                    xlabel = r"$\chi$ multiplicity"
                else:
                    xlabel = f"{ptype} multiplicity"

                plt.ylabel("Probability density", fontsize=48)
                plt.title(xlabel, fontsize=39)
                if max_count <=10:
                    tick_positions = np.arange(0, 11, 1)
                else:
                    tick_positions = np.arange(0, int(max_count) + 1, 2)
                plt.xticks(tick_positions, fontsize=50)
                plt.yticks(fontsize=50)
                plt.tick_params(axis='both', which='both', labelsize=50, width=2, length=10)
                plt.text(0.95, 0.95, textstr,
                         horizontalalignment='right', verticalalignment='top',
                         transform=plt.gca().transAxes, fontsize=36,
                         bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                                   edgecolor="black", alpha=0.8))
                plt.tight_layout()
                plt.savefig(os.path.join(multiplicities_path, f"decay_products_counts_{ptype}.pdf"), bbox_inches='tight')
                plt.close()

    # Neutrino Counts
    if 'nu' in quantities['decay_products_per_event_counts']:
        nu_counts_list = quantities['nu_counts']
        if len(nu_counts_list) > 0:
            plt.figure(figsize=(18, 12))
            max_nu = max(nu_counts_list)
            bins = range(0, int(max_nu) + 2)
            plt.hist(nu_counts_list, bins=bins, align='left', edgecolor='black', color='green', density=True)
            plt.xlabel("Number of neutrinos per event", fontsize=48)
            plt.ylabel("Probability density", fontsize=48)
            plt.title("Neutrino multiplicity per event", fontsize=39)
            plt.tick_params(axis='both', which='both', labelsize=50, width=2, length=10)
            plt.text(0.95, 0.95, textstr,
                     horizontalalignment='right', verticalalignment='top',
                     transform=plt.gca().transAxes, fontsize=36,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                               edgecolor="black", alpha=0.8))
            plt.tight_layout()
            plt.savefig(os.path.join(multiplicities_path, "decay_products_counts_nu.pdf"), bbox_inches='tight')
            plt.close()

    # 3D Decay Positions Unweighted
    max_points = 10000
    total_points = len(x_mother)
    if total_points > max_points:
        np.random.seed(42)
        indices_unw = np.random.choice(total_points, max_points, replace=False)
        x_plot_unw = x_mother[indices_unw]
        y_plot_unw = y_mother[indices_unw]
        z_plot_unw = z_mother[indices_unw]
    else:
        x_plot_unw = x_mother
        y_plot_unw = y_mother
        z_plot_unw = z_mother

    fig = plt.figure(figsize=(24, 16))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_plot_unw, y_plot_unw, z_plot_unw, s=10, alpha=0.5, c='blue')
    plot_decay_volume(ax)
    ax.set_xlabel("x [m]", fontsize=36)
    ax.set_ylabel("y [m]", fontsize=36)
    ax.set_zlabel("z [m]", fontsize=36)
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    ax.zaxis.labelpad = 15
    plt.title("Decay positions of LLP (unweighted)", fontsize=30)

    ax.text2D(0.95, 0.95, textstr,
              horizontalalignment='right', verticalalignment='top',
              transform=ax.transAxes, fontsize=24,
              bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                        edgecolor="black", alpha=0.8))
    ax.tick_params(axis='both', which='both', labelsize=24, width=2, length=10)
    right_padding = ax.text2D(1.2, 0.5, '', transform=ax.transAxes)
    plt.savefig(os.path.join(output_path, "decay_positions_unweighted_3D.pdf"),
                bbox_inches='tight', bbox_extra_artists=[right_padding],
                pad_inches=0.5)
    plt.close()

    # 3D Decay Positions Weighted
    N_selected = len(x_mother) // 10
    max_selected = 10000
    if N_selected > max_selected:
        N_selected = max_selected
    if N_selected > len(x_mother):
        N_selected = len(x_mother)

    if N_selected > 0 and P_decay_mother.sum() > 0:
        probabilities = P_decay_mother / P_decay_mother.sum()
        np.random.seed(24)
        try:
            indices_w = np.random.choice(len(x_mother), size=N_selected, replace=False, p=probabilities)
        except ValueError:
            indices_w = np.random.choice(len(x_mother), size=N_selected, replace=False)
        x_plot_w = x_mother[indices_w]
        y_plot_w = y_mother[indices_w]
        z_plot_w = z_mother[indices_w]
    else:
        x_plot_w = np.array([])
        y_plot_w = np.array([])
        z_plot_w = np.array([])

    fig = plt.figure(figsize=(24, 16))
    ax = fig.add_subplot(111, projection='3d')
    if len(x_plot_w) > 0:
        ax.scatter(x_plot_w, y_plot_w, z_plot_w, s=10, alpha=0.5, c='red')
    plot_decay_volume(ax)
    ax.set_xlabel("x [m]", fontsize=36)
    ax.set_ylabel("y [m]", fontsize=36)
    ax.set_zlabel("z [m]", fontsize=36)
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    ax.zaxis.labelpad = 15
    plt.title("Decay positions of LLP (weighted by $P_{\\mathrm{decay}}$)", fontsize=30)
    ax.set_zlim(z_min, z_max + 5)
    ax.text2D(0.95, 0.95, textstr,
              horizontalalignment='right', verticalalignment='top',
              transform=ax.transAxes, fontsize=24,
              bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                        edgecolor="black", alpha=0.8))
    ax.tick_params(axis='both', which='both', labelsize=24, width=2, length=10)
    right_padding = ax.text2D(1.2, 0.5, '', transform=ax.transAxes)
    plt.savefig(os.path.join(output_path, "decay_positions_weighted_3D.pdf"),
                bbox_inches='tight', bbox_extra_artists=[right_padding],
                pad_inches=0.5)
    plt.close()

    # Decay Positions z < z_min_decay Unweighted
    z_min_decay = z_min + 1
    mask_z = z_mother < z_min_decay
    plt.figure(figsize=(20, 15))
    plt.scatter(x_mother[mask_z], y_mother[mask_z], s=10, alpha=0.5, c='blue')
    plt.xlabel("x [m]", fontsize=36)
    plt.ylabel("y [m]", fontsize=36)
    plt.title(f"Decay positions (z < {z_min_decay:.0f} m) unweighted", fontsize=39)
    plt.legend(fontsize=36)
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right', verticalalignment='top',
             transform=plt.gca().transAxes, fontsize=36,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="black", alpha=0.8))
    plt.tick_params(axis='both', which='both', labelsize=50, width=2, length=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"decay_positions_xy_unweighted_z_less_{int(z_min_decay)}.pdf"),
                bbox_inches='tight')
    plt.close()

    # Decay Positions z < z_min_decay Weighted
    x_masked = x_mother[mask_z]
    y_masked = y_mother[mask_z]
    P_decay_masked = P_decay_mother[mask_z]
    total_masked = len(x_masked)
    N_selected_xy = total_masked // 10
    max_selected = 10000
    if N_selected_xy > max_selected:
        N_selected_xy = max_selected
    if N_selected_xy > total_masked:
        N_selected_xy = total_masked

    if N_selected_xy > 0 and P_decay_masked.sum() > 0:
        probabilities_xy = P_decay_masked / P_decay_masked.sum()
        np.random.seed(100)
        try:
            indices_xy = np.random.choice(total_masked, size=N_selected_xy, replace=False, p=probabilities_xy)
        except ValueError:
            indices_xy = np.random.choice(total_masked, size=N_selected_xy, replace=False)
        x_plot_xy_w = x_masked[indices_xy]
        y_plot_xy_w = y_masked[indices_xy]
    else:
        x_plot_xy_w = np.array([])
        y_plot_xy_w = np.array([])

    plt.figure(figsize=(20, 15))
    if len(x_plot_xy_w) > 0:
        plt.scatter(x_plot_xy_w, y_plot_xy_w, s=15, alpha=0.5, c='red')
    plt.xlabel("x [m]", fontsize=36)
    plt.ylabel("y [m]", fontsize=36)
    plt.title(f"Decay positions (z < {z_min_decay:.0f} m) weighted by $P_{{\\mathrm{{decay}}}}$", fontsize=51)
    if len(x_plot_xy_w) > 0:
        plt.legend(fontsize=36)
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right', verticalalignment='top',
             transform=plt.gca().transAxes, fontsize=36,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="black", alpha=0.8))
    plt.tick_params(axis='both', which='both', labelsize=50, width=2, length=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"decay_positions_xy_weighted_z_less_{int(z_min_decay)}.pdf"),
                bbox_inches='tight')
    plt.close()

    # Channels IfAllPoint Ratio
    plt.figure(figsize=(20, 15))
    channel_names = list(channels.keys())
    ratios = [quantities['ifAllPoint_ratios'].get(ch, 0) for ch in channel_names]
    plt.bar(channel_names, ratios, color='green', edgecolor='black')
    plt.title("Fraction of events where all non-v decay products point to detector", fontsize=35)
    plt.ylabel("Fraction", fontsize=68)
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha='right', fontsize=39)
    plt.yticks(fontsize=39)
    plt.text(0.95, 0.95, textstr,
             horizontalalignment='right', verticalalignment='top',
             transform=plt.gca().transAxes, fontsize=36,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor="black", alpha=0.8))
    plt.tick_params(axis='both', which='both', labelsize=39, width=2, length=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "channels_ifAllPoint_ratio.pdf"), bbox_inches='tight')
    plt.close()

    # Commenting out old invariant mass plots:
    # These were originally plotting invariant mass distributions for all products and detectable decay products.
    # They served as a cross-check for the invariant mass distributions.
    # The code below is commented out as requested.
    #
    # for channel in channels.keys():
    #     invariant_mass_all_channel = quantities['invariant_mass_all_per_channel'][channel]
    #     invariant_mass_detectable_channel = quantities['invariant_mass_detectable_per_channel'][channel]
    #
    #     # All products (commented out, used to cross-check):
    #     # if invariant_mass_all_channel:
    #     #     plt.figure(figsize=(18, 12))
    #     #     plt.hist(invariant_mass_all_channel, bins=50, color='purple', edgecolor='black', density=True)
    #     #     plt.yscale('log')
    #     #     plt.xlim(mother_mass - 0.01, mother_mass + 0.01)
    #     #     plt.xlabel("Invariant Mass [GeV]", fontsize=48)
    #     #     plt.ylabel("Probability density", fontsize=48)
    #     #     plt.title(f"Invariant Mass Distribution (unweighted, all products) - {channel}", fontsize=39)
    #     #     plt.text(0.95, 0.95, textstr,
    #     #              horizontalalignment='right', verticalalignment='top',
    #     #              transform=plt.gca().transAxes, fontsize=36,
    #     #              bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
    #     #                        edgecolor="black", alpha=0.8))
    #     #     plt.tick_params(axis='both', which='both', labelsize=50, width=2, length=10)
    #     #     sanitized_channel = channel.replace(" ", "_").replace("/", "_")
    #     #     plt.tight_layout()
    #     #     plt.savefig(os.path.join(invariant_mass_path, f"invariant_mass_unweighted_all_{sanitized_channel}.pdf"), bbox_inches='tight')
    #     #     plt.close()
    #
    #     # Detectable decay products (commented out, used to cross-check):
    #     # if invariant_mass_detectable_channel:
    #     #     plt.figure(figsize=(18, 12))
    #     #     plt.hist(invariant_mass_detectable_channel, bins=50, color='purple', edgecolor='black', density=True)
    #     #     plt.yscale('log')
    #     #     plt.xlim(0, mother_mass)
    #     #     plt.xlabel("Invariant Mass [GeV]", fontsize=48)
    #     #     plt.ylabel("Probability density", fontsize=48)
    #     #     plt.title(f"Invariant Mass Distribution (unweighted, detectable decay products) - {channel}", fontsize=39)
    #     #     plt.text(0.95, 0.95, textstr,
    #     #              horizontalalignment='right', verticalalignment='top',
    #     #              transform=plt.gca().transAxes, fontsize=36,
    #     #              bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
    #     #                        edgecolor="black", alpha=0.8))
    #     #     plt.tick_params(axis='both', which='both', labelsize=50, width=2, length=10)
    #     #     sanitized_channel = channel.replace(" ", "_").replace("/", "_")
    #     #     plt.tight_layout()
    #     #     plt.savefig(os.path.join(invariant_mass_path, f"invariant_mass_unweighted_detectable_{sanitized_channel}.pdf"), bbox_inches='tight')
    #     #     plt.close()

    # Now we plot weighted reconstructible invariant mass histograms per channel and a global one.
    # Range: from 0 to mother_mass, density=True, weights=P_decay_mother
    x_min_recon = 0
    x_max_recon = mother_mass+0.02

    # Per channel
    for channel in channels.keys():
        mass_arr = np.array(quantities['reconstructed_invariant_mass_per_channel'][channel]['mass'])
        weight_arr = np.array(quantities['reconstructed_invariant_mass_per_channel'][channel]['weight'])
        if len(mass_arr) > 0:
            plt.figure(figsize=(18, 12))
            plt.hist(mass_arr, bins=50, color='purple', edgecolor='black', density=True, weights=weight_arr)
            plt.yscale('log')
            plt.xlim(x_min_recon, x_max_recon)
            plt.xlabel(r"$m_{\mathrm{inv}}$ [GeV]", fontsize=48)
            plt.ylabel("Probability density", fontsize=48)
            plt.title(f"Reconstructible $m_{{\\mathrm{{inv}}}}$ - {channel}", fontsize=39)
            plt.text(0.95, 0.95, textstr,
                     horizontalalignment='right', verticalalignment='top',
                     transform=plt.gca().transAxes, fontsize=36,
                     bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                               edgecolor="black", alpha=0.8))
            plt.tick_params(axis='both', which='both', labelsize=50, width=2, length=10)
            sanitized_channel = channel.replace(" ", "_").replace("/", "_")
            plt.tight_layout()
            plt.savefig(os.path.join(invariant_mass_path, f"reconstructible_m_inv_weighted_{sanitized_channel}.pdf"), bbox_inches='tight')
            plt.close()

    # Combined over all channels
    mass_all = np.array(quantities['reconstructed_invariant_mass_all']['mass'])
    weight_all = np.array(quantities['reconstructed_invariant_mass_all']['weight'])
    if len(mass_all) > 0:
        plt.figure(figsize=(18, 12))
        plt.hist(mass_all, bins=50, color='purple', edgecolor='black', density=True, weights=weight_all)
        plt.yscale('log')
        plt.xlim(x_min_recon, x_max_recon)
        plt.xlabel(r"$m_{\mathrm{inv}}$ [GeV]", fontsize=48)
        plt.ylabel("Probability density", fontsize=48)
        plt.title("Reconstructible $m_{\\mathrm{inv}}$ (all channels)", fontsize=39)
        plt.text(0.95, 0.95, textstr,
                 horizontalalignment='right', verticalalignment='top',
                 transform=plt.gca().transAxes, fontsize=36,
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                           edgecolor="black", alpha=0.8))
        plt.tick_params(axis='both', which='both', labelsize=50, width=2, length=10)
        plt.tight_layout()
        plt.savefig(os.path.join(invariant_mass_path, "reconstructible_m_inv_weighted_all_channels.pdf"), bbox_inches='tight')
        plt.close()

def main():
    directory = 'outputs'
    ifExportData = True
    #Change to True if you want to display event details in console (pdg list, 4-momentum, etc.)
    ifDisplaypdgs = False

    LLP_dict = parse_filenames(directory)
    if not LLP_dict:
        print("No LLP files found in the specified directory.")
        sys.exit(1)

    selected_file, selected_LLP, selected_mass, selected_lifetime, selected_mixing_patterns = user_selection(LLP_dict)
    plots_directory = os.path.join('plots', selected_LLP)
    output_filename = os.path.splitext(os.path.basename(selected_file))[0]
    output_path = os.path.join(plots_directory, output_filename)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    finalEvents, coupling_squared, epsilon_polar, epsilon_azimuthal, br_visible_val, channels = read_file(os.path.join(directory, selected_file))

    plot_channels(channels, finalEvents, output_path, selected_LLP, selected_mass, selected_lifetime)
    quantities = extract_quantities(channels, ifDisplaypdgs=ifDisplaypdgs)

    if ifExportData:
        energy_mother = np.array(quantities['energy_mother'])
        P_decay_mother = np.array(quantities['P_decay_mother'])
        z_mother = np.array(quantities['z_mother'])
        data_table = np.column_stack((P_decay_mother, energy_mother, z_mother))
        np.savetxt(os.path.join(output_path, 'data_table.txt'), data_table, fmt='%.6e', delimiter=' ')
        print(f"Data table exported to '{output_path}/data_table.txt'.")

    plot_histograms(quantities, channels, output_path, selected_LLP, selected_mass, selected_lifetime)

    final_states_path = os.path.join(output_path, 'final_states.txt')
    detailed_final_state_particles = [
        'e-', 'e+', 'mu-', 'mu+', 'pi-', 'pi+', 'K-', 'K+', 'K_L',
        'p', 'bar[p]', 'n', 'bar[n]', 'nu', 'gamma', 'chi'
    ]
    with open(final_states_path, 'w') as f:
        header = 'N_occurences ' + ' '.join([f'N_{ptype}' for ptype in detailed_final_state_particles])
        f.write('channel ' + header + '\n')
        for channel, state_counter in quantities['final_states_per_channel'].items():
            if not state_counter:
                continue
            f.write(f"{channel}\n")
            for state, count in state_counter.items():
                state_counts = ' '.join(map(str, state))
                f.write(f"{count} {state_counts}\n")

    print(f"Final states exported to '{final_states_path}'.")

if __name__ == '__main__':
    main()

