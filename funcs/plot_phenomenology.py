# funcs/plot_phenomenology.py
import matplotlib.pyplot as plt
import os
import math
from matplotlib import gridspec

def plot_production_probability(masses_plot, Yield_plot, LLP, plot_folder):
    """
    Plots the total production probability of the LLP.

    Parameters:
    - masses_plot: Array of LLP masses.
    - Yield_plot: Array of production probabilities.
    - LLP: LLP object containing properties like LLP_name.
    - plot_folder: Directory to save the plot.
    """
    fig = plt.figure(figsize=(9, 6))  # 3:2 aspect ratio
    ax = fig.add_subplot(111)
    ax.loglog(masses_plot, Yield_plot, color='blue', linewidth=2)
    ax.set_xlabel(r"$m_{\mathrm{LLP}}\,[\mathrm{GeV}]$", fontsize=12)
    ax.set_ylabel(r"$P_{\mathrm{prod,LLP}}/\mathrm{coupling}^{2}\,[\mathrm{units}_{\mathrm{coupling}^{-2}}]$", fontsize=12)
    ax.set_title(f"Total Production Probability of {LLP.LLP_name}", fontsize=14)
    
    if LLP.LLP_name == "HNL":
        Ue2, Umu2, Utau2 = LLP.MixingPatternArray
        mixing_text = (f"Mixing pattern:\n"
                       f"$|U_e|^2$={Ue2:.3f}, "
                       f"$|U_\\mu|^2$={Umu2:.3f}, "
                       f"$|U_\\tau|^2$={Utau2:.3f}")
        ax.text(0.05, 0.95, mixing_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))
    elif LLP.LLP_name == "Dark-photons":
        uncertainty_text = f"The {LLP.uncertainty} position of production yield"
        ax.text(0.05, 0.95, uncertainty_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))
    
    ax.grid(True, which="both", linestyle='--', linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_folder, "production_probability.png"), dpi=300)
    plt.close(fig)

def plot_lifetime(masses_plot, ctau_int_plot, LLP, plot_folder):
    """
    Plots the proper lifetime of the LLP.

    Parameters:
    - masses_plot: Array of LLP masses.
    - ctau_int_plot: Array of proper lifetimes.
    - LLP: LLP object containing properties like LLP_name.
    - plot_folder: Directory to save the plot.
    """
    fig = plt.figure(figsize=(9, 6))  # 3:2 aspect ratio
    ax = fig.add_subplot(111)
    ax.loglog(masses_plot, ctau_int_plot, color='green', linewidth=2)
    ax.set_xlabel(r"$m_{\mathrm{LLP}}\,[\mathrm{GeV}]$", fontsize=12)
    ax.set_ylabel(r"$c\tau_{\mathrm{LLP}}\cdot \mathrm{coupling}^{2}\,[\mathrm{m}\cdot \mathrm{units}_{\mathrm{coupling}^{2}}]$", fontsize=12)
    ax.set_title(f"Proper Lifetime of {LLP.LLP_name}", fontsize=14)
    
    if LLP.LLP_name == "HNL":
        Ue2, Umu2, Utau2 = LLP.MixingPatternArray
        mixing_text = (f"Mixing pattern:\n"
                       f"$|U_e|^2$={Ue2:.3f}, "
                       f"$|U_\\mu|^2$={Umu2:.3f}, "
                       f"$|U_\\tau|^2$={Utau2:.3f}")
        ax.text(0.05, 0.95, mixing_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))
    elif LLP.LLP_name == "Dark-photons":
        uncertainty_text = f"The {LLP.uncertainty} position of production yield"
        ax.text(0.05, 0.95, uncertainty_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))
    
    ax.grid(True, which="both", linestyle='--', linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_folder, "lifetime.png"), dpi=300)
    plt.close(fig)

def plot_branching_ratios(masses_plot, Br_plot, chosen_channels, selected_decay_indices, LLP, plot_folder):
    """
    Plots the decay branching ratios of the LLP with a dynamic legend layout.

    Parameters:
    - masses_plot: Array of LLP masses.
    - Br_plot: 2D Array of branching ratios.
    - chosen_channels: List of decay channels to plot.
    - selected_decay_indices: List of indices corresponding to the chosen channels in Br_plot.
    - LLP: LLP object containing properties like LLP.LLP_name and LLP.uncertainty.
    - plot_folder: Directory to save the plot.
    """
    # Define fixed colors and line styles
    colors = [
    'green', 'goldenrod', 'red', 'blue', 'cyan', 'magenta', 'black',
    'deepskyblue', 'lightcoral', 'brown', 'peru', 'gray', 'lightgray',
    'teal', 'plum', 'orange', 'moccasin'
    ]
    line_styles = ['solid', 'dashed', 'dashdot']

    # Create a list of (color, linestyle) tuples
    color_style_combinations = [(color, style) for color in colors for style in line_styles]
    
    # Assign a unique color and style to each channel
    channel_styles = {}
    for i, channel in enumerate(chosen_channels):
        color, style = color_style_combinations[i % len(color_style_combinations)]
        channel_styles[channel] = {'color': color, 'linestyle': style}
    
    # Determine the number of channels and legend columns
    n_channels = len(chosen_channels)
    max_entries_per_col = 10  # Maximum legend entries per column
    n_legend_cols = math.ceil(n_channels / max_entries_per_col)
    n_legend_cols = min(n_legend_cols, 4)  # Cap the number of columns to 4
    
    # Define fixed plot dimensions
    plot_width = 9  # inches (3 units)
    plot_height = 6  # inches (2 units)
    legend_width_per_col = 0.5  # inches per legend column
    legend_width = legend_width_per_col * n_legend_cols  # Total legend width
    total_width = plot_width + legend_width + 1  # Additional inch for padding
    
    # Create the figure with dynamic width to accommodate legend
    fig = plt.figure(figsize=(total_width, plot_height))
    
    # Create a GridSpec with two columns: one for the plot and one for the legend
    gs = gridspec.GridSpec(1, 2, width_ratios=[plot_width, legend_width + 1], figure=fig)
    
    # Add the main plot axes
    ax = fig.add_subplot(gs[0])
    
    # Plot each decay channel with fixed color and line style
    for channel in chosen_channels:
        idx = chosen_channels.index(channel)
        decay_index = selected_decay_indices[idx]
        style = channel_styles[channel]
        ax.loglog(
            masses_plot,
            Br_plot[:, decay_index],
            label=channel,
            color=style['color'],
            linestyle=style['linestyle'],
            linewidth=1.5
        )
    
    # Set labels and title
    ax.set_xlabel(r"$m_{\mathrm{LLP}}\,[\mathrm{GeV}]$", fontsize=12)
    ax.set_ylabel(r"$\mathrm{Br}_{\mathrm{LLP}}\to X$", fontsize=12)
    ax.set_title(f"Decay Branching Ratios of {LLP.LLP_name}", fontsize=14)
    
    # Add mixing pattern or uncertainty information
    if LLP.LLP_name == "HNL":
        Ue2, Umu2, Utau2 = LLP.MixingPatternArray
        mixing_text = (f"Mixing pattern:\n"
                       f"$|U_e|^2$={Ue2:.3f}, "
                       f"$|U_\\mu|^2$={Umu2:.3f}, "
                       f"$|U_\\tau|^2$={Utau2:.3f}")
        ax.text(0.05, 0.95, mixing_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))
    elif LLP.LLP_name == "Dark-photons":
        uncertainty_text = f"The {LLP.uncertainty} position of production yield"
        ax.text(0.05, 0.95, uncertainty_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))
    
    # Add grid
    ax.grid(True, which="both", linestyle='--', linewidth=0.5)
    
    # Extract handles and labels for the legend
    handles, labels = ax.get_legend_handles_labels()
    
    # Add the legend axes
    legend_ax = fig.add_subplot(gs[1])
    legend_ax.axis('off')  # Hide the legend axes
    
    # Create the legend with dynamic number of columns
    legend = legend_ax.legend(
        handles, labels, loc='center left',
        ncol=n_legend_cols, fontsize=9
    )
    
    # Adjust layout to prevent overlap and maintain fixed plot aspect ratio
    plt.tight_layout()
    
    # Save the figure with tight bounding box to include all elements
    fig.savefig(os.path.join(plot_folder, "branching_ratios.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

