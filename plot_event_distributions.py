#!/usr/bin/env python3
# plot_event_distributions.py
"""
Two-panel figure for the paper: polar-angle and energy distributions of the
dark photons decaying inside the SHiP decay volume, for the primary and
secondary-proton-bremsstrahlung (brem-cascade) sources, two masses
(0.5 and 1 GeV) and a reference large lifetime (c*tau = 1000 m).

Each distribution carries the absolute normalization: the integral equals the
total number of decay events N_ev of the given (mass, lifetime) combination,
with the coupling fixed as epsilon^2 = c*tau_int / c*tau.
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from funcs.selecting_processing import (
    discover_output_files,
    filter_output_files,
    require_unique_output,
)

HERE = os.path.dirname(os.path.abspath(__file__))
# Figures are written inside the repository. Set EVENTCALC_FIGURE_DIR to write
# them somewhere else.
FIGDIR = os.environ.get(
    "EVENTCALC_FIGURE_DIR", os.path.join(HERE, "plots", "Dark-photons")
)
EVENT_DATA_DIR = os.path.join(HERE, "outputs", "Dark-photons", "eventData")

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 15, "legend.fontsize": 10.5,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "figure.dpi": 150,
})

CTAU = 1000.0
GENERATOR_TAG = os.environ.get("EVENTCALC_GENERATOR_TAG")
CHANNEL_TAG = os.environ.get("EVENTCALC_CHANNEL_TAG")
CONFIGS = [
    # (mass, label suffix, color, linestyle, legend)
    (0.5, "central", "royalblue", "--", r"primary, $m_V=0.5$ GeV"),
    (0.5, "brem-cascade-central", "crimson", "--", r"sec. brem, $m_V=0.5$ GeV"),
    (1.0, "central", "royalblue", "-", r"primary, $m_V=1$ GeV"),
    (1.0, "brem-cascade-central", "crimson", "-", r"sec. brem, $m_V=1$ GeV"),
]


def resolve_event_path(
        mass, label, c_tau=CTAU, event_data_dir=EVENT_DATA_DIR,
        generator_tag=GENERATOR_TAG, channel_tag=CHANNEL_TAG):
    """Resolve one legacy/tagged event file without guessing a variant."""
    records = discover_output_files(
        event_data_dir, "event", llp_name="Dark-photons"
    )
    matches = filter_output_files(
        records,
        physics_label=label,
        mass=mass,
        c_tau=c_tau,
        generator_tag=generator_tag,
        channel_tag=channel_tag,
    )
    return require_unique_output(
        matches,
        description=(
            f"Dark-photons '{label}' event at mass {mass:g} GeV and "
            f"c*tau {c_tau:g} m"
        ),
    ).path


def load_events(
        mass, label, c_tau=CTAU, event_data_dir=EVENT_DATA_DIR,
        generator_tag=GENERATOR_TAG, channel_tag=CHANNEL_TAG):
    path = resolve_event_path(
        mass,
        label,
        c_tau=c_tau,
        event_data_dir=event_data_dir,
        generator_tag=generator_tag,
        channel_tag=channel_tag,
    )
    with open(path) as f:
        header = f.readline()
        m = re.search(r"Total number of events:\s*([0-9.eE+-]+)", header)
        n_ev_tot = float(m.group(1))
        E, TH, W = [], [], []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vals = np.fromstring(line, sep=" ")
            if vals.size < 10:
                continue
            px, py, pz, En = vals[0], vals[1], vals[2], vals[3]
            p = np.sqrt(px * px + py * py + pz * pz)
            TH.append(np.arccos(pz / p))
            E.append(En)
            W.append(vals[6])          # P_decay
    E, TH, W = map(np.array, (E, TH, W))
    scale = n_ev_tot / W.sum()
    print(f"{os.path.basename(path)}: N_ev={n_ev_tot:.3e}, "
          f"sampled={len(E)}, <E>={np.average(E, weights=W):.1f} GeV")
    return E, TH, W * scale, n_ev_tot


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    fig, (ax_th, ax_e) = plt.subplots(1, 2, figsize=(11, 4.4))

    th_bins = np.geomspace(8e-4, 0.05, 34)
    # coarser bins at low energy: the near-threshold events carry large
    # decay-probability weights, so fine bins there are noisy
    e_bins = np.unique(np.concatenate([
        np.geomspace(0.6, 12.0, 9),
        np.geomspace(12.0, 400.0, 22),
    ]))

    for mass, label, color, ls, leg in CONFIGS:
        E, TH, W, n_ev = load_events(mass, label)
        h, _ = np.histogram(TH, bins=th_bins, weights=W)
        ax_th.stairs(h / np.diff(th_bins), th_bins, color=color, ls=ls, lw=2,
                     label=leg + rf" ($N_{{\rm ev}}={n_ev:.2g}$)")
        h, _ = np.histogram(E, bins=e_bins, weights=W)
        ax_e.stairs(h / np.diff(e_bins), e_bins, color=color, ls=ls, lw=2,
                    label=leg)

    ax_th.set_xscale("log")
    ax_th.set_yscale("log")
    ax_th.set_xlabel(r"$\theta_{V}$ [rad]")
    ax_th.set_ylabel(r"$dN_{\rm ev}/d\theta_{V}$ [rad$^{-1}$]")
    ax_th.grid(alpha=0.3, which="both")
    ax_th.legend(loc="lower right")

    ax_e.set_xscale("log")
    ax_e.set_yscale("log")
    ax_e.set_xlabel(r"$E_{V}$ [GeV]")
    ax_e.set_ylabel(r"$dN_{\rm ev}/dE_{V}$ [GeV$^{-1}$]")
    ax_e.set_ylim(1e-5, 0.7)
    ax_e.grid(alpha=0.3, which="both")
    ax_e.legend(loc="lower left")
    ax_th.set_ylim(5e-2, 2e3)

    fig.tight_layout()
    out = os.path.join(FIGDIR, "event-distributions-DP.pdf")
    fig.savefig(out)
    print("saved", out)


if __name__ == "__main__":
    main()
