#!/usr/bin/env python3
# analyze_cascade_vs_primary.py
"""
Comparison of a selected dark-photon cascade source against the standard
primary source in EventCalc. ``CASCADE_LABEL`` defaults to the full
``cascade`` source and may also select ``brem-cascade`` or ``combined``.

Combines:
  * the *_total.txt scan summaries (N_ev before daughter-level requirements),
  * the pointing_summary.csv from events_pointing_analysis.py (the
    daughter-level requirement: >= 2 decay products crossing the
    4 m x 6 m plane at z = 95 m).

Outputs:
  * cascade_vs_primary_table.csv (per mass: ratios, lifetimes at N_ev = 3),
  * figures: Nev-ratio-DP.pdf, Nev-vs-ctau-DP.pdf, dNev-dE-DP.pdf
    (saved into plots/Dark-photons, or into EVENTCALC_FIGURE_DIR when that
    environment variable is set).
"""

import os
import numpy as np
import pandas as pd
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
OUTPUT_DIR = os.path.join(HERE, "outputs", "Dark-photons")
TOTAL_DIR = os.path.join(OUTPUT_DIR, "total")
EVENT_DATA_DIR = os.path.join(OUTPUT_DIR, "eventData")

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 15, "legend.fontsize": 11,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "figure.dpi": 150,
})

MASSES = [0.05, 0.1, 0.3, 0.5, 0.775, 1.0, 1.022, 1.25, 1.5, 1.75, 2.0]
CTAUS = [10., 30., 100., 300., 1000., 3000., 10000., 30000.]
VARIANTS = ["lower", "central", "upper"]
CASCADE_LABEL = os.environ.get("CASCADE_LABEL", "cascade")
GENERATOR_TAG = os.environ.get("EVENTCALC_GENERATOR_TAG")
CHANNEL_TAG = os.environ.get("EVENTCALC_CHANNEL_TAG")


def resolve_total_path(
        label, total_dir=TOTAL_DIR, generator_tag=GENERATOR_TAG,
        channel_tag=CHANNEL_TAG):
    """Resolve one legacy/tagged total file without arbitrary selection."""
    records = discover_output_files(
        total_dir, "total", llp_name="Dark-photons"
    )
    matches = filter_output_files(
        records,
        physics_label=label,
        generator_tag=generator_tag,
        channel_tag=channel_tag,
    )
    return require_unique_output(
        matches, description=f"Dark-photons '{label}' total"
    ).path


def load_totals(
        label, total_dir=TOTAL_DIR, generator_tag=GENERATOR_TAG,
        channel_tag=CHANNEL_TAG):
    path = resolve_total_path(
        label,
        total_dir=total_dir,
        generator_tag=generator_tag,
        channel_tag=channel_tag,
    )
    return pd.read_csv(path, sep=r"\s+")


def load_pointing(path=None):
    if path is None:
        path = os.path.join(OUTPUT_DIR, "pointing_summary.csv")
    df = pd.read_csv(path)
    return df


def resolve_largest_lifetime_event(
        label, mass, event_data_dir=EVENT_DATA_DIR,
        generator_tag=GENERATOR_TAG, channel_tag=CHANNEL_TAG):
    """Resolve the largest-lifetime sample for one exact provenance variant."""
    records = discover_output_files(
        event_data_dir, "event", llp_name="Dark-photons"
    )
    matches = filter_output_files(
        records,
        physics_label=label,
        mass=mass,
        generator_tag=generator_tag,
        channel_tag=channel_tag,
    )
    if not matches:
        return require_unique_output(
            matches,
            description=(
                f"Dark-photons '{label}' event at mass {mass:g} GeV"
            ),
        ).path
    largest_c_tau = max(record.c_tau for record in matches)
    matches = filter_output_files(matches, c_tau=largest_c_tau)
    return require_unique_output(
        matches,
        description=(
            f"Dark-photons '{label}' event at mass {mass:g} GeV and "
            f"c*tau {largest_c_tau:g} m"
        ),
    ).path


def eps_point_lookup(pointing, label, mass, ctau):
    """Pointing efficiency for (label, mass, ctau); falls back to the largest
    tabulated lifetime for this mass (the event kinematics of surviving
    events converge in the long-lifetime limit)."""
    sub = pointing[(pointing.label == label) & (np.isclose(pointing.mass, mass))]
    if len(sub) == 0:
        return np.nan
    exact = sub[np.isclose(sub.c_tau, ctau)]
    if len(exact):
        return float(exact.eps_point.iloc[0])
    sub = sub.sort_values("c_tau")
    return float(sub.eps_point.iloc[-1])


def n_ev_with_pointing(totals, pointing, label, mass, ctau):
    row = totals[(np.isclose(totals.mass, mass)) & (np.isclose(totals.c_tau, ctau))]
    if len(row) == 0:
        return np.nan, np.nan
    n = float(row.N_ev_tot.iloc[0])
    eps = eps_point_lookup(pointing, label, mass, ctau)
    return n, n * eps


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    pointing = load_pointing()
    totals = {}
    for v in VARIANTS:
        totals[("cascade", v)] = load_totals(f"{CASCADE_LABEL}-{v}")
        totals[("primary", v)] = load_totals(v)

    # ---------------- per-mass summary at the N_ev = 3 lifetime -----------
    rows = []
    for m in MASSES:
        rec = {"mass": m}
        for v in VARIANTS:
            # N_ev(ctau) with pointing for both sources
            npv, ncv = [], []
            for ct in CTAUS:
                _, npt = n_ev_with_pointing(totals[("primary", v)], pointing, v, m, ct)
                _, nct = n_ev_with_pointing(
                    totals[("cascade", v)], pointing, f"{CASCADE_LABEL}-{v}", m, ct)
                npv.append(npt)
                ncv.append(nct)
            npv, ncv = np.array(npv), np.array(ncv)
            ok = np.isfinite(npv) & np.isfinite(ncv) & (npv > 0)
            # plateau ratio in the long-lifetime regime: average over the
            # three largest lifetimes with data
            if ok.sum() >= 1:
                idx = np.where(ok)[0][-3:]
                rec[f"ratio_point_{v}"] = float(np.mean(ncv[idx] / npv[idx]))
            # ratio without pointing
            npv0 = np.array([n_ev_with_pointing(totals[("primary", v)],
                                                pointing, v, m, ct)[0]
                             for ct in CTAUS])
            ncv0 = np.array([n_ev_with_pointing(totals[("cascade", v)],
                                                pointing, f"{CASCADE_LABEL}-{v}",
                                                m, ct)[0] for ct in CTAUS])
            ok0 = np.isfinite(npv0) & np.isfinite(ncv0) & (npv0 > 0)
            if ok0.sum() >= 1:
                idx = np.where(ok0)[0][-3:]
                rec[f"ratio_nopoint_{v}"] = float(np.mean(ncv0[idx] / npv0[idx]))
            # lifetime at which the total (primary + cascade) pointing-level
            # N_ev crosses 3 on the long-lifetime branch
            tot = npv + ncv
            okt = np.isfinite(tot) & (tot > 0)
            if okt.sum() >= 2:
                ct_arr = np.array(CTAUS)[okt]
                t_arr = tot[okt]
                # long-lifetime branch: N decreases with ctau
                dec = np.where(np.diff(t_arr) < 0)[0]
                if len(dec) and t_arr.max() > 3 > t_arr.min():
                    lo = np.searchsorted(-t_arr, -3.0)
                    i1, i0 = min(lo, len(t_arr) - 1), max(lo - 1, 0)
                    if t_arr[i0] != t_arr[i1]:
                        logct = np.interp(np.log(3.0),
                                          [np.log(t_arr[i1]), np.log(t_arr[i0])],
                                          [np.log(ct_arr[i1]), np.log(ct_arr[i0])])
                        rec[f"ctau3_{v}"] = float(np.exp(logct))
        # pointing efficiencies at the largest available lifetime (central)
        rec["eps_point_prim"] = eps_point_lookup(pointing, "central", m, 1e9)
        rec["eps_point_casc"] = eps_point_lookup(pointing, f"{CASCADE_LABEL}-central", m, 1e9)
        rows.append(rec)

    table = pd.DataFrame(rows)
    table.to_csv(
        os.path.join(OUTPUT_DIR, "cascade_vs_primary_table.csv"), index=False
    )
    print(table.to_string(index=False))

    # ---------------- figure: ratio vs mass ------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    mm = table.mass.values
    ax.axhline(1, color="gray", lw=1, ls=":")
    ax.fill_between(mm, table.ratio_point_lower, table.ratio_point_upper,
                    color="crimson", alpha=0.18, lw=0)
    ax.plot(mm, table.ratio_point_central, "o-", color="crimson", lw=2,
            label="with daughter pointing ($z=95$ m plane)")
    ax.plot(mm, table.ratio_nopoint_central, "s--", color="royalblue", lw=2,
            label="decay-in-volume only")
    ax.set_xlabel(r"$m_{V}$ [GeV]")
    ax.set_ylabel(r"$N_{\rm ev}^{\rm cascade}/N_{\rm ev}^{\rm primary}$")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "Nev-ratio-DP.pdf"))
    plt.close(fig)

    # ---------------- figure: N_ev vs ctau for m = 1 GeV ------------------
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    m = 1.0
    for src, color, lab in [("primary", "royalblue", "primary (all channels)"),
                            ("cascade", "crimson", "secondary (all channels)")]:
        for v, ls, alpha in [("central", "-", 1.0)]:
            lbl = v if src == "primary" else f"{CASCADE_LABEL}-{v}"
            ns, ns_pt = [], []
            for ct in CTAUS:
                n, npt = n_ev_with_pointing(totals[(src, v)], pointing,
                                            lbl, m, ct)
                ns.append(n)
                ns_pt.append(npt)
            ax.plot(CTAUS, ns, ls="--", color=color, alpha=0.55)
            ax.plot(CTAUS, ns_pt, ls="-", color=color, lw=2, label=lab)
    ax.axhline(3, color="gray", lw=1.2, ls=":")
    ax.text(12, 3.4, r"$N_{\rm ev}=3$", color="gray", fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$c\tau_{V}$ [m]")
    ax.set_ylabel(r"$N_{\rm ev}$ (with $\epsilon^2 = c\tau_{\rm int}/c\tau$)")
    ax.set_title(r"$m_V = 1$ GeV; dashed: before daughter pointing",
                 fontsize=12)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "Nev-vs-ctau-DP.pdf"))
    plt.close(fig)

    # ---------------- figure: event energy distributions ------------------
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    for lbl, color, name in [("central", "royalblue", "primary (all channels)"),
                             (f"{CASCADE_LABEL}-central", "crimson",
                              "secondary protons (cascade)")]:
        # largest lifetime with an event file at m = 1 GeV
        try:
            path = resolve_largest_lifetime_event(lbl, 1.0)
        except FileNotFoundError:
            continue
        E, W, Epass, Wpass = [], [], [], []
        import events_pointing_analysis as epa
        n_ev_tot, events = epa.parse_event_file(path)
        # header energies: mother E is column 3 of the mother block
        with open(path) as f:
            f.readline()
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                vals = np.fromstring(line, sep=" ")
                if vals.size < 16:
                    continue
                E.append(vals[3])
                W.append(vals[6])
                plist = vals[10:].reshape(-1, 6)
                if epa.pointing_pass(vals[7], vals[8], vals[9], plist):
                    Epass.append(vals[3])
                    Wpass.append(vals[6])
        E, W = np.array(E), np.array(W)
        Epass, Wpass = np.array(Epass), np.array(Wpass)
        # coarser bins at low energy: near-threshold events carry large
        # decay-probability weights, so fine bins there are noisy
        bins = np.unique(np.concatenate([np.geomspace(1.0, 12.0, 8),
                                         np.geomspace(12.0, 400.0, 26)]))
        # normalize to the absolute number of events: N_ev_tot spread over
        # the sample
        scale = n_ev_tot / W.sum() if W.sum() > 0 else 0.0
        h, _ = np.histogram(E, bins=bins, weights=W * scale)
        hp, _ = np.histogram(Epass, bins=bins, weights=Wpass * scale)
        ax.stairs(h / np.diff(bins), bins, color=color, ls="--", alpha=0.55)
        ax.stairs(hp / np.diff(bins), bins, color=color, lw=2, label=name)
        print(f"{lbl}: file={os.path.basename(path)}  <E>={np.average(E, weights=W):.1f} "
              f"-> pass <E>={np.average(Epass, weights=Wpass):.1f} GeV")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-6, 0.2)
    ax.set_xlabel(r"$E_{V}$ [GeV]")
    ax.set_ylabel(r"$dN_{\rm ev}/dE_{V}$ [GeV$^{-1}$]")
    ax.set_title(r"$m_V=1$ GeV; dashed: before daughter pointing", fontsize=12)
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "dNev-dE-DP.pdf"))
    plt.close(fig)
    print("figures saved")


if __name__ == "__main__":
    main()
