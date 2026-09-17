#!/usr/bin/env python3
"""
run_batch.py -- non-interactive EventCalc driver for parameter scans.

Runs exactly the same generation loop as simulate.py -- it calls into it -- but
takes every choice (LLP, mixing pattern, mass grid, lifetime grid, decay
channels, and the hadronic-decay generator) from the command line instead of
interactive prompts. This makes mass/lifetime scans scriptable, reproducible,
and HPC/batch friendly.

Outputs land in the repository's own outputs/ directory, wherever the command
was started from.

Examples
--------
# HNL, pure electron mixing, all jet-labelled rows, exhad ON, mass scan:
python3 run_batch.py --llp HNL --mixing 1 0 0 \
    --mass-range 1.0 3.0 0.5 --ctaus 10 \
    --channels jets --nevents 100000 --exhad on

# Dark photon, central flux, primary production, all channels, log lifetime
# grid, exhad AUTO (card):
python3 run_batch.py --llp Dark-photons --masses 2.0 2.5 \
    --uncertainty central --dp-production primary \
    --ctau-logrange 0.1 100 5 --nevents 50000

# Higgs-like scalar (mixing), raw Pythia (exhad OFF) for the hadronic decays:
python3 run_batch.py --llp Scalar-mixing --masses 2.2 --ctaus 5 \
    --nevents 50000 --exhad off
"""

import argparse
from dataclasses import replace
import math
import os
import sys

import numpy as np

# Resolve to the EventCalc directory so this script runs from anywhere.  The
# working directory is never changed: every path the pipeline writes is
# resolved against the project root instead.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import simulate
from simulate import DEFAULT_SEED, run_mass_lifetime_grid
from funcs.channel_selection import selected_channel_labels
from funcs.LLP_selection import (
    N_pot,
    normalize_hnl_mixing,
    validate_event_count,
    validate_masses_in_domain,
    validate_positive_finite,
)
from funcs.output_provenance import effective_generator_identity
from funcs.simulation_config import (
    DEFAULT_SCALAR_PRESCRIPTION,
    MODEL_SPECS,
    PROJECT_ROOT,
    SCALAR_PRESCRIPTIONS,
    ConfigurationError,
    add_hadronization_arguments,
    config_from_mapping,
    resolve_decay_channels,
    _EXHAD_INAPPLICABLE_MODEL_NAMES,
)

# Heavy pipeline imports (numpy/pandas + the EventCalc funcs) are done lazily
# inside main(), so --help and the pure grid/channel logic are importable
# without the full environment.

N_POT = N_pot  # protons on target; the one statement is funcs/LLP_selection.py
LLP_CHOICES = [spec.name for spec in MODEL_SPECS]


# --------------------------------------------------------------------------
# grid builders
# --------------------------------------------------------------------------
def _inclusive_linear_grid(lo, hi, step):
    """Build ``LO + k*STEP`` points without stepping beyond inclusive ``HI``."""
    span = (hi - lo) / step
    nearest = round(span)
    if math.isclose(span, nearest, rel_tol=1.0e-12, abs_tol=1.0e-12):
        steps = int(nearest)
        endpoint_is_grid_point = True
    else:
        steps = int(math.floor(span))
        endpoint_is_grid_point = False
    values = lo + step * np.arange(steps + 1, dtype=float)
    if endpoint_is_grid_point:
        values[-1] = hi
    return list(np.round(values, 10))


def build_masses(args):
    if args.masses is not None:
        return validate_positive_finite(args.masses, "masses")
    lo, hi, step = args.mass_range
    lo, hi, step = validate_positive_finite(
        (lo, hi, step), "--mass-range LO, HI, and STEP")
    if hi < lo:
        raise ValueError("--mass-range HI must be greater than or equal to LO")
    masses = _inclusive_linear_grid(lo, hi, step)
    return validate_positive_finite(masses, "masses")


def build_ctaus(args):
    """Return ONE list of lifetimes, applied to every mass (a full grid)."""
    if args.ctaus is not None:
        return validate_positive_finite(args.ctaus, "lifetimes")
    if args.ctau_range is not None:
        lo, hi, step = args.ctau_range
        lo, hi, step = validate_positive_finite(
            (lo, hi, step), "--ctau-range LO, HI, and STEP")
        if hi < lo:
            raise ValueError(
                "--ctau-range HI must be greater than or equal to LO")
        c_taus = _inclusive_linear_grid(lo, hi, step)
        return validate_positive_finite(c_taus, "lifetimes")
    lo, hi, n_raw = args.ctau_logrange
    lo, hi = validate_positive_finite(
        (lo, hi), "--ctau-logrange LO and HI")
    if hi < lo:
        raise ValueError(
            "--ctau-logrange HI must be greater than or equal to LO")
    if not np.isfinite(n_raw) or n_raw < 1 or not float(n_raw).is_integer():
        raise ValueError("--ctau-logrange N must be a positive integer")
    n = int(n_raw)
    return validate_positive_finite(
        np.logspace(np.log10(lo), np.log10(hi), n), "lifetimes")


def resolve_channels(tokens, decay_channels):
    """Resolve collapsed user choices to raw decay-table indices."""
    try:
        return resolve_decay_channels(list(tokens), list(decay_channels))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def validate_stock_pythia_pool_request(
        enabled, llp_name, requested_exhad, channel_tokens, masses):
    """Reject configurations that would make the ALP comparator ambiguous."""
    if not enabled:
        return
    if llp_name != "ALP-fermion":
        raise ValueError(
            "--stock-pythia-pool is defined only for --llp ALP-fermion"
        )
    if requested_exhad != "off":
        raise ValueError(
            "--stock-pythia-pool requires the explicit option --exhad off"
        )
    if list(channel_tokens) != ["all"]:
        raise ValueError(
            "--stock-pythia-pool requires exactly --channels all so the "
            "complete hadronic allocation is retained"
        )
    unsupported = [
        float(mass) for mass in masses
        if float(mass) < 1.911 - 1.0e-12
    ]
    if unsupported:
        raise ValueError(
            "--stock-pythia-pool starts at 1.911 GeV; unsupported masses: "
            + ", ".join("%g" % mass for mass in unsupported)
        )


def resolve_exhad_selection(
        exhad_module, requested_mode, llp_name, applicable=True):
    """Resolve the batch generator without an implicit ``on -> raw`` path.

    ``auto`` may use the EventCalc baseline when the optional bridge is simply
    absent.  An explicit applicable ``on`` request is fail-closed; a selection
    with no possible matched hadronic channel is an explicitly reported no-op.
    Exceptions from card/variant resolution deliberately propagate in either
    mode, since silently pairing (for example) a lower scalar BR table with
    the central exhad benchmark is not a valid fallback.
    """
    if requested_mode == "off":
        exhad_module.set_force(False)
        return None, "disabled; EventCalc baseline is forced"
    if not applicable:
        exhad_module.set_force(False)
        return None, "not applicable (no selected matched hadronic channel)"
    exhad_module.set_force(True if requested_mode == "on" else None)
    if exhad_module.can_use_exhad():
        bench = exhad_module.get_bench()
        return bench, "available benchmark %s" % bench
    if requested_mode == "on":
        raise RuntimeError(
            "--exhad on was requested, but no usable exhad "
            "benchmark/bridge resolved for %s" % llp_name)
    return None, (
        "no usable card/bridge for %s; EventCalc baseline" % llp_name
    )


def exhad_applicable_to_selection(
        llp_name, decay_products, PDGdecay, selected_decay_indices):
    """Whether an explicit exhad request has any possible selected target.

    The photon-coupled ALP currently has only gamma/lepton final states and no
    matched hadronic card.  Treating ``--exhad on`` as a no-op for that
    selection is more accurate than requiring an irrelevant bridge.  Other
    installed portals retain fail-closed card/bridge validation because their
    matched backends can own non-partonic explicit rows as well as Jets rows.
    """
    if llp_name not in _EXHAD_INAPPLICABLE_MODEL_NAMES:
        return True
    return any(
        decay_products.channel_is_partonic(PDGdecay[table_index])
        for table_index in selected_decay_indices
    )


def enforce_hnl_matched_mass_scope(exhad_module, llp_name,
                                   resolved_exhad_bench, masses):
    """Apply the HNL parent-mass gate before any matched output is tagged."""
    if llp_name == "HNL" and resolved_exhad_bench == "hnl":
        exhad_module.require_hnl_parent_masses_supported(masses)


def get_or_create_fixed_exhad_pool(cache, mass, decay_products, **kwargs):
    """Lazily build one fixed-mass exhad pool and cache even ``None``.

    The helper is intentionally small and dependency-injected so scan reuse
    can be tested without importing the heavy EventCalc pipeline.  Caching a
    ``None`` result also avoids repeatedly probing an inapplicable raw/HNL
    configuration at every lifetime.
    """
    return simulate._get_or_create_fixed_exhad_pool(
        cache, mass, decay_products, **kwargs)


def scan_batch_indices(local_mass_index, local_ctau_index, n_ctaus,
                       batch_index_offset=0):
    """Return global fixed-mass and mass/lifetime RNG coordinates.

    ``batch_index_offset`` counts mass-grid entries that precede the current
    command.  Supplying the same offset when resuming a split mass scan makes
    its first local mass use exactly the RNG coordinates it had in the
    original ordered scan.  The lifetime ordering must be unchanged.  The
    indices are zero-based here; the arithmetic is the scan's own, in
    ``simulate``.
    """
    return simulate._scan_batch_indices(
        int(local_mass_index) + 1, int(local_ctau_index) + 1, n_ctaus,
        batch_index_offset)


# --------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Non-interactive EventCalc generator for scans.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    ap.add_argument("--llp", required=True, choices=LLP_CHOICES,
                    help="LLP to simulate (a Distributions/<LLP> folder).")
    ap.add_argument("--nevents", type=int, required=True,
                    help="events to sample per (mass, lifetime) point "
                         "(= interactive 'number of events').")

    gm = ap.add_mutually_exclusive_group(required=True)
    gm.add_argument("--masses", type=float, nargs="+",
                    help="explicit mass list in GeV.")
    gm.add_argument("--mass-range", type=float, nargs=3,
                    metavar=("LO", "HI", "STEP"),
                    help="linear mass grid LO..HI inclusive, step STEP.")

    gc = ap.add_mutually_exclusive_group(required=True)
    gc.add_argument("--ctaus", type=float, nargs="+",
                    help="explicit lifetime list c*tau in m (applied to every "
                         "mass).")
    gc.add_argument("--ctau-range", type=float, nargs=3,
                    metavar=("LO", "HI", "STEP"), help="linear lifetime grid.")
    gc.add_argument("--ctau-logrange", type=float, nargs=3,
                    metavar=("LO", "HI", "N"),
                    help="log-spaced lifetime grid, N points LO..HI.")

    ap.add_argument("--channels", "--decay-channels", "--decays",
                    dest="channels", nargs="+", default=["all"],
                    help="decay channels: non-jet names, 'jets' for every "
                         "Jets-* row, or 'all' (default).")
    add_hadronization_arguments(ap)
    ap.add_argument(
        "--stock-pythia-pool", action="store_true",
        help="ALP-fermion comparator above 1.911 GeV: retain the complete "
             "hadronic event allocation, discard the explicit hadronic "
             "rows, and hadronize the whole pool with unmodified "
             "Pythia 8.317. Requires --channels all --exhad off.")
    ap.add_argument("--llp-pdg", dest="llp_pdg", type=int, default=None, metavar="CODE",
                    help="override the mother PDG code (default: the exHad release "
                         "default -- dark-photon 4900022, b-l 32, scalars 35, "
                         "alp-fermion 36, hnl 9900012). EventCalc's internal "
                         "hadronization mother code 25 is unaffected.")
    ap.add_argument("--exhad-seed", type=int, default=DEFAULT_SEED,
                    help="base seed for EventCalc phase space, Pythia, exhad, "
                         "and pi0/KS-decay streams; fixed-mass pools "
                         "and scan channels receive deterministic derived "
                         "seeds (default: %d)." % DEFAULT_SEED)
    ap.add_argument(
        "--batch-index-offset", type=int, default=0,
        help="number of mass-grid entries preceding this command in an "
             "original ordered scan. Use when splitting/resuming a mass "
             "scan so upstream and exhad RNG streams retain their original "
             "indices; keep the lifetime list and order unchanged "
             "(default: 0).")

    # LLP-specific
    ap.add_argument("--mixing", type=float, nargs=3,
                    metavar=("Ue2", "Umu2", "Utau2"),
                    help="HNL mixing pattern (renormalized to 1). Required "
                         "for --llp HNL.")
    ap.add_argument("--uncertainty", choices=["lower", "central", "upper"],
                    default=None, help="Dark-photon flux variation. Required "
                                       "for --llp Dark-photons.")
    ap.add_argument("--dp-production",
                    choices=[
                        "primary", "cascade", "brem-cascade", "combined",
                        "primary+cascade", "cascade+primary",
                    ],
                    default=None, help="Dark-photon production source. "
                                      "Required for --llp Dark-photons.")
    ap.add_argument("--alp-production", choices=[
                        "primary", "cascade", "cascades", "combined",
                        "primary+cascade", "cascade+primary",
                    ],
                    default=None, help="ALP-photon distribution source. "
                                      "Required for --llp ALP-photon.")
    ap.add_argument("--scalar-lifetime", default=None,
                    choices=list(SCALAR_PRESCRIPTIONS),
                    help="Scalar decay-table prescription (default for the "
                         "scalar models: %s)." % DEFAULT_SCALAR_PRESCRIPTION)
    ap.add_argument("--xi", type=float, default=None,
                    help="ALP-mixed SU(2)_L operator fraction in [0, 1].")
    ap.add_argument("--interference",
                    choices=["constructive", "destructive"], default=None,
                    help="ALP-mixed relative sign of the direct and induced "
                         "diphoton amplitudes.")

    ap.add_argument("--plots", action="store_true",
                    help="also emit the phenomenology plots simulate.py makes.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved configuration and exit.")
    return ap, ap.parse_args()


def _card_from_arguments(args, masses, c_taus, exhad_mode):
    """The launch fields this command names, in launch-card form."""
    name = args.llp
    card = {
        "model": name,
        "events": args.nevents,
        "masses": masses,
        "c_taus": c_taus,
        "decay_channels": list(args.channels),
        "exhad_mode": exhad_mode,
        "seed": args.exhad_seed,
        "n_pot": N_POT,
        "plots": bool(args.plots),
        "export_events": True,
        "llp_pdg": args.llp_pdg,
        "batch_index_offset": args.batch_index_offset,
        "stock_pythia_pool": bool(args.stock_pythia_pool),
    }
    if name == "HNL":
        card["mixing_pattern"] = list(normalize_hnl_mixing(args.mixing))
    if name == "Dark-photons":
        card["uncertainty"] = args.uncertainty
        card["dp_production_mode"] = args.dp_production
    if name == "ALP-photon":
        card["alp_production_mode"] = args.alp_production
    if "Scalar" in name and args.scalar_lifetime is not None:
        card["scalar_prescription"] = args.scalar_lifetime
    if name == "ALP-mixed":
        card["xi"] = args.xi
        card["interference"] = args.interference
    return card


def main():
    ap, args = parse_args()

    name = args.llp
    exhad_mode = "auto" if args.exhad_mode is None else args.exhad_mode
    try:
        validate_event_count(args.nevents)
        masses = build_masses(args)
        c_taus = build_ctaus(args)
    except ValueError as exc:
        ap.error(str(exc))

    if name == "HNL" and args.mixing is None:
        ap.error("--mixing Ue2 Umu2 Utau2 is required for --llp HNL")
    if args.exhad_seed < 0:
        ap.error("--exhad-seed must be a non-negative integer")
    if args.batch_index_offset < 0:
        ap.error("--batch-index-offset must be a non-negative integer")

    try:
        validate_stock_pythia_pool_request(
            args.stock_pythia_pool, name, exhad_mode, args.channels, masses)
        config = config_from_mapping(
            _card_from_arguments(args, masses, c_taus, exhad_mode),
            project_root=PROJECT_ROOT)
    except (ValueError, ConfigurationError) as exc:
        ap.error(str(exc))

    runtime = simulate._load_runtime(headless=True)
    decayProducts = runtime.decayProducts
    exhad = runtime.exhadDecays

    # ---- build the LLP object (no prompts) and bind the generator ----
    try:
        LLP = simulate._make_llp(runtime, config)
    except Exception as exc:
        # An explicit exhad run must never become a baseline run silently.
        # Configuration/alias errors are also fatal in auto mode: using a
        # central scalar benchmark with a noncentral BR table would be a
        # physically inconsistent fallback.
        raise SystemExit("exhad configuration failed: %s" % exc) from exc

    try:
        masses = validate_masses_in_domain(
            config.masses, LLP.m_min_tabulated, LLP.m_max_tabulated)
    except ValueError as exc:
        ap.error(str(exc))
    config = replace(config, masses=tuple(masses))

    selected_decay_indices = resolve_channels(
        args.channels, list(LLP.decayChannels))
    channel_tag = runtime.mergeResults.channel_selection_tag(
        LLP.decayChannels, selected_decay_indices)

    try:
        exhad_applicable = exhad_applicable_to_selection(
            name, decayProducts, LLP.PDGs, selected_decay_indices)
        resolved_exhad_bench, exhad_status = resolve_exhad_selection(
            exhad, exhad_mode, name, applicable=exhad_applicable)
        if resolved_exhad_bench is None and getattr(
                LLP, "_exhad_binding", None) is not None:
            config = replace(config, exhad_mode="off")
            exhad.configure_llp(LLP, config)
        enforce_hnl_matched_mass_scope(
            exhad, name, resolved_exhad_bench, config.masses)
    except Exception as exc:
        raise SystemExit("exhad configuration failed: %s" % exc) from exc
    if args.stock_pythia_pool:
        exhad_status = (
            "stock Pythia 8.317 applied to the complete ALP hadronic pool"
        )

    sel_names = selected_channel_labels(
        selected_decay_indices, LLP.decayChannels)
    print("\n=== run_batch configuration ===")
    print("LLP              :", name)
    if config.mixing_pattern is not None:
        print("mixing (Ue,Umu,Utau2):", list(np.round(config.mixing_pattern, 6)))
    if name == "Dark-photons":
        print("DP production    :", LLP.dp_production_mode)
        print("DP uncertainty   :", LLP.uncertainty)
    if "Scalar" in name:
        print("prescription     :", LLP.scalar_lifetime)
    print("masses [GeV]     :", list(config.masses))
    print("lifetimes [m]    :", [float(np.round(c, 6)) for c in c_taus],
          "(applied to every mass)")
    print("channels         :", sel_names)
    print("events/point     :", config.events)
    print("exhad policy/card:", exhad_mode, "-", exhad_status)
    print("exhad base seed   :", config.seed)
    print("batch index offset:", config.batch_index_offset,
          "(preceding mass-grid entries)")
    print("channel tag       :", channel_tag)
    print("points to run    :", len(config.masses) * len(c_taus))
    print("================================")
    print("Every random stream is derived from the base seed, so repeating "
          "this command reproduces this run exactly.\n")
    if args.dry_run:
        print("Effective per-mass routes:")
        for mass in config.masses:
            LLP.set_mass(mass)
            LLP.compute_mass_dependent_properties()
            _, generator_tag, _ = effective_generator_identity(
                decayProducts,
                LLP,
                selected_decay_indices,
                config.seed,
                stock_pythia_pool=config.stock_pythia_pool,
            )
            br_visible_val = sum(
                LLP.BrRatios_distr[index]
                for index in selected_decay_indices
            )
            suffix = (
                " (no positive selected branching ratio)"
                if br_visible_val == 0
                else ""
            )
            print(f"  {float(mass):g} GeV: {generator_tag}{suffix}")
        return

    # ---- optional phenomenology plots (as in simulate.py) ----
    if args.plots:
        simulate._generate_phenomenology_plots(
            runtime, LLP, selected_decay_indices)

    # ---- the scan: one loop, in simulate.py ----
    # Fixed-mass exhad events do not depend on c_tau, so every lifetime at one
    # mass reuses a prefix of the same pool.
    fixed_exhad_pools = {}
    run_mass_lifetime_grid(
        runtime, config, LLP, selected_decay_indices,
        batch_index_offset=config.batch_index_offset,
        fixed_exhad_pools=fixed_exhad_pools)


if __name__ == "__main__":
    main()
