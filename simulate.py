#!/usr/bin/env python3
"""EventCalc's interactive and non-interactive simulation launcher.

The mass/lifetime loop lives here once.  ``run_batch.py`` builds its grids from
command-line flags and then runs the same loop through
:func:`run_mass_lifetime_grid`, so an interactive run and a batch run of the
same point produce the same events.
"""

from __future__ import annotations

import json
import os
import sys
import time
from types import SimpleNamespace
from typing import Sequence

from funcs.simulation_config import (
    PROJECT_ROOT,
    SimulationConfig,
    config_from_command_line,
    config_from_mapping,
    load_decay_channel_names,
)

#: Run seed used when a card or command line names none, so that every run is
#: reproducible from the seed the driver reports.
DEFAULT_SEED = 1

#: The channel coordinate every scan point's seed is derived under, so that a
#: point's seed and the exHad call seeds inside it share one stream family.
_SCAN_POINT_CHANNEL = 0x45564341


def _effective_seed(config: SimulationConfig) -> int:
    """The seed every random stream of this run is derived from."""
    return DEFAULT_SEED if config.seed is None else config.seed


def _interactive_seed() -> int:
    """The run seed of an interactive session.

    An interactive session has no command line to carry ``--seed`` and no card
    to carry ``seed``, so the seed comes from ``EXHAD_SEED`` in the
    environment, and from ``DEFAULT_SEED`` when that is unset.
    """
    raw = os.environ.get("EXHAD_SEED")
    if raw is None:
        return DEFAULT_SEED
    try:
        seed = int(raw)
    except ValueError as exc:
        raise ValueError("EXHAD_SEED must be a non-negative integer") from exc
    if seed < 0:
        raise ValueError("EXHAD_SEED must be a non-negative integer")
    return seed


def _load_runtime(*, headless: bool) -> SimpleNamespace:
    # Preserve a headless plotting default without importing Matplotlib during
    # numerical runtime initialization.
    if headless:
        os.environ["MPLBACKEND"] = "Agg"

    import numpy as np

    from funcs import (
        boost,
        coupling_conventions,
        decayProducts,
        exhadDecays,
        initLLP,
        kinematics,
        mergeResults,
        output_provenance,
        seeding,
        thresholds,
    )
    from funcs.ship_setup import (
        Delta_x_in,
        Delta_x_out,
        Delta_y_in,
        Delta_y_out,
        theta_max_dec_vol,
        z_max,
        z_min,
    )

    return SimpleNamespace(
        np=np,
        boost=boost,
        coupling_conventions=coupling_conventions,
        decayProducts=decayProducts,
        exhadDecays=exhadDecays,
        initLLP=initLLP,
        kinematics=kinematics,
        mergeResults=mergeResults,
        output_provenance=output_provenance,
        seeding=seeding,
        thresholds=thresholds,
        z_min=z_min,
        z_max=z_max,
        Delta_x_in=Delta_x_in,
        Delta_x_out=Delta_x_out,
        Delta_y_in=Delta_y_in,
        Delta_y_out=Delta_y_out,
        theta_max_dec_vol=theta_max_dec_vol,
    )


def _print_ship_setup(runtime: SimpleNamespace) -> None:
    print("\nSHiP setup (modify ship_setup.py if needed):\n")
    print(
        f"z_min={runtime.z_min} m, z_max={runtime.z_max} m, "
        f"Δx_in={runtime.Delta_x_in} m, Δx_out={runtime.Delta_x_out} m, "
        f"Δy_in={runtime.Delta_y_in} m, Δy_out={runtime.Delta_y_out} m, "
        f"θ_max_dec_vol={runtime.theta_max_dec_vol:.6f} rad\n"
    )


def _make_llp(runtime: SimpleNamespace, config):
    """Build the selected LLP and bind it to the configured exHad release."""
    mixing = getattr(config, "mixing_pattern", None)
    mixing = (
        runtime.np.asarray(mixing, dtype=float) if mixing is not None else None
    )
    llp = runtime.initLLP.LLP(
        mass=None,
        particle_selection=config.particle_selection,
        mixing_pattern=mixing,
        uncertainty=getattr(config, "uncertainty", None),
        alp_production_mode=getattr(config, "alp_production_mode", None),
        dp_production_mode=getattr(config, "dp_production_mode", None),
        scalar_lifetime=getattr(config, "scalar_prescription", None),
        xi=getattr(config, "xi", None),
        interference=getattr(config, "interference", None),
    )

    exhad = runtime.exhadDecays
    mode = getattr(config, "exhad_mode", "auto")
    exhad.set_selection(llp.LLP_name, llp.particle_path, llp.scalar_lifetime)
    exhad.set_force({"on": True, "off": False, "auto": None}[mode])
    exhad.set_mother_pdg(getattr(config, "llp_pdg", None))
    llp.LLP_pdg = exhad.mother_pdg()
    exhad.configure_llp(llp, config)
    masses = getattr(config, "masses", None)
    if masses and exhad.can_use_exhad() and exhad.is_hnl_bench():
        exhad.require_hnl_parent_masses_supported(masses)
    return llp


def _generate_phenomenology_plots(
    runtime: SimpleNamespace,
    llp,
    selected_decay_indices: Sequence[int],
) -> None:
    from funcs.plot_phenomenology import (
        plot_branching_ratios,
        plot_lifetime,
        plot_production_probability,
    )

    print("\nGenerating LLP phenomenology plots...")
    masses_plot = runtime.np.geomspace(llp.m_min_tabulated, llp.m_max_tabulated, 250)
    yield_plot = runtime.np.array([llp.get_total_yield(mass) for mass in masses_plot])
    ctau_plot = runtime.np.array([llp.get_ctau(mass) for mass in masses_plot])
    branching_plot = runtime.np.vstack([llp.get_Br(mass) for mass in masses_plot])
    plot_folder = os.path.join(
        str(PROJECT_ROOT), "plots", llp.LLP_name, "phenomenology")
    if llp.LLP_name == "ALP-photon":
        plot_folder = f"{plot_folder}_{llp.alp_production_mode}"
    elif llp.LLP_name == "ALP-mixed":
        plot_folder = f"{plot_folder}_xi{llp.xi:g}_{llp.interference}"
    elif (llp.LLP_name == "Dark-photons"
          and getattr(llp, "dp_production_mode", "primary") != "primary"):
        plot_folder = f"{plot_folder}_{llp.dp_production_mode}"
    os.makedirs(plot_folder, exist_ok=True)
    plot_production_probability(masses_plot, yield_plot, llp, plot_folder)
    plot_lifetime(masses_plot, ctau_plot, llp, plot_folder)
    plot_branching_ratios(
        masses_plot,
        branching_plot,
        [llp.decayChannels[index] for index in selected_decay_indices],
        list(selected_decay_indices),
        llp,
        plot_folder,
    )
    print("Phenomenology plots generated.")


def _mass_is_tabulated(mass: float, llp) -> bool:
    """Return whether ``mass`` lies in the closed tabulated mass interval."""
    return llp.m_min_tabulated <= mass <= llp.m_max_tabulated


def _scan_batch_indices(mass_index, ctau_index, n_ctaus, batch_index_offset=0):
    """Return the global fixed-mass and point coordinates of one scan point.

    ``mass_index`` and ``ctau_index`` are one-based positions in this command's
    own grids.  ``batch_index_offset`` counts mass-grid entries that precede
    this command in an ordered scan, so a split scan reuses exactly the
    coordinates its points had in the whole one.
    """
    mass_index = int(mass_index)
    ctau_index = int(ctau_index)
    n_ctaus = int(n_ctaus)
    batch_index_offset = int(batch_index_offset)
    if (mass_index < 1 or ctau_index < 1 or n_ctaus < 1
            or ctau_index > n_ctaus or batch_index_offset < 0):
        raise ValueError("invalid non-negative scan batch coordinates")
    fixed_mass_batch_index = batch_index_offset + mass_index - 1
    point_batch_index = fixed_mass_batch_index * n_ctaus + (ctau_index - 1)
    return fixed_mass_batch_index, point_batch_index


def scan_point_seed(base_seed, mass_index, ctau_index, n_ctaus,
                    batch_index_offset=0):
    """Seed every stream of one scan point and return its coordinates.

    Returns ``(fixed_mass_batch_index, point_batch_index, applied_seed)``.  The
    point is seeded before its first draw, which is the mother sample, so a
    point's events depend on the run seed and on its position in the scan alone.
    One seed reaches every stream: NumPy's legacy global generator, the RNG
    Numba compiles into the njit samplers, and, through the NumPy stream,
    Pythia's own seed.  exHad's generators take their seed explicitly.
    """
    from funcs import decayProducts, seeding

    fixed_mass_batch_index, point_batch_index = _scan_batch_indices(
        mass_index, ctau_index, n_ctaus, batch_index_offset)
    applied = seeding.seed_all(decayProducts.derive_exhad_seed(
        base_seed, point_batch_index, _SCAN_POINT_CHANNEL))
    return fixed_mass_batch_index, point_batch_index, applied


def _get_or_create_fixed_exhad_pool(cache, mass, decay_products, **kwargs):
    """Lazily build one fixed-mass exHad pool and cache even ``None``.

    The rest-frame decay does not depend on the lifetime, so every lifetime at
    one mass reuses a prefix of one pool.  Caching a ``None`` result also stops
    an inapplicable raw or HNL configuration being probed at every lifetime.
    """
    if mass not in cache:
        cache[mass] = decay_products.prepare_fixed_exhad_pool(
            mass=mass, **kwargs)
    return cache[mass]


def _gated_rates(runtime, llp, selected_decay_indices):
    """The tabulated rates with every closed row zeroed, and their visible sum.

    A parent below the sum of its daughters' rest masses has no phase space, so
    a rate an interpolant opened there is removed rather than sampled.  The
    visible branching ratio the output records is the sum of what survives.
    """
    gated, leaked = runtime.thresholds.gate_rates(
        llp.mass, llp.PDGs, llp.BrRatios_distr,
        hnl=llp.LLP_name == "HNL")
    visible = float(sum(gated[index] for index in selected_decay_indices))
    return gated, visible, leaked


def _coupling_squared(runtime, llp, c_tau):
    if llp.LLP_name == "Scalar-quartic":
        return 0.01
    if llp.LLP_name == runtime.coupling_conventions.ALP_FERMION:
        # Native table convention: coupling_squared is g_Y^2 with
        # g_Y == y == 2*v_h/f_a; no factor-of-four conversion applies.
        return runtime.coupling_conventions.native_coupling_squared(
            llp.LLP_name, llp.c_tau_int, c_tau, metadata=llp.coupling_metadata)
    return llp.c_tau_int / c_tau


def run_scan_point(runtime, config, llp, selected_decay_indices,
                   mass_index, ctau_index, n_ctaus, *,
                   batch_index_offset=0, fixed_exhad_pools=None):
    """Generate, decay and write one (mass, lifetime) point."""
    mass = float(config.masses[mass_index - 1])
    c_taus = config.c_taus[mass_index - 1]
    c_tau = float(c_taus[ctau_index - 1])
    if llp.mass != mass:
        llp.set_mass(mass)
        llp.compute_mass_dependent_properties()

    fixed_mass_batch_index, point_batch_index, _applied = scan_point_seed(
        _effective_seed(config), mass_index, ctau_index, n_ctaus,
        batch_index_offset)

    llp.set_c_tau(c_tau)
    uncertainty_label = getattr(llp, "uncertainty_label", config.uncertainty)
    channel_tag = runtime.mergeResults.channel_selection_tag(
        llp.decayChannels, selected_decay_indices)
    _gated, br_visible, _leaked = _gated_rates(
        runtime, llp, selected_decay_indices)
    if br_visible == 0:
        print("No visible decay channels at this mass. Skipping.")
        return

    _bench, generator_tag, generator_provenance = (
        runtime.output_provenance.effective_generator_identity(
            runtime.decayProducts, llp, selected_decay_indices,
            _effective_seed(config),
            stock_pythia_pool=config.stock_pythia_pool))

    coupling_squared = _coupling_squared(runtime, llp, c_tau)
    n_llp_total = config.n_pot * llp.Yield * coupling_squared
    if llp.Yield * coupling_squared < 1.0e-21:
        print("    Negligible yield. Skipping.")
        return

    grid = runtime.kinematics.Grids(
        llp.Distr,
        llp.Energy_distr,
        config.n_events,
        llp.mass,
        llp.c_tau_input,
        theta_max_sim=runtime.theta_max_dec_vol,
        cache_token=llp.kinematics_cache_token(),
    )
    grid.interpolate(False)
    grid.resample(config.events, False)
    grid.true_samples(False)

    momentum = grid.get_momentum()
    final_events = len(momentum)
    epsilon_polar = grid.epsilon_polar
    epsilon_azimuthal = final_events / config.events
    mother_results = grid.get_kinematics()
    average_decay_probability = mother_results[:, 6].mean()
    n_events_total = (
        n_llp_total
        * epsilon_polar
        * epsilon_azimuthal
        * average_decay_probability
        * br_visible
    )

    if n_events_total < config.min_events_threshold:
        print(
            f"    N_ev_tot = {n_events_total:.6e} < "
            f"{config.min_events_threshold}. Skipping decay computation..."
        )
        runtime.mergeResults.save_total_only(
            llp.LLP_name,
            llp.mass,
            coupling_squared,
            c_tau,
            n_llp_total,
            epsilon_polar,
            epsilon_azimuthal,
            average_decay_probability,
            br_visible,
            n_events_total,
            uncertainty_label,
            llp.MixingPatternArray,
            llp.decayChannels,
            llp.alp_production_mode,
            generator_tag=generator_tag,
            channel_tag=channel_tag,
            coupling_metadata=llp.coupling_metadata,
            generator_provenance=generator_provenance,
            alp_mixing_xi=getattr(llp, "xi", None),
            alp_interference=getattr(llp, "interference", None),
            output_root=PROJECT_ROOT,
        )
        return

    pools = {} if fixed_exhad_pools is None else fixed_exhad_pools
    fixed_exhad_pool = _get_or_create_fixed_exhad_pool(
        pools, llp.mass, runtime.decayProducts,
        size=config.events, PDGdecay=llp.PDGs,
        selected_decay_indices=selected_decay_indices,
        BrRatio=llp.BrRatios_distr,
        br_visible_val=br_visible,
        llp_name=llp.LLP_name, particle_path=llp.particle_path,
        exhad_variant=llp.scalar_lifetime,
        seed=_effective_seed(config),
        batch_index=fixed_mass_batch_index)

    unboosted, size_per_channel, process_labels = (
        runtime.decayProducts.simulateDecays_rest_frame(
            llp.mass,
            llp.PDGs,
            llp.BrRatios_distr,
            final_events,
            llp.Matrix_elements,
            list(selected_decay_indices),
            br_visible,
            llp_name=llp.LLP_name,
            particle_path=llp.particle_path,
            exhad_variant=llp.scalar_lifetime,
            seed=_effective_seed(config),
            batch_index=point_batch_index,
            fixed_exhad_pool=fixed_exhad_pool,
            stock_pythia_pool=config.stock_pythia_pool,
            return_process_labels=True,
            decay_channels=llp.decayChannels,
        )
    )
    boosted = runtime.boost.tab_boosted_decay_products(
        llp.mass, momentum, unboosted)

    started = time.time()
    runtime.mergeResults.save(
        mother_results,
        boosted,
        llp.LLP_name,
        llp.mass,
        llp.MixingPatternArray,
        llp.c_tau_input,
        llp.decayChannels,
        size_per_channel,
        final_events,
        epsilon_polar,
        epsilon_azimuthal,
        n_llp_total,
        coupling_squared,
        average_decay_probability,
        n_events_total,
        br_visible,
        list(selected_decay_indices),
        uncertainty_label,
        config.export_events,
        llp.alp_production_mode,
        generator_tag=generator_tag,
        channel_tag=channel_tag,
        process_labels=process_labels,
        coupling_metadata=llp.coupling_metadata,
        generator_provenance=generator_provenance,
        alp_mixing_xi=getattr(llp, "xi", None),
        alp_interference=getattr(llp, "interference", None),
        hadronization_metadata=runtime.exhadDecays.metadata(llp),
        output_root=PROJECT_ROOT,
    )
    print(f"    Exported in {time.time() - started:.1f} s")
    coupling_summary = (
        f"|g_agammagamma,total|^2: {coupling_squared:.6e}"
        if llp.LLP_name == "ALP-mixed"
        else f"Squared coupling:      {coupling_squared:.6e}"
    )
    print(
        f"LLP mass {mass} GeV ({mass_index}/{len(config.masses)}) "
        f"cτ {c_tau} m ({ctau_index}/{n_ctaus}) processed.\n"
        f"Sampled inside volume: {final_events:.6e}\n"
        f"{coupling_summary}\n"
        f"N_LLP_total:           {n_llp_total:.6e}\n"
        f"ε_polar:               {epsilon_polar:.6e}\n"
        f"ε_azimuthal:           {epsilon_azimuthal:.6e}\n"
        f"⟨P_decay⟩:             {average_decay_probability:.6e}\n"
        f"Visible Br:            {br_visible:.6e}\n"
        f"N_events_tot:          {n_events_total:.6e}\n"
    )


def run_mass_lifetime_grid(runtime, config, llp, selected_decay_indices, *,
                           batch_index_offset=0, fixed_exhad_pools=None):
    """Run every (mass, lifetime) point of one configuration, in order."""
    pools = {} if fixed_exhad_pools is None else fixed_exhad_pools
    total_masses = len(config.masses)
    print(f"\nTotal masses to process: {total_masses}")

    for mass_index, mass in enumerate(config.masses, 1):
        print(f"\nProcessing mass {mass} GeV  ({mass_index}/{total_masses})")
        llp.set_mass(mass)
        llp.compute_mass_dependent_properties()

        _bench, generator_tag, _provenance = (
            runtime.output_provenance.effective_generator_identity(
                runtime.decayProducts, llp, selected_decay_indices,
                _effective_seed(config),
                stock_pythia_pool=config.stock_pythia_pool))
        print("  Effective generator/output tag:", generator_tag)

        _gated, br_visible, _leaked = _gated_rates(
            runtime, llp, selected_decay_indices)
        if br_visible == 0:
            print("No visible decay channels at this mass. Skipping.")
            continue

        c_taus = config.c_taus[mass_index - 1]
        n_ctaus = len(c_taus)
        print(f"  Lifetimes to process: {n_ctaus}")
        for ctau_index, c_tau in enumerate(c_taus, 1):
            print(f"  Processing c_tau = {c_tau} m  ({ctau_index}/{n_ctaus})")
            run_scan_point(
                runtime, config, llp, selected_decay_indices,
                mass_index, ctau_index, n_ctaus,
                batch_index_offset=batch_index_offset,
                fixed_exhad_pools=pools)


def run_simulation(config: SimulationConfig) -> None:
    """Run one fully resolved non-interactive configuration."""
    from funcs import seeding
    from funcs.LLP_selection import validate_masses_in_domain

    runtime = _load_runtime(headless=True)
    # One seed reaches every stream: the host NumPy generator, Numba's own
    # RNG inside the njit samplers, and (through the NumPy stream) Pythia's
    # own seed.  exHad's generators take their seed explicitly per call.
    seed = _effective_seed(config)
    seeding.seed_all(seed)
    print(f"\nRun seed: {seed}. Repeating this card reproduces this run exactly.")
    _print_ship_setup(runtime)
    llp = _make_llp(runtime, config)
    # A mass the installed tables do not cover is refused before any point
    # runs, rather than dropped from the grid with a printed line.
    validate_masses_in_domain(
        config.masses, llp.m_min_tabulated, llp.m_max_tabulated)
    selected = config.selected_decay_indices(
        tuple(str(item) for item in llp.decayChannels))
    if config.plots:
        _generate_phenomenology_plots(runtime, llp, selected)
    run_mass_lifetime_grid(
        runtime, config, llp, selected,
        batch_index_offset=config.batch_index_offset)


def _interactive_main(*, exhad_mode: str = "auto") -> None:
    # Imports remain here so importing simulate itself can never prompt.
    from funcs.LLP_selection import (
        configure_interactively,
        prompt_decay_channels,
        prompt_masses_and_c_taus,
    )

    runtime = _load_runtime(headless=False)
    _print_ship_setup(runtime)
    inputs = configure_interactively()
    particle_selection = inputs["particle_selection"]

    # A flag on the command line has already said which generator to use; with
    # no flag, ask, because an interactive user is here to choose.
    exhad = runtime.exhadDecays
    if exhad_mode == "auto" and sys.stdin.isatty():
        exhad.set_selection(
            particle_selection["LLP_name"], particle_selection["particle_path"],
            inputs["scalar_lifetime"])
        # Invalid cards/paths/imports deliberately propagate instead of silently
        # turning a requested matched run into raw Pythia.
        if exhad.can_use_exhad():
            answer = input(
                f"\nHadronic (quark/gluon) decays of "
                f"{particle_selection['LLP_name']}: use the "
                "reference-matched exhad\ngenerator or raw Pythia?"
                "\n  [E] exhad (default, enforces conservation laws + "
                "exclusive-data calibration)\n  [P] raw Pythia\n : "
            ).strip().lower()
            if answer.startswith("p"):
                exhad_mode = "off"
                print("  -> raw Pythia will be used for hadronic decays.")
            else:
                exhad_mode = "on"
                print(
                    "  -> exhad (reference-matched) will be used for hadronic "
                    "decays (within its calibrated mass window)."
                )

    mixing = inputs["mixing_pattern"]
    partial = SimpleNamespace(
        exhad_mode=exhad_mode,
        exhad_root=None,
        exhad_python=None,
        particle_selection=particle_selection,
        mixing_pattern=mixing,
        uncertainty=inputs["uncertainty"],
        alp_production_mode=inputs["alp_production_mode"],
        dp_production_mode=inputs["dp_production_mode"],
        scalar_prescription=inputs["scalar_lifetime"],
        xi=inputs.get("xi"),
        interference=inputs.get("interference"),
        llp_pdg=None,
        masses=None,
    )
    llp = _make_llp(runtime, partial)
    selected = prompt_decay_channels(llp.decayChannels)
    _generate_phenomenology_plots(runtime, llp, selected)
    masses, c_taus = prompt_masses_and_c_taus(
        llp.m_min_tabulated, llp.m_max_tabulated)

    config = config_from_mapping(
        {
            "exhad_mode": exhad_mode,
            "model": particle_selection["LLP_name"],
            "events": inputs["resampleSize"],
            "masses": masses,
            "c_taus": c_taus,
            "decay_channels": [str(llp.decayChannels[index]) for index in selected],
            "mixing_pattern": list(mixing) if mixing is not None else None,
            "uncertainty": inputs["uncertainty"],
            "alp_production_mode": inputs["alp_production_mode"],
            "dp_production_mode": inputs["dp_production_mode"],
            "scalar_prescription": inputs["scalar_lifetime"],
            "xi": inputs.get("xi"),
            "interference": inputs.get("interference"),
            "plots": True,
            "seed": _interactive_seed(),
        },
        project_root=PROJECT_ROOT,
    )
    if runtime.exhadDecays.can_use_exhad() and runtime.exhadDecays.is_hnl_bench():
        runtime.exhadDecays.require_hnl_parent_masses_supported(config.masses)
    print(
        f"\nRun seed: {_effective_seed(config)}. "
        "Repeating these choices reproduces this run exactly."
    )
    run_mass_lifetime_grid(runtime, config, llp, selected)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        _interactive_main()
        return 0
    if arguments in (["--rawPythia"], ["--raw-pythia"], ["--exhad"]):
        _interactive_main(exhad_mode="on" if arguments == ["--exhad"] else "off")
        return 0

    config, validate_only = config_from_command_line(arguments, project_root=PROJECT_ROOT)
    if validate_only:
        payload = config.as_dict()
        channel_names = load_decay_channel_names(config)
        resolved_indices = config.selected_decay_indices(channel_names)
        payload["resolved_decay_indices"] = resolved_indices
        payload["resolved_decay_channels"] = [channel_names[index] for index in resolved_indices]
        print(json.dumps(payload, indent=2, sort_keys=True))
        print("Configuration is valid; no simulation was run.")
        return 0

    run_simulation(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
