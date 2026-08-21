#!/usr/bin/env python3
"""EventCalc's interactive and non-interactive simulation launcher."""

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


def _load_runtime(*, headless: bool) -> SimpleNamespace:
    # Force a non-GUI backend before ship_setup imports matplotlib.pyplot.
    if headless:
        os.environ["MPLBACKEND"] = "Agg"
        import matplotlib

        matplotlib.use("Agg", force=True)

    import numpy as np

    from funcs import boost, decayProducts, initLLP, kinematics, mergeResults
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
        decayProducts=decayProducts,
        initLLP=initLLP,
        kinematics=kinematics,
        mergeResults=mergeResults,
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


def _make_llp(runtime: SimpleNamespace, config: SimulationConfig):
    mixing = (
        runtime.np.asarray(config.mixing_pattern, dtype=float)
        if config.mixing_pattern is not None
        else None
    )
    return runtime.initLLP.LLP(
        mass=None,
        particle_selection=config.particle_selection,
        mixing_pattern=mixing,
        uncertainty=config.uncertainty,
        alp_production_mode=config.alp_production_mode,
        xi=config.xi,
        interference=config.interference,
    )


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
    plot_folder = f"plots/{llp.LLP_name}/phenomenology"
    if llp.LLP_name == "ALP-photon":
        plot_folder = f"{plot_folder}_{llp.alp_production_mode}"
    elif llp.LLP_name == "ALP-mixed":
        plot_folder = f"{plot_folder}_xi{llp.xi:g}_{llp.interference}"
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


def _run_mass_lifetime_grid(
    runtime: SimpleNamespace,
    config: SimulationConfig,
    llp,
    selected_decay_indices: Sequence[int],
) -> None:
    total_masses = len(config.masses)
    print(f"\nTotal masses to process: {total_masses}")

    for mass_index, (mass, c_taus) in enumerate(zip(config.masses, config.c_taus), 1):
        if not _mass_is_tabulated(mass, llp):
            print(f"Mass {mass} GeV outside tabulated range. Skipping.")
            continue

        print(f"\nProcessing mass {mass} GeV  ({mass_index}/{total_masses})")
        llp.set_mass(mass)
        llp.compute_mass_dependent_properties()

        br_visible = sum(llp.BrRatios_distr[index] for index in selected_decay_indices)
        if br_visible == 0:
            print("No visible decay channels at this mass. Skipping.")
            continue

        total_c_taus = len(c_taus)
        print(f"  Lifetimes to process: {total_c_taus}")
        for ctau_index, c_tau in enumerate(c_taus, 1):
            print(f"  Processing c_tau = {c_tau} m  ({ctau_index}/{total_c_taus})")

            llp.set_c_tau(c_tau)
            coupling_squared = llp.c_tau_int / c_tau if llp.LLP_name != "Scalar-quartic" else 0.01
            n_llp_total = config.n_pot * llp.Yield * coupling_squared
            if llp.Yield * coupling_squared < 1.0e-21:
                print("    Negligible yield. Skipping.")
                continue

            grid = runtime.kinematics.Grids(
                llp.Distr,
                llp.Energy_distr,
                config.n_events,
                llp.mass,
                llp.c_tau_input,
                theta_max_sim=runtime.theta_max_dec_vol,
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
                    config.uncertainty,
                    llp.MixingPatternArray,
                    llp.decayChannels,
                    llp.alp_production_mode,
                    llp.xi,
                    llp.interference,
                )
                continue

            unboosted, size_per_channel = runtime.decayProducts.simulateDecays_rest_frame(
                llp.mass,
                llp.PDGs,
                llp.BrRatios_distr,
                final_events,
                llp.Matrix_elements,
                list(selected_decay_indices),
                br_visible,
            )
            boosted = runtime.boost.tab_boosted_decay_products(llp.mass, momentum, unboosted)

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
                config.uncertainty,
                config.export_events,
                llp.alp_production_mode,
                llp.xi,
                llp.interference,
            )
            print(f"    Exported in {time.time() - started:.1f} s")
            coupling_summary = (
                f"|g_agammagamma,total|^2: {coupling_squared:.6e}"
                if llp.LLP_name == "ALP-mixed"
                else f"Squared coupling:      {coupling_squared:.6e}"
            )
            print(
                f"LLP mass {mass} GeV ({mass_index}/{total_masses}) "
                f"cτ {c_tau} m ({ctau_index}/{total_c_taus}) processed.\n"
                f"Sampled inside volume: {final_events:.6e}\n"
                f"{coupling_summary}\n"
                f"N_LLP_total:           {n_llp_total:.6e}\n"
                f"ε_polar:               {epsilon_polar:.6e}\n"
                f"ε_azimuthal:           {epsilon_azimuthal:.6e}\n"
                f"⟨P_decay⟩:             {average_decay_probability:.6e}\n"
                f"Visible Br:            {br_visible:.6e}\n"
                f"N_events_tot:          {n_events_total:.6e}\n"
            )


def run_simulation(config: SimulationConfig) -> None:
    """Run one fully resolved non-interactive configuration."""
    runtime = _load_runtime(headless=True)
    if config.seed is not None:
        runtime.np.random.seed(config.seed)
    _print_ship_setup(runtime)
    llp = _make_llp(runtime, config)
    selected = config.selected_decay_indices(tuple(str(item) for item in llp.decayChannels))
    if config.plots:
        _generate_phenomenology_plots(runtime, llp, selected)
    _run_mass_lifetime_grid(runtime, config, llp, selected)


def _interactive_main() -> None:
    # Imports remain here so importing simulate itself can never prompt.
    from funcs.LLP_selection import (
        prompt_alp_production_mode,
        prompt_alp_mixing,
        prompt_decay_channels,
        prompt_masses_and_c_taus,
        prompt_mixing_pattern,
        prompt_resample_size,
        prompt_uncertainty,
        select_particle,
    )

    runtime = _load_runtime(headless=False)
    _print_ship_setup(runtime)
    events = prompt_resample_size()
    particle_selection = select_particle(PROJECT_ROOT / "Distributions")
    uncertainty = prompt_uncertainty(particle_selection)
    alp_mode = prompt_alp_production_mode(particle_selection)
    xi, interference = prompt_alp_mixing(particle_selection)
    mixing = prompt_mixing_pattern(particle_selection)

    partial = SimpleNamespace(
        particle_selection=particle_selection,
        mixing_pattern=mixing,
        uncertainty=uncertainty,
        alp_production_mode=alp_mode,
        xi=xi,
        interference=interference,
    )
    llp = _make_llp(runtime, partial)
    selected = prompt_decay_channels(llp.decayChannels)
    _generate_phenomenology_plots(runtime, llp, selected)
    masses, c_taus = prompt_masses_and_c_taus()

    config = config_from_mapping(
        {
            "model": particle_selection["LLP_name"],
            "events": events,
            "masses": masses,
            "c_taus": c_taus,
            "decay_channels": [str(llp.decayChannels[index]) for index in selected],
            "mixing_pattern": mixing,
            "uncertainty": uncertainty,
            "alp_production_mode": alp_mode,
            "xi": xi,
            "interference": interference,
            "plots": True,
        },
        project_root=PROJECT_ROOT,
    )
    _run_mass_lifetime_grid(runtime, config, llp, selected)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        _interactive_main()
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
