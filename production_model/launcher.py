"""Shared full-source scan loop for simulate.py and run_batch.py."""
from dataclasses import replace
from pathlib import Path
import json
import time
import numpy as np


def run_full_source_scan(runtime, config, llp, selected, *, batch_index_offset=0):
    from runtime_generator import RuntimeModelScan
    from simulate import _effective_seed, scan_point_seed
    scan = RuntimeModelScan(config)
    # The launcher has already resolved the decay backend and loaded the model.
    scan.runtime = runtime
    scan._llp_template = llp
    scan._initialized = True
    merge = runtime.mergeResults
    tag_channels = merge.channel_selection_tag(llp.decayChannels, selected)
    for im, mass in enumerate(config.masses, 1):
        # Retain the costly provider, but only one mass's folded source arrays.
        scan._collision_sources.clear()
        for it, ctau in enumerate(config.c_taus[im-1], 1):
            started = time.perf_counter()
            _, _, seed = scan_point_seed(_effective_seed(config), im, it,
                                         len(config.c_taus[im-1]), batch_index_offset)
            generator = scan.generator(mass=mass, c_tau=ctau, seed=seed)
            generator.init()
            sample = generator._parents_for_output(config.events)
            estimate = generator.yields()
            local = generator.llp
            _, decay_tag, provenance = runtime.output_provenance.effective_generator_identity(
                runtime.decayProducts, local, selected, _effective_seed(config))
            source = generator._collision_source
            source_tag = (f"full-source-{local.dp_production_mode}-p{config.production_power}"
                          f"-s{source.metadata['shower_sha256'][:12]}")
            if config.model == "Inelastic-DM":
                source_tag += f"-delta{config.idm_delta:g}"
            tag = f"{decay_tag}-{source_tag}"
            label = getattr(local, "uncertainty_label", config.uncertainty)
            total = float(estimate.expected_events)
            args = (config.model, mass, estimate.coupling_squared, ctau,
                    estimate.produced_llps, estimate.epsilon_polar,
                    estimate.epsilon_transverse, estimate.mean_decay_probability,
                    estimate.visible_branching_ratio, total, label,
                    local.MixingPatternArray, local.decayChannels, local.alp_production_mode)
            output_root = config.project_root
            if total < config.min_events_threshold:
                merge.save_total_only(*args, generator_tag=tag, channel_tag=tag_channels,
                                      generator_provenance=provenance, output_root=output_root)
                event_path = None
            else:
                products, channels = generator._decay(sample.mother, sample.momentum)
                names, indices, counts = np.unique(channels, return_inverse=True, return_counts=True)
                order = np.argsort(indices, kind="stable")
                merge.save(sample.mother[order], products[order], config.model, mass,
                           local.MixingPatternArray, ctau, names, counts, len(order),
                           estimate.epsilon_polar, estimate.epsilon_transverse,
                           estimate.produced_llps, estimate.coupling_squared,
                           estimate.mean_decay_probability, total,
                           estimate.visible_branching_ratio, list(range(len(names))), label,
                           config.export_events, local.alp_production_mode,
                           generator_tag=tag, channel_tag=tag_channels,
                           generator_provenance=provenance, output_root=output_root)
                event_dir, _ = merge._output_directories(output_root, config.model)
                event_path = Path(merge._event_output_path(
                    event_dir, config.model, mass, ctau, local.MixingPatternArray,
                    label, local.alp_production_mode, tag, tag_channels))
                if config.export_events:
                    np.savez_compressed(str(event_path)+".production.npz",
                        origins_m=sample.origins[order], momenta_GeV=sample.momentum[order],
                        category=sample.categories[order], mechanism=sample.mechanisms[order],
                        signed_decay_weights=sample.mother[order, 6],
                        event_weights_per_pot=(estimate.coupling_squared *
                            generator._production_cdf.absolute_yield *
                            estimate.visible_branching_ratio / len(order) * sample.mother[order, 6]))
            metadata = {
                "schema": "eventcalc-full-collision-source-v1", "production": source.metadata,
                "model": config.model, "input_mass_GeV": mass,
                "mother_mass_GeV": source.mass, "ctau_m": ctau,
                "mass_input": "mediator" if config.model == "Inelastic-DM" else "dark photon",
                "idm_benchmark": {"ratio": 3, "alpha_D": .1, "delta": config.idm_delta}
                    if config.model == "Inelastic-DM" else None,
                "production_power": config.production_power, "source_seed": _effective_seed(config),
                "event_seed": seed, "category": local.dp_production_mode,
                "normalization": estimate.as_dict(),
                "source_weight_sum_per_pot_epsilon2": source.total_yield,
                "intersecting_absolute_weight_sum": generator._production_cdf.absolute_yield,
                "mother_column_6": "signed decay probability; retain negative weights",
                "epsilon_polar_field": "absolute intersecting source weight / signed total production; proposal normalization, not an angular cut",
                "vertex_convention": "metres from target front face; each ray starts at its correlated collision position",
                "event_weight": "N_POT * epsilon^2 * absolute intersecting source weight * selected BR * column_6 / number of rows",
                "integration_error_scope": "Reported Monte Carlo error is conditional on the finite source sample; vary production_power/seed to check source integration.",
                "production_uncertainty_command": "model_sensitivity.py (both boundaries, automatic correlated variations)",
            }
            _, total_dir = merge._output_directories(output_root, config.model)
            total_file = Path(total_dir)/merge._total_filename(config.model, local.MixingPatternArray,
                label, local.alp_production_mode, tag, tag_channels)
            # Each coordinate gets its own metadata; other lifetimes cannot overwrite it.
            sidecar = str(total_file)+f".m{mass:.17g}.ctau{ctau:.17g}.production.json"
            Path(sidecar).write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
            if event_path is not None and config.export_events:
                Path(str(event_path)+".production.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
            print(f"Full correlated source: {config.model}, input mass {mass:g} GeV, "
                  f"mother mass {source.mass:g} GeV, ctau {ctau:g} m; "
                  f"N_events={total:.8g}, epsilon^2={estimate.coupling_squared:.8g}; "
                  f"{time.perf_counter()-started:.2f} s")
