"""Evaluate lifetime-dependent mother-level spectra and stability distances."""

from __future__ import annotations

from argparse import ArgumentParser

import numpy as np
import pandas as pd

from analysis2.config import spectrum_model_seed
from analysis2.models import MODELS
from analysis2.paths import profile_output_dir
from analysis2.plotting import plot_lifetime_spectra
from analysis2.spectra import normalized_weighted_spectrum, weighted_quantiles
from analysis2.statistics import maximum_cdf_distance, total_variation_distance
from analysis2.workflows import (
    add_profile_cache_arguments, config_and_adapter, require_columns, write_dataframe, write_manifest,
)


def parse_arguments():
    parser = ArgumentParser(description=__doc__)
    add_profile_cache_arguments(parser)
    return parser.parse_args()


def lifetime_values(lower_m: float, upper_m: float, points: int, reference_m: float) -> tuple[float, ...]:
    if not 0.0 < lower_m < upper_m:
        raise ValueError("common lifetime interval must be finite, positive and ordered")
    values = np.concatenate([np.geomspace(max(3.0, lower_m), upper_m, points), [reference_m]])
    return tuple(float(value) for value in np.unique(np.round(values, 12)))


def spectrum_summary(spectrum, histogram, model, reference_ctau_m: float) -> dict:
    weights, energies = spectrum.absolute_event_weights, spectrum.energies_gev
    q16, median, q84 = weighted_quantiles(energies, weights, np.array([0.16, 0.50, 0.84]))
    return {
        "selection_name": spectrum.selection_name, "model": model.legacy_name,
        "model_id": model.identifier, "plot_label": model.plot_label,
        "mass_GeV": spectrum.mass_gev, "ctau_m": spectrum.ctau_m,
        "weighted_mean_energy_GeV": float(np.average(energies, weights=weights)),
        "weighted_median_energy_GeV": float(median), "energy_q16_GeV": float(q16),
        "energy_q84_GeV": float(q84),
        "fraction_below_1_GeV": float(weights[energies < 1.0].sum() / weights.sum()),
        "effective_sample_size": spectrum.total_n_eff, "accepted_samples": spectrum.accepted_samples,
        "normalization": float(histogram.bin_probabilities.sum()),
        "range_coverage": histogram.range_coverage, "N_LLP_total": spectrum.n_llp_total,
        "epsilon_polar": spectrum.epsilon_polar, "epsilon_azimuthal": spectrum.epsilon_azimuthal,
        "mean_P_decay": spectrum.mean_decay_probability, "visible_Br": spectrum.visible_br,
        "N_events": spectrum.expected_events,
        "N_events_primary": spectrum.source_expected_events.get("primary", np.nan),
        "N_events_cascade": spectrum.source_expected_events.get("cascade", np.nan),
        "cascade_event_fraction": (
            spectrum.source_expected_events.get("cascade", 0.0) / spectrum.expected_events
        ),
        "reference_ctau_m": reference_ctau_m,
    }


def main() -> None:
    args = parse_arguments()
    config, adapter = config_and_adapter(args)
    input_path = profile_output_dir(config.name, "ctau_ranges") / "common_ctau_ranges.csv"
    ranges = pd.read_csv(input_path)
    require_columns(ranges, {"mass_GeV", "ctau_lower_m", "ctau_upper_m", "upper_extends_beyond_scan"}, input_path)
    flags = ranges["upper_extends_beyond_scan"].astype(str).str.lower().map({"true": True, "false": False})
    if flags.isna().any():
        raise ValueError("upper_extends_beyond_scan must contain only True or False")
    if flags.any():
        raise RuntimeError("lifetime stability scan requires finite common intervals")
    output_dir = profile_output_dir(config.name, "lifetime_spectra")
    rows = []
    for mass_index, range_row in enumerate(ranges.sort_values("mass_GeV").itertuples(index=False)):
        mass_gev = float(range_row.mass_GeV)
        ctaus = lifetime_values(
            float(range_row.ctau_lower_m), float(range_row.ctau_upper_m),
            config.lifetimes.scan_points, config.lifetimes.reference_ctau_m,
        )
        edges = np.geomspace(mass_gev, config.energy_max_gev, config.initial_energy_bins + 1)
        for model_index, model in enumerate(MODELS):
            seed = spectrum_model_seed(mass_index, model_index)
            weighted, histograms = {}, {}
            for ctau_m in ctaus:
                weighted[ctau_m] = adapter.evaluate_model(
                    model.identifier, mass_gev, ctau_m, seed, "spectrum"
                )
                histograms[ctau_m] = normalized_weighted_spectrum(weighted[ctau_m], edges)
            reference = histograms[config.lifetimes.reference_ctau_m].bin_probabilities
            for ctau_m in ctaus:
                row = spectrum_summary(
                    weighted[ctau_m], histograms[ctau_m], model,
                    config.lifetimes.reference_ctau_m,
                )
                row["profile"] = config.name
                probabilities = histograms[ctau_m].bin_probabilities
                row.update({
                    "is_reference": bool(np.isclose(ctau_m, config.lifetimes.reference_ctau_m)),
                    "weighted_mean_gamma": row["weighted_mean_energy_GeV"] / mass_gev,
                    "ctau_times_weighted_mean_gamma_m": (
                        ctau_m * row["weighted_mean_energy_GeV"] / mass_gev
                    ),
                    "N_events_ge_10": weighted[ctau_m].expected_events >= config.lifetimes.event_threshold,
                    "total_variation_distance_to_reference": total_variation_distance(probabilities, reference),
                    "binned_cdf_max_distance_to_reference": maximum_cdf_distance(probabilities, reference),
                    "nonempty_bins_with_low_N_eff": int(np.count_nonzero(
                        (histograms[ctau_m].sum_weights_per_bin > 0.0)
                        & (histograms[ctau_m].effective_samples_per_bin < config.n_eff_warning)
                    )),
                    "common_ctau_lower_m": float(range_row.ctau_lower_m),
                    "common_ctau_upper_m": float(range_row.ctau_upper_m),
                })
                rows.append(row)
            plot_lifetime_spectra(
                histograms, model.plot_label, mass_gev, config.lifetimes.reference_ctau_m,
                config.n_eff_warning,
                output_dir / "plots" / f"lifetime_dependence_{model.identifier}_ma_{mass_gev:g}.pdf",
            )
    summary = pd.DataFrame(rows).sort_values(["mass_GeV", "model_id", "ctau_m"])
    write_dataframe(summary, output_dir / "lifetime_stability_summary.csv")
    write_manifest(config, "scan_lifetime_spectra", output_dir)
    print(f"Saved lifetime spectra summary to {output_dir}")


if __name__ == "__main__":
    main()
