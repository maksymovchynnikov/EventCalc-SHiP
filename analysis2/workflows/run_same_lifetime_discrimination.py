"""Run the completed equal-mass, equal-lifetime shape discrimination grid."""

from __future__ import annotations

from argparse import ArgumentParser

import numpy as np
import pandas as pd

from analysis2.config import pseudoexperiment_seed, spectrum_model_seed
from analysis2.lifetimes import lifetime_point_records
from analysis2.models import ALP_PHOTON_COMBINED, ALP_SU2L, MODELS
from analysis2.paths import profile_output_dir
from analysis2.plotting import plot_accuracy
from analysis2.statistics import (
    kl_divergence, minimum_events_for_accuracy, simulate_shape_discrimination,
    total_variation_distance,
)
from analysis2.templates import cached_probability_templates
from analysis2.workflows import (
    add_profile_cache_arguments, config_and_adapter, float_token, require_columns,
    write_dataframe, write_manifest,
)


def parse_arguments():
    parser = ArgumentParser(description=__doc__)
    add_profile_cache_arguments(parser)
    parser.add_argument("--selection-only", action="store_true", help="Write grid points without EventCalc")
    return parser.parse_args()


def select_grid_points(ranges: pd.DataFrame, fractions) -> pd.DataFrame:
    return pd.DataFrame(lifetime_point_records(ranges.to_dict("records"), fractions))


def threshold_column(target: float) -> str:
    return f"minimum_events_for_{100 * target:.0f}pct_worst_case_accuracy"


def main() -> None:
    args = parse_arguments()
    config, adapter = config_and_adapter(args)
    range_path = profile_output_dir(config.name, "ctau_ranges") / "common_ctau_ranges.csv"
    ranges = pd.read_csv(range_path)
    require_columns(ranges, {"mass_GeV", "ctau_lower_m", "ctau_upper_m"}, range_path)
    points = select_grid_points(ranges, config.discrimination.lifetime_points)
    output_dir = profile_output_dir(config.name, "same_lifetime_discrimination")
    write_dataframe(points, output_dir / "selected_lifetime_points.csv")
    if args.selection_only:
        write_manifest(config, "run_same_lifetime_discrimination_selection_only", output_dir)
        print(points.to_string(index=False))
        return
    mass_indices = {mass: index for index, mass in enumerate(sorted(points["mass_GeV"].unique()))}
    accuracy_frames, summary_rows = [], []
    for point_index, point in enumerate(points.itertuples(index=False)):
        mass_gev, ctau_m = float(point.mass_GeV), float(point.ctau_m)
        spectra = {}
        for model_index, model in enumerate(MODELS):
            seed = spectrum_model_seed(mass_indices[mass_gev], model_index)
            spectra[model.identifier] = adapter.evaluate_model(
                model.identifier, mass_gev, ctau_m, seed, "spectrum"
            )
        if min(spectrum.expected_events for spectrum in spectra.values()) < config.lifetimes.event_threshold:
            raise RuntimeError("a selected point no longer passes the event threshold")
        initial_edges = np.geomspace(mass_gev, config.energy_max_gev, config.initial_energy_bins + 1)
        templates = cached_probability_templates(
            adapter.cache, spectra, initial_edges, config.discrimination.minimum_bin_n_eff,
            config.discrimination.jeffreys_alpha, force=args.force,
        )
        photon = templates[ALP_PHOTON_COMBINED.identifier]
        su2 = templates[ALP_SU2L.identifier]
        template_table = pd.DataFrame({
            "profile": config.name, "selection_name": config.selection_name,
            "mass_GeV": mass_gev, "ctau_m": ctau_m,
            "lifetime_label": point.lifetime_label,
            "bin_index": np.arange(len(photon.probabilities)),
            "energy_low_GeV": photon.energy_edges_gev[:-1],
            "energy_high_GeV": photon.energy_edges_gev[1:],
            "photon_probability": photon.probabilities, "su2_probability": su2.probabilities,
            "log_su2_over_photon": np.log(su2.probabilities / photon.probabilities),
        })
        stem = f"ma_{float_token(mass_gev)}_ctau_{float_token(ctau_m)}_{point.lifetime_label}"
        write_dataframe(template_table, output_dir / "templates" / f"probability_templates_{stem}.csv")
        simulation = simulate_shape_discrimination(
            photon.probabilities, su2.probabilities,
            config.discrimination.maximum_observed_events,
            config.discrimination.pseudoexperiments, pseudoexperiment_seed(point_index),
        )
        accuracy = pd.DataFrame(simulation.records())
        accuracy.insert(0, "lifetime_label", point.lifetime_label)
        accuracy.insert(0, "ctau_m", ctau_m)
        accuracy.insert(0, "mass_GeV", mass_gev)
        accuracy_frames.append(accuracy)
        write_dataframe(accuracy, output_dir / "accuracy_tables" / f"classification_accuracy_{stem}.csv")
        plot_accuracy(accuracy, mass_gev, ctau_m, output_dir / "plots" / f"classification_accuracy_{stem}.pdf")
        summary = {
            "profile": config.name, "selection_name": config.selection_name,
            "mass_GeV": mass_gev, "lifetime_label": point.lifetime_label,
            "log_interval_fraction": point.log_interval_fraction, "ctau_m": ctau_m,
            "ctau_lower_m": point.ctau_lower_m, "ctau_upper_m": point.ctau_upper_m,
            "photon_expected_events": spectra[ALP_PHOTON_COMBINED.identifier].expected_events,
            "su2_expected_events": spectra[ALP_SU2L.identifier].expected_events,
            "number_of_adaptive_bins": len(photon.probabilities),
            "minimum_required_bin_N_eff": config.discrimination.minimum_bin_n_eff,
            "photon_template_total_N_eff": photon.total_n_eff,
            "su2_template_total_N_eff": su2.total_n_eff,
            "model_total_variation_distance": total_variation_distance(
                photon.probabilities, su2.probabilities
            ),
            "KL_photon_to_su2_per_event": kl_divergence(photon.probabilities, su2.probabilities),
            "KL_su2_to_photon_per_event": kl_divergence(su2.probabilities, photon.probabilities),
        }
        for target in config.discrimination.target_accuracies:
            summary[threshold_column(target)] = minimum_events_for_accuracy(
                simulation.number_of_events, simulation.worst_case_correct_fraction, target
            )
        summary_rows.append(summary)
        write_dataframe(pd.DataFrame(summary_rows), output_dir / "discrimination_grid_summary_checkpoint.csv")
    summary = pd.DataFrame(summary_rows).sort_values(["mass_GeV", "ctau_m"])
    write_dataframe(summary, output_dir / "discrimination_grid_summary.csv")
    write_dataframe(pd.concat(accuracy_frames, ignore_index=True), output_dir / "classification_accuracy_grid_all.csv")
    write_manifest(config, "run_same_lifetime_discrimination", output_dir)
    print(f"Saved same-lifetime discrimination to {output_dir}")


if __name__ == "__main__":
    main()
