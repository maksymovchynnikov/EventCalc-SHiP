from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from analysis2.cache import CacheStore
from analysis2.config import PRODUCTION, QUICK
from analysis2.lifetime_template_banks import (
    LifetimeTemplateBank,
    load_template_bank,
    save_bank_artifacts,
)
from analysis2.profiled_statistics import PROFILED_ACCURACY_COLUMNS
from analysis2.workflows.lifetime_blind_profiled_likelihood import (
    THRESHOLD_SUMMARY_COLUMNS,
    parse_arguments,
    profiled_run_axes,
    run_workflow,
    summarize_mass_threshold,
)


def toy_bank(profile: str = "quick") -> LifetimeTemplateBank:
    return LifetimeTemplateBank(
        mass_gev=0.3,
        energy_edges_gev=np.array([0.3, 1.0, 4.0]),
        minimum_bin_n_eff=5.0,
        jeffreys_alpha=0.5,
        event_threshold=10.0,
        template_seed_offset=0,
        template_base_seed=54_321,
        photon_ctau_m=np.array([3.0, 30.0]),
        photon_probabilities=np.array([[0.99, 0.01], [0.98, 0.02]]),
        photon_n_events=np.array([30.0, 10.0]),
        photon_n_events_before_ecal=np.array([60.0, 20.0]),
        photon_epsilon_ecal_weighted=np.array([0.5, 0.5]),
        photon_total_n_eff=np.array([1000.0, 1000.0]),
        photon_interval_m=np.array([3.0, 30.0]),
        su2_ctau_m=np.array([4.0, 40.0]),
        su2_probabilities=np.array([[0.01, 0.99], [0.02, 0.98]]),
        su2_n_events=np.array([40.0, 10.0]),
        su2_n_events_before_ecal=np.array([80.0, 20.0]),
        su2_epsilon_ecal_weighted=np.array([0.5, 0.5]),
        su2_total_n_eff=np.array([1000.0, 1000.0]),
        su2_interval_m=np.array([4.0, 40.0]),
        profile=profile,
        selection_name="diphoton_ecal",
    )


def write_toy_bank(input_dir: Path) -> Path:
    path = input_dir / "template_bank_ma_0p3.npz"
    save_bank_artifacts(
        toy_bank(),
        bank_path=path,
        summary_path=input_dir / "template_bank_summary_ma_0p3.csv",
        probability_path=input_dir / "template_probabilities_ma_0p3.csv",
    )
    return path


def test_saved_bank_retains_profile_and_selection_identity(tmp_path):
    path = write_toy_bank(tmp_path)
    loaded = load_template_bank(path)
    assert loaded.profile == "quick"
    assert loaded.selection_name == "diphoton_ecal"


class ProfiledWorkflowTests(unittest.TestCase):
    def test_threshold_summary_header_exactly_matches_the_golden_table(self):
        self.assertEqual(
            THRESHOLD_SUMMARY_COLUMNS,
            (
                "mass_GeV",
                "rebin_factor",
                "number_of_energy_bins",
                "jeffreys_alpha",
                "stored_jeffreys_alpha",
                "truth_grid",
                "profile_grid",
                "number_of_photon_truth_lifetimes",
                "number_of_su2_truth_lifetimes",
                "number_of_photon_profile_lifetimes",
                "number_of_su2_profile_lifetimes",
                "pseudoexperiments_per_truth_and_seed",
                "number_of_seeds",
                "target_accuracy",
                "threshold_reached",
                "minimum_persistent_events",
                "maximum_tested_events",
                "worst_case_accuracy_at_maximum_events",
                "accuracy_at_threshold",
                "limiting_seed_at_threshold",
                "limiting_truth_model_at_threshold",
                "limiting_truth_lifetime_index_at_threshold",
                "limiting_truth_ctau_m_at_threshold",
            ),
        )

    def test_cli_defaults_to_production_and_profiles_are_explicit(self):
        self.assertEqual(parse_arguments([]).profile, "production")
        self.assertEqual(parse_arguments(["--profile", "quick"]).profile, "quick")
        self.assertEqual(
            parse_arguments(["--profile", "validation"]).profile,
            "validation",
        )

    def test_production_run_axes_are_the_frozen_five_seed_plan(self):
        events, seeds = profiled_run_axes(PRODUCTION)
        np.testing.assert_array_equal(events, np.arange(1, 13))
        self.assertEqual(seeds, (73_241, 83_244, 93_247, 103_250, 113_253))
        settings = PRODUCTION.profiled_likelihood
        self.assertEqual(settings.pseudoexperiments_per_truth_and_seed, 100_000)
        self.assertEqual(settings.chunk_size, 5_000)
        self.assertEqual(settings.tie_tolerance, 1.0e-12)
        self.assertEqual(settings.target_accuracy, 0.90)

    def test_tables_manifest_and_cached_reuse_are_profile_scoped(self):
        settings = replace(
            QUICK.profiled_likelihood,
            pseudoexperiments_per_truth_and_seed=200,
            maximum_observed_events=3,
            chunk_size=37,
        )
        config = replace(
            QUICK,
            masses_gev=(0.3,),
            profiled_likelihood=settings,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_stage = root / "lifetime_blind_discrimination"
            input_dir = input_stage / "template_banks"
            output_dir = root / "lifetime_blind_profiled_likelihood"
            cache_dir = root / "cache"
            write_toy_bank(input_dir)
            upstream_manifest = input_stage / "manifest.json"
            upstream_manifest.write_text('{"profile":"quick","revision":1}\n')

            first = run_workflow(
                config,
                input_dir=input_dir,
                output_dir=output_dir,
                cache=CacheStore("quick", cache_dir),
                make_plots=False,
            )
            self.assertEqual(
                first.cache_stats,
                {"hits": 0, "misses": 1, "writes": 1, "rejected": 0},
            )
            detailed_path = output_dir / "tables" / "profiled_accuracy_ma_0p3.csv"
            detailed = pd.read_csv(detailed_path)
            self.assertEqual(tuple(detailed.columns), PROFILED_ACCURACY_COLUMNS)
            self.assertEqual(len(detailed), 2 * 2 * 3)
            self.assertEqual(set(detailed["seed"]), {73_241})
            self.assertEqual(set(detailed["number_of_pseudoexperiments"]), {200})

            expected_names = {
                "profiled_accuracy_ma_0p3.csv",
                "profiled_worst_case_by_seed_ma_0p3.csv",
                "profiled_conservative_curve_ma_0p3.csv",
                "profiled_threshold_ma_0p3.csv",
            }
            self.assertEqual(
                {path.name for path in (output_dir / "tables").iterdir()},
                expected_names,
            )
            summary = pd.read_csv(output_dir / "profiled_threshold_summary.csv")
            self.assertEqual(tuple(summary.columns), THRESHOLD_SUMMARY_COLUMNS)
            self.assertEqual(summary.loc[0, "minimum_persistent_events"], 1)

            manifest_text = first.manifest_path.read_text()
            manifest = json.loads(manifest_text)
            self.assertEqual(manifest["profile"], "quick")
            self.assertEqual(
                manifest["workflow"],
                "lifetime_blind_profiled_likelihood",
            )
            self.assertEqual(manifest["event_counts"], [1, 2, 3])
            self.assertEqual(manifest["pseudoexperiment_seeds"], [73_241])
            self.assertEqual(manifest["cache_stats"], first.cache_stats)
            self.assertGreaterEqual(manifest["elapsed_seconds"], 0.0)
            self.assertNotIn("/Users/", manifest_text)
            self.assertNotIn(str(root), manifest_text)

            second = run_workflow(
                config,
                input_dir=input_dir,
                output_dir=output_dir,
                cache=CacheStore("quick", cache_dir),
                make_plots=False,
            )
            self.assertEqual(
                second.cache_stats,
                {"hits": 1, "misses": 0, "writes": 0, "rejected": 0},
            )
            pd.testing.assert_frame_equal(first.summary, second.summary, check_exact=True)

            upstream_manifest.write_text('{"profile":"quick","revision":2}\n')
            invalidated = run_workflow(
                config,
                input_dir=input_dir,
                output_dir=output_dir,
                cache=CacheStore("quick", cache_dir),
                make_plots=False,
            )
            self.assertEqual(
                invalidated.cache_stats,
                {"hits": 0, "misses": 1, "writes": 1, "rejected": 0},
            )

    def test_threshold_summary_uses_the_persistent_nonmonotonic_rule(self):
        config = replace(
            QUICK,
            profiled_likelihood=replace(
                QUICK.profiled_likelihood,
                maximum_observed_events=5,
                target_accuracy=0.90,
            ),
        )
        curve = pd.DataFrame(
            {
                "mass_GeV": 0.3,
                "number_of_events": [1, 2, 3, 4, 5],
                "photon_truth_worst_accuracy": [0.89, 0.91, 0.89, 0.91, 0.92],
                "su2_truth_worst_accuracy": [0.95, 0.95, 0.95, 0.95, 0.95],
                "worst_case_correct_fraction": [0.89, 0.91, 0.89, 0.91, 0.92],
                "limiting_seed": [73_241] * 5,
                "limiting_truth_model": ["photon"] * 5,
                "limiting_truth_lifetime_index": [0] * 5,
                "limiting_truth_ctau_m": [3.0] * 5,
            }
        )
        summary = summarize_mass_threshold(toy_bank(), curve, config)
        self.assertEqual(tuple(summary), THRESHOLD_SUMMARY_COLUMNS)
        self.assertEqual(summary["minimum_persistent_events"], 4)
        self.assertEqual(summary["accuracy_at_threshold"], 0.91)


if __name__ == "__main__":
    unittest.main()
