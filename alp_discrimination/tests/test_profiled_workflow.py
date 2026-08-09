from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from alp_discrimination.cache import CacheStore
from alp_discrimination.config import PRODUCTION, QUICK
from alp_discrimination.templates.lifetime_banks import (
    LifetimeTemplateBank,
    load_template_bank,
    save_bank_artifacts,
)
from alp_discrimination.statistics.profiled import (
    PROFILED_ACCURACY_COLUMNS,
    run_profiled_seed,
)
from alp_discrimination.workflows.profiled_likelihood_cache import (
    CACHE_KIND,
    WORKFLOW_FORMAT_VERSION,
    input_fingerprint,
)
from alp_discrimination.workflows.lifetime_blind_profiled_likelihood import (
    PROFILED_ACCURACY_WITH_DOMAIN_COLUMNS,
    THRESHOLD_SUMMARY_COLUMNS,
    parse_arguments,
    load_truth_subset_table,
    parse_event_count_grid,
    profiled_run_axes,
    run_workflow,
    summarize_mass_threshold,
)


def toy_bank(
    profile: str = "quick",
    *,
    selection_name: str = "diphoton_ecal",
    minimum_photon_energy_gev: float | None = None,
) -> LifetimeTemplateBank:
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
        photon_interval_m=np.array([2.0, 40.0]),
        photon_interval_index=np.array([0, 1]),
        photon_allowed_intervals_m=np.array([[2.0, 4.0], [20.0, 40.0]]),
        su2_ctau_m=np.array([4.0, 40.0]),
        su2_probabilities=np.array([[0.01, 0.99], [0.02, 0.98]]),
        su2_n_events=np.array([40.0, 10.0]),
        su2_n_events_before_ecal=np.array([80.0, 20.0]),
        su2_epsilon_ecal_weighted=np.array([0.5, 0.5]),
        su2_total_n_eff=np.array([1000.0, 1000.0]),
        su2_interval_m=np.array([3.0, 50.0]),
        su2_interval_index=np.array([0, 1]),
        su2_allowed_intervals_m=np.array([[3.0, 5.0], [30.0, 50.0]]),
        profile=profile,
        selection_name=selection_name,
        minimum_photon_energy_gev=minimum_photon_energy_gev,
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
    assert loaded.minimum_photon_energy_gev is None


def test_saved_energy_selection_bank_records_one_gev_threshold(tmp_path):
    bank = toy_bank(
        selection_name="diphoton_ecal_e1gev",
        minimum_photon_energy_gev=1.0,
    )
    path = tmp_path / "template_bank_ma_0p3_e1gev.npz"
    save_bank_artifacts(
        bank,
        bank_path=path,
        summary_path=tmp_path / "template_bank_summary_ma_0p3_e1gev.csv",
        probability_path=tmp_path / "template_probabilities_ma_0p3_e1gev.csv",
    )

    loaded = load_template_bank(path)
    assert loaded.selection_name == "diphoton_ecal_e1gev"
    assert loaded.minimum_photon_energy_gev == 1.0
    with np.load(path, allow_pickle=False) as archive:
        assert int(archive["bank_format_version"]) == 3
        assert float(archive["minimum_photon_energy_GeV"]) == 1.0


def test_version_two_geometry_bank_loads_without_energy_threshold(tmp_path):
    arrays = toy_bank().arrays()
    arrays["bank_format_version"] = np.asarray(2)
    arrays.pop("minimum_photon_energy_GeV")
    path = tmp_path / "legacy_v2_geometry_bank.npz"
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)

    loaded = load_template_bank(path)
    assert loaded.selection_name == "diphoton_ecal"
    assert loaded.minimum_photon_energy_gev is None


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
                "number_of_photon_allowed_intervals",
                "number_of_su2_allowed_intervals",
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
                "limiting_truth_interval_index_at_threshold",
                "limiting_truth_ctau_m_at_threshold",
            ),
        )

    def test_cli_defaults_to_production_and_profiles_are_explicit(self):
        self.assertEqual(parse_arguments([]).profile, "production")
        self.assertEqual(parse_arguments([]).workers, 1)
        self.assertEqual(parse_arguments(["--workers", "2"]).workers, 2)
        self.assertEqual(
            parse_arguments(
                ["--truth-subset-path", "truth-subset.csv"]
            ).truth_subset_path,
            Path("truth-subset.csv"),
        )
        self.assertEqual(parse_arguments(["--profile", "quick"]).profile, "quick")
        self.assertEqual(
            parse_arguments(["--profile", "validation"]).profile,
            "validation",
        )
        parsed = parse_arguments(
            ["--output-dir", "custom-profiled-output"]
        )
        self.assertEqual(
            parsed.output_dir,
            Path("custom-profiled-output"),
        )

    def test_explicit_event_count_grid_is_inclusive_sorted_and_deduplicated(self):
        np.testing.assert_array_equal(
            parse_event_count_grid("1:10:3,5,8:10,4"),
            np.array([1, 4, 5, 7, 8, 9, 10]),
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_event_count_grid("0:3")
        with self.assertRaisesRegex(ValueError, "STOP < START"):
            parse_event_count_grid("5:2")
        with self.assertRaisesRegex(ValueError, "steps must be positive"):
            parse_event_count_grid("1:5:0")

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
                {"hits": 0, "misses": 4, "writes": 4, "rejected": 0},
            )
            detailed_path = output_dir / "tables" / "profiled_accuracy_ma_0p3.csv"
            detailed = pd.read_csv(detailed_path)
            self.assertEqual(
                tuple(detailed.columns),
                PROFILED_ACCURACY_WITH_DOMAIN_COLUMNS,
            )
            self.assertEqual(set(detailed["truth_interval_index"]), {0, 1})
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
            self.assertTrue(manifest["shape_only"])
            self.assertTrue(manifest["conditioned_on_observed_event_count"])
            self.assertFalse(manifest["expected_event_rates_used_in_likelihood"])
            self.assertTrue(manifest["independent_lifetime_profiling_by_model"])
            self.assertTrue(
                manifest["disconnected_domains_profiled_as_saved_template_unions"]
            )
            self.assertEqual(
                manifest["allowed_lifetime_domains"][0][
                    "photon_allowed_intervals_m"
                ],
                [[2.0, 4.0], [20.0, 40.0]],
            )
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
                {"hits": 4, "misses": 0, "writes": 0, "rejected": 0},
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
                {"hits": 0, "misses": 4, "writes": 4, "rejected": 0},
            )

    def test_cached_superset_is_sliced_without_recomputing_requested_counts(self):
        settings = replace(
            QUICK.profiled_likelihood,
            pseudoexperiments_per_truth_and_seed=100,
            maximum_observed_events=5,
            chunk_size=23,
        )
        config = replace(QUICK, profiled_likelihood=settings)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_stage = root / "lifetime_blind_discrimination"
            input_dir = input_stage / "template_banks"
            cache_dir = root / "cache"
            write_toy_bank(input_dir)
            (input_stage / "manifest.json").write_text(
                '{"profile":"quick","revision":1}\n'
            )

            full = run_workflow(
                config,
                input_dir=input_dir,
                output_dir=root / "full",
                cache=CacheStore("quick", cache_dir),
                make_plots=False,
                event_counts=np.arange(1, 6),
            )
            self.assertEqual(
                full.cache_stats,
                {"hits": 0, "misses": 4, "writes": 4, "rejected": 0},
            )

            subset_config = replace(
                config,
                profiled_likelihood=replace(
                    config.profiled_likelihood,
                    chunk_size=7,
                ),
            )
            subset = run_workflow(
                subset_config,
                input_dir=input_dir,
                output_dir=root / "subset",
                cache=CacheStore("quick", cache_dir),
                make_plots=False,
                event_counts=np.array([2, 4]),
            )
            self.assertEqual(
                subset.cache_stats,
                {"hits": 4, "misses": 4, "writes": 4, "rejected": 0},
            )
            full_table = pd.read_csv(
                root / "full" / "tables" / "profiled_accuracy_ma_0p3.csv"
            )
            subset_table = pd.read_csv(
                root / "subset" / "tables" / "profiled_accuracy_ma_0p3.csv"
            )
            expected = full_table.loc[
                full_table["number_of_events"].isin([2, 4])
            ].reset_index(drop=True)
            pd.testing.assert_frame_equal(
                subset_table.reset_index(drop=True),
                expected,
                check_exact=True,
            )

    def test_truth_subset_checkpoints_are_reused_by_a_larger_truth_run(self):
        settings = replace(
            QUICK.profiled_likelihood,
            pseudoexperiments_per_truth_and_seed=60,
            maximum_observed_events=2,
            chunk_size=17,
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
            cache_dir = root / "cache"
            write_toy_bank(input_dir)
            (input_stage / "manifest.json").write_text(
                '{"profile":"quick","revision":1}\n'
            )
            subset_path = root / "truth_subset.csv"
            pd.DataFrame(
                [
                    {
                        "mass_GeV": 0.3,
                        "truth_model": "photon",
                        "truth_lifetime_index": 0,
                        "truth_ctau_m": 3.0,
                        "truth_interval_index": 0,
                    },
                    {
                        "mass_GeV": 0.3,
                        "truth_model": "su2",
                        "truth_lifetime_index": 1,
                        "truth_ctau_m": 40.0,
                        "truth_interval_index": 1,
                    },
                ]
            ).to_csv(subset_path, index=False)

            loaded_subset = load_truth_subset_table(subset_path)
            self.assertEqual(len(loaded_subset), 2)

            subset = run_workflow(
                config,
                input_dir=input_dir,
                output_dir=root / "subset_output",
                cache=CacheStore("quick", cache_dir),
                make_plots=False,
                truth_subset_path=subset_path,
            )
            self.assertEqual(
                subset.cache_stats,
                {"hits": 0, "misses": 2, "writes": 2, "rejected": 0},
            )
            subset_summary = subset.summary.iloc[0]
            self.assertEqual(subset_summary["truth_grid"], "custom_subset")
            self.assertEqual(
                int(subset_summary["number_of_photon_truth_lifetimes"]),
                1,
            )
            self.assertEqual(
                int(subset_summary["number_of_su2_truth_lifetimes"]),
                1,
            )
            self.assertEqual(
                int(subset_summary["number_of_photon_profile_lifetimes"]),
                2,
            )
            self.assertEqual(
                int(subset_summary["number_of_su2_profile_lifetimes"]),
                2,
            )
            subset_manifest = json.loads(subset.manifest_path.read_text())
            self.assertTrue(subset_manifest["threshold_is_screening_only"])
            self.assertFalse(subset_manifest["complete_truth_domain_coverage"])
            self.assertFalse(
                subset_manifest["profile_lifetime_grid_reduced_by_truth_subset"]
            )

            full = run_workflow(
                config,
                input_dir=input_dir,
                output_dir=root / "full_output",
                cache=CacheStore("quick", cache_dir),
                make_plots=False,
            )
            self.assertEqual(
                full.cache_stats,
                {"hits": 2, "misses": 2, "writes": 2, "rejected": 0},
            )
            full_detailed = pd.read_csv(
                root / "full_output" / "tables"
                / "profiled_accuracy_ma_0p3.csv"
            )
            self.assertEqual(
                set(
                    zip(
                        full_detailed["truth_model"],
                        full_detailed["truth_lifetime_index"],
                    )
                ),
                {
                    ("photon", 0),
                    ("photon", 1),
                    ("su2", 0),
                    ("su2", 1),
                },
            )

    def test_truth_subset_requires_explicit_output_directory(self):
        settings = replace(
            QUICK.profiled_likelihood,
            pseudoexperiments_per_truth_and_seed=10,
            maximum_observed_events=1,
        )
        config = replace(QUICK, profiled_likelihood=settings)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "template_banks"
            write_toy_bank(input_dir)
            subset_path = root / "truth_subset.csv"
            pd.DataFrame(
                [
                    {
                        "mass_GeV": 0.3,
                        "truth_model": "photon",
                        "truth_lifetime_index": 0,
                    },
                    {
                        "mass_GeV": 0.3,
                        "truth_model": "su2",
                        "truth_lifetime_index": 0,
                    },
                ]
            ).to_csv(subset_path, index=False)
            with self.assertRaisesRegex(ValueError, "explicit output"):
                run_workflow(
                    config,
                    input_dir=input_dir,
                    cache=CacheStore("quick", root / "cache"),
                    make_plots=False,
                    truth_subset_path=subset_path,
                )

    def test_progressive_truth_extension_reuses_lower_statistics_exactly(self):
        lower_settings = replace(
            QUICK.profiled_likelihood,
            pseudoexperiments_per_truth_and_seed=60,
            maximum_observed_events=3,
            chunk_size=17,
        )
        lower_config = replace(
            QUICK,
            masses_gev=(0.3,),
            profiled_likelihood=lower_settings,
        )
        higher_config = replace(
            lower_config,
            profiled_likelihood=replace(
                lower_settings,
                pseudoexperiments_per_truth_and_seed=100,
                chunk_size=11,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_stage = root / "lifetime_blind_discrimination"
            input_dir = input_stage / "template_banks"
            write_toy_bank(input_dir)
            (input_stage / "manifest.json").write_text(
                '{"profile":"quick","revision":1}\n'
            )
            cache_dir = root / "progressive_cache"

            lower = run_workflow(
                lower_config,
                input_dir=input_dir,
                output_dir=root / "lower",
                cache=CacheStore("quick", cache_dir),
                make_plots=False,
            )
            self.assertEqual(
                lower.cache_stats,
                {"hits": 0, "misses": 4, "writes": 4, "rejected": 0},
            )

            staged = run_workflow(
                higher_config,
                input_dir=input_dir,
                output_dir=root / "staged",
                cache=CacheStore("quick", cache_dir),
                make_plots=False,
            )
            self.assertEqual(
                staged.cache_stats,
                {"hits": 4, "misses": 4, "writes": 4, "rejected": 0},
            )

            direct = run_workflow(
                higher_config,
                input_dir=input_dir,
                output_dir=root / "direct",
                cache=CacheStore("quick", root / "direct_cache"),
                make_plots=False,
            )

            staged_table = pd.read_csv(
                root / "staged" / "tables" / "profiled_accuracy_ma_0p3.csv"
            )
            direct_table = pd.read_csv(
                root / "direct" / "tables" / "profiled_accuracy_ma_0p3.csv"
            )
            exact_columns = [
                "mass_GeV",
                "seed",
                "truth_model",
                "truth_lifetime_index",
                "truth_interval_index",
                "truth_ctau_m",
                "number_of_events",
                "number_of_pseudoexperiments",
                "correct_fraction",
                "selected_photon_fraction",
                "selected_su2_fraction",
                "tie_fraction",
            ]
            pd.testing.assert_frame_equal(
                staged_table[exact_columns],
                direct_table[exact_columns],
                check_exact=True,
            )
            np.testing.assert_allclose(
                staged_table[
                    ["mean_profile_statistic_T", "std_profile_statistic_T"]
                ],
                direct_table[
                    ["mean_profile_statistic_T", "std_profile_statistic_T"]
                ],
                rtol=0.0,
                atol=5.0e-14,
            )
            manifest = json.loads(staged.manifest_path.read_text())
            self.assertTrue(
                manifest["progressive_truth_level_pseudoexperiment_caching"]
            )
            self.assertEqual(
                manifest["pseudoexperiment_ranges_by_mass_seed"][0][
                    "contributing_ranges"
                ],
                [[0, 60], [60, 100]],
            )
            truth_metadata = [
                json.loads(path.read_text())
                for path in (
                    cache_dir / "profiled_truth_pseudoexperiments"
                ).glob("*.json")
                if json.loads(path.read_text())["identity"]["settings"][
                    "pseudoexperiments_per_truth_and_seed"
                ]
                == 100
            ]
            self.assertEqual(len(truth_metadata), 4)
            self.assertTrue(
                all(
                    metadata["rng_state_resume_used"]
                    and metadata["rng_state_after"] is not None
                    for metadata in truth_metadata
                )
            )

    def test_legacy_seed_cache_is_migrated_before_progressive_extension(self):
        lower_settings = replace(
            QUICK.profiled_likelihood,
            pseudoexperiments_per_truth_and_seed=40,
            maximum_observed_events=2,
            chunk_size=13,
        )
        lower_config = replace(QUICK, profiled_likelihood=lower_settings)
        higher_config = replace(
            lower_config,
            profiled_likelihood=replace(
                lower_settings,
                pseudoexperiments_per_truth_and_seed=70,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_stage = root / "lifetime_blind_discrimination"
            input_dir = input_stage / "template_banks"
            bank_path = write_toy_bank(input_dir)
            (input_stage / "manifest.json").write_text(
                '{"profile":"quick","revision":1}\n'
            )
            bank = load_template_bank(bank_path)
            event_counts = np.array([1, 2])
            seed = lower_settings.seeds[0]
            legacy = run_profiled_seed(
                mass_gev=bank.mass_gev,
                photon_ctau_m=bank.photon_ctau_m,
                photon_probabilities=bank.photon_probabilities,
                su2_ctau_m=bank.su2_ctau_m,
                su2_probabilities=bank.su2_probabilities,
                event_counts=event_counts,
                number_of_pseudoexperiments=40,
                seed=seed,
                chunk_size=lower_settings.chunk_size,
                tie_tolerance=lower_settings.tie_tolerance,
                truth_grid=lower_settings.truth_lifetime_grid,
                profile_grid=lower_settings.profile_lifetime_grid,
            )
            legacy_identity = {
                "workflow_format_version": WORKFLOW_FORMAT_VERSION,
                "input": input_fingerprint(bank_path),
                "mass_gev": bank.mass_gev,
                "seed": int(seed),
                "settings": asdict(lower_settings),
                "event_counts": event_counts.tolist(),
            }
            cache_dir = root / "cache"
            cache = CacheStore("quick", cache_dir)
            cache.save(
                CACHE_KIND,
                legacy_identity,
                {
                    column: legacy[column].to_numpy(
                        dtype=str if column == "truth_model" else None
                    )
                    for column in PROFILED_ACCURACY_COLUMNS
                },
                {
                    "columns": PROFILED_ACCURACY_COLUMNS,
                    "rows": len(legacy),
                },
            )

            subset_path = root / "truth_subset.csv"
            pd.DataFrame(
                [
                    {
                        "mass_GeV": 0.3,
                        "truth_model": "photon",
                        "truth_lifetime_index": 0,
                    },
                    {
                        "mass_GeV": 0.3,
                        "truth_model": "su2",
                        "truth_lifetime_index": 1,
                    },
                ]
            ).to_csv(subset_path, index=False)

            staged = run_workflow(
                higher_config,
                input_dir=input_dir,
                output_dir=root / "staged",
                cache=CacheStore("quick", cache_dir),
                make_plots=False,
                truth_subset_path=subset_path,
            )
            self.assertEqual(
                staged.cache_stats,
                {"hits": 3, "misses": 2, "writes": 4, "rejected": 0},
            )
            manifest = json.loads(staged.manifest_path.read_text())
            self.assertEqual(
                manifest["pseudoexperiment_ranges_by_mass_seed"][0][
                    "contributing_ranges"
                ],
                [[0, 40], [40, 70]],
            )

    def test_two_worker_execution_matches_serial_exactly(self):
        settings = replace(
            QUICK.profiled_likelihood,
            pseudoexperiments_per_truth_and_seed=80,
            number_of_seeds=2,
            maximum_observed_events=3,
            chunk_size=19,
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
            write_toy_bank(input_dir)
            (input_stage / "manifest.json").write_text(
                '{"profile":"quick","revision":1}\n'
            )

            serial = run_workflow(
                config,
                input_dir=input_dir,
                output_dir=root / "serial_output",
                cache=CacheStore("quick", root / "serial_cache"),
                make_plots=False,
                workers=1,
            )
            parallel = run_workflow(
                config,
                input_dir=input_dir,
                output_dir=root / "parallel_output",
                cache=CacheStore("quick", root / "parallel_cache"),
                make_plots=False,
                workers=2,
            )

            expected_cache_stats = {
                "hits": 0,
                "misses": 8,
                "writes": 8,
                "rejected": 0,
            }
            self.assertEqual(serial.cache_stats, expected_cache_stats)
            self.assertEqual(parallel.cache_stats, expected_cache_stats)
            pd.testing.assert_frame_equal(
                parallel.summary,
                serial.summary,
                check_exact=True,
            )

            serial_detailed = pd.read_csv(
                root / "serial_output" / "tables"
                / "profiled_accuracy_ma_0p3.csv"
            )
            parallel_detailed = pd.read_csv(
                root / "parallel_output" / "tables"
                / "profiled_accuracy_ma_0p3.csv"
            )
            pd.testing.assert_frame_equal(
                parallel_detailed,
                serial_detailed,
                check_exact=True,
            )
            self.assertEqual(
                tuple(parallel_detailed["seed"].drop_duplicates()),
                settings.seeds,
            )

            serial_manifest = json.loads(serial.manifest_path.read_text())
            parallel_manifest = json.loads(parallel.manifest_path.read_text())
            self.assertEqual(serial_manifest["workers"], 1)
            self.assertEqual(parallel_manifest["workers"], 2)
            self.assertEqual(
                parallel_manifest["pseudoexperiment_seeds"],
                list(settings.seeds),
            )

    def test_workers_above_two_are_rejected(self):
        settings = replace(
            QUICK.profiled_likelihood,
            pseudoexperiments_per_truth_and_seed=10,
            maximum_observed_events=1,
        )
        config = replace(QUICK, profiled_likelihood=settings)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "template_banks"
            write_toy_bank(input_dir)
            with self.assertRaisesRegex(ValueError, "one or two"):
                run_workflow(
                    config,
                    input_dir=input_dir,
                    output_dir=root / "output",
                    cache=CacheStore("quick", root / "cache"),
                    make_plots=False,
                    workers=3,
                )

    def test_bank_discovery_is_not_limited_by_legacy_config_masses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = toy_bank()
            second = replace(toy_bank(), mass_gev=2.5)
            save_bank_artifacts(
                first,
                bank_path=root / "template_bank_ma_0p3.npz",
                summary_path=root / "summary_0p3.csv",
                probability_path=root / "probabilities_0p3.csv",
            )
            save_bank_artifacts(
                second,
                bank_path=root / "template_bank_ma_2p5.npz",
                summary_path=root / "summary_2p5.csv",
                probability_path=root / "probabilities_2p5.csv",
            )
            from alp_discrimination.workflows.lifetime_blind_profiled_likelihood import (
                resolve_bank_paths,
            )

            resolved = resolve_bank_paths(QUICK, input_dir=root)
            self.assertEqual(
                [load_template_bank(path).mass_gev for path in resolved],
                [0.3, 2.5],
            )
            selected = resolve_bank_paths(QUICK, input_dir=root, masses=[2.5])
            self.assertEqual(len(selected), 1)
            self.assertEqual(load_template_bank(selected[0]).mass_gev, 2.5)

    def test_disconnected_domains_require_the_full_truth_and_profile_grids(self):
        settings = replace(
            QUICK.profiled_likelihood,
            truth_lifetime_grid="even",
            pseudoexperiments_per_truth_and_seed=10,
            maximum_observed_events=1,
        )
        config = replace(QUICK, profiled_likelihood=settings)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "template_banks"
            write_toy_bank(input_dir)
            with self.assertRaisesRegex(ValueError, "Disconnected Week-8"):
                run_workflow(
                    config,
                    input_dir=input_dir,
                    output_dir=root / "output",
                    cache=CacheStore("quick", root / "cache"),
                    make_plots=False,
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
        self.assertEqual(summary["limiting_truth_interval_index_at_threshold"], 0)


if __name__ == "__main__":
    unittest.main()
