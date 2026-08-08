from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from alp_discrimination.config import PRODUCTION
from alp_discrimination.observable_domains import (
    collect_observable_domains,
    load_lifetime_scan,
    padded_lifetime_grid,
)
from alp_discrimination.paths import LEGACY_ANALYSIS_ROOT
from alp_discrimination.frozen_regression import (
    RegressionMismatchError,
    RegressionPaths,
    assert_regression_matches,
    compare_frozen_outputs,
)


def _scan_frame():
    rows = []
    for model, rates in (
        ("ALP-photon-combined", (20.0, 15.0, 5.0)),
        ("ALP-SU2L", (22.0, 14.0, 4.0)),
    ):
        for ctau, rate in zip((3.0, 10.0, 30.0), rates):
            rows.append(
                {
                    "model": model,
                    "mass_GeV": 0.3,
                    "ctau_m": ctau,
                    "N_events": rate,
                    "passes_event_cut": rate >= 10.0,
                }
            )
    return pd.DataFrame(rows)


def _bank_arrays():
    return {
        "mass_GeV": np.asarray(0.3),
        "energy_edges_GeV": np.array([0.3, 3.0, 400.0]),
        "minimum_bin_N_eff": np.asarray(100.0),
        "jeffreys_alpha": np.asarray(0.5),
        "event_threshold": np.asarray(10.0),
        "template_seed_offset": np.asarray(0, dtype=np.int64),
        "template_base_seed": np.asarray(54_321, dtype=np.int64),
        "photon_ctau_m": np.array([3.0, 20.0]),
        "photon_probabilities": np.array([[0.8, 0.2], [0.7, 0.3]]),
        "photon_n_events": np.array([20.0, 10.0]),
        "su2_ctau_m": np.array([3.0, 25.0]),
        "su2_probabilities": np.array([[0.4, 0.6], [0.2, 0.8]]),
        "su2_n_events": np.array([22.0, 10.0]),
    }


def _current_bank_arrays():
    arrays = _bank_arrays()
    arrays.update(
        {
            "bank_format_version": np.asarray(1),
            "profile": np.asarray("production"),
            "selection_name": np.asarray("diphoton_ecal"),
        }
    )
    return arrays


def _write_current_endpoint_artifacts(
    current_scan: Path,
    *,
    lifetime_points: int = 2,
):
    domains = collect_observable_domains(
        load_lifetime_scan(current_scan),
        threshold=10.0,
    )
    model_ids = {
        "ALP-photon-combined": "alp_photon_combined",
        "ALP-SU2L": "alp_su2l",
    }
    domain_rows = []
    diagnostic_rows = []
    for (model, mass_gev), domain in domains.items():
        grid = padded_lifetime_grid(domain, lifetime_points, 0.002)
        common = {
            "model": model,
            "model_id": model_ids[model],
            "mass_GeV": mass_gev,
        }
        domain_rows.append(
            {
                **common,
                "template_domain_lower_m": domain.lower_m,
                "template_domain_upper_m": domain.upper_m,
                "template_grid_lower_m": grid[0],
                "template_grid_upper_m": grid[-1],
                "lower_is_scan_boundary": domain.lower_is_scan_boundary,
                "upper_is_scan_boundary": domain.upper_is_scan_boundary,
                "number_of_lifetime_templates": lifetime_points,
                "template_endpoint_convention": "log_log_rate_interpolation",
                "diagnostic_endpoint_convention": (
                    "fixed_step_log_bisection_midpoint"
                ),
                "template_log_endpoint_padding_fraction": 0.002,
            }
        )
        diagnostic_rows.append(
            {
                **common,
                "bisection_diagnostic_lower_m": domain.bisection_lower_m,
                "bisection_diagnostic_upper_m": domain.bisection_upper_m,
            }
        )
    pd.DataFrame(domain_rows).to_csv(
        current_scan.with_name("observable_lifetime_domains.csv"),
        index=False,
    )
    pd.DataFrame(diagnostic_rows).to_csv(
        current_scan.with_name("bisection_diagnostic_ranges.csv"),
        index=False,
    )


def _write_artifacts(root: Path):
    reference_scan = root / "reference_scan.csv"
    current_scan = root / "current_scan.csv"
    _scan_frame().to_csv(reference_scan, index=False)
    _scan_frame().to_csv(current_scan, index=False)
    _write_current_endpoint_artifacts(current_scan)
    frozen = root / "frozen"
    current_banks = root / "current_banks"
    frozen_bank_dir = frozen / "template_banks"
    current_bank_dir = current_banks / "template_banks"
    frozen_bank_dir.mkdir(parents=True, exist_ok=True)
    current_bank_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        frozen_bank_dir / "template_bank_ma_0p3.npz",
        **_bank_arrays(),
    )
    np.savez_compressed(
        current_bank_dir / "template_bank_ma_0p3.npz",
        **_current_bank_arrays(),
    )

    frozen_distance = frozen / "distance_maps"
    current_distance = root / "current_distance"
    distance_summary = pd.DataFrame(
        [{"mass_GeV": 0.3, "minimum_D_TV": 0.4, "number_of_energy_bins": 2}]
    )
    distance_table = pd.DataFrame(
        [
            {
                "mass_GeV": 0.3,
                "photon_lifetime_index": 0,
                "su2_lifetime_index": 0,
                "D_TV": 0.4,
            }
        ]
    )
    for directory in (frozen_distance, current_distance):
        (directory / "tables").mkdir(parents=True, exist_ok=True)
        distance_summary.to_csv(directory / "distance_map_summary.csv", index=False)
        distance_table.to_csv(directory / "tables/distance_map_ma_0p3.csv", index=False)

    frozen_profiled = frozen / "profiled_likelihood"
    current_profiled = root / "current_profiled"
    thresholds = pd.DataFrame(
        [{"mass_GeV": 0.3, "minimum_persistent_events": 2, "target_accuracy": 0.9}]
    )
    by_seed = pd.DataFrame(
        [
            {
                "mass_GeV": 0.3,
                "seed": 73_241,
                "number_of_events": 1,
                "worst_case_correct_fraction": 0.89,
            },
            {
                "mass_GeV": 0.3,
                "seed": 83_244,
                "number_of_events": 1,
                "worst_case_correct_fraction": 0.88,
            },
        ]
    )
    for directory in (frozen_profiled, current_profiled):
        (directory / "tables").mkdir(parents=True, exist_ok=True)
        thresholds.to_csv(directory / "profiled_threshold_summary.csv", index=False)
        by_seed.iloc[::-1].to_csv(
            directory / "tables/profiled_worst_case_by_seed_ma_0p3.csv",
            index=False,
        )
    return RegressionPaths(
        reference_scan,
        frozen,
        current_scan,
        current_banks,
        current_distance,
        current_profiled,
    )


class FrozenRegressionTests(unittest.TestCase):
    def test_all_tracked_frozen_artifact_schemas_compare_successfully(self):
        frozen = LEGACY_ANALYSIS_ROOT / "lifetime_blind_discrimination_final"
        scan = LEGACY_ANALYSIS_ROOT / "ctau_scan" / "ctau_scan.csv"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current_scan = root / "scan_ctau_ranges" / "ctau_scan.csv"
            current_scan.parent.mkdir(parents=True)
            current_scan.write_bytes(scan.read_bytes())
            _write_current_endpoint_artifacts(
                current_scan,
                lifetime_points=PRODUCTION.templates.lifetime_points_per_model,
            )
            current_bank_root = root / "lifetime_blind_discrimination"
            current_bank_dir = current_bank_root / "template_banks"
            current_bank_dir.mkdir(parents=True)
            for frozen_bank in (frozen / "template_banks").glob("*.npz"):
                with np.load(frozen_bank, allow_pickle=False) as archive:
                    arrays = {name: archive[name] for name in archive.files}
                    arrays.update(
                        {
                            "bank_format_version": np.asarray(1),
                            "profile": np.asarray("production"),
                            "selection_name": np.asarray("diphoton_ecal"),
                        }
                    )
                    np.savez_compressed(current_bank_dir / frozen_bank.name, **arrays)
            paths = RegressionPaths(
                scan,
                frozen,
                current_scan,
                current_bank_root,
                frozen / "distance_maps",
                frozen / "profiled_likelihood",
            )
            report = compare_frozen_outputs(paths, PRODUCTION.masses_gev)
            assert_regression_matches(report)
            self.assertFalse((report["status"] == "genuine_mismatch").any())

    def test_identical_artifacts_compare_exactly_and_semantically(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_artifacts(Path(directory))
            report = compare_frozen_outputs(
                paths,
                (0.3,),
                lifetime_points=2,
            )
            assert_regression_matches(report)
            self.assertFalse((report["status"] == "genuine_mismatch").any())
            quantities = " ".join(report["quantity"])
            self.assertIn("production_padded_grid_upper_m", quantities)
            self.assertIn("fixed_step_bisection_diagnostic_upper_m", quantities)
            probability = report.loc[
                (report["category"] == "template_bank")
                & (report["quantity"] == "photon_probabilities")
            ].iloc[0]
            self.assertEqual(probability["comparison_mode"], "exact_npz")
            self.assertEqual(probability["status"], "exact_agreement")
            metadata = report.loc[
                (report["category"] == "template_bank")
                & (report["quantity"] == "metadata:profile")
            ].iloc[0]
            self.assertEqual(metadata["status"], "validated_metadata_extension")

    def test_npz_one_ulp_change_is_a_genuine_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_artifacts(Path(directory))
            arrays = _current_bank_arrays()
            arrays["photon_probabilities"][0, 0] = np.nextafter(0.8, 1.0)
            np.savez_compressed(
                paths.current_bank_root / "template_banks/template_bank_ma_0p3.npz",
                **arrays,
            )
            report = compare_frozen_outputs(paths, (0.3,), lifetime_points=2)
            row = report.loc[
                (report["category"] == "template_bank")
                & (report["quantity"] == "photon_probabilities")
            ].iloc[0]
            self.assertEqual(row["status"], "genuine_mismatch")
            with self.assertRaises(RegressionMismatchError):
                assert_regression_matches(report)

    def test_missing_or_unexpected_bank_keys_are_genuine_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_artifacts(Path(directory))
            arrays = _current_bank_arrays()
            arrays.pop("photon_n_events")
            arrays["undocumented_scientific_array"] = np.asarray(12.0)
            np.savez_compressed(
                paths.current_bank_root / "template_banks/template_bank_ma_0p3.npz",
                **arrays,
            )
            report = compare_frozen_outputs(paths, (0.3,), lifetime_points=2)
            key_rows = report.loc[
                (report["category"] == "template_bank")
                & report["quantity"].isin(
                    ("scientific_key_set", "metadata_key_set")
                )
            ].set_index("quantity")
            self.assertEqual(
                key_rows.loc["scientific_key_set", "status"],
                "genuine_mismatch",
            )
            self.assertEqual(
                key_rows.loc["metadata_key_set", "status"],
                "validated_metadata_extension",
            )

    def test_wrong_bank_provenance_is_a_genuine_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_artifacts(Path(directory))
            arrays = _current_bank_arrays()
            arrays["profile"] = np.asarray("quick")
            np.savez_compressed(
                paths.current_bank_root / "template_banks/template_bank_ma_0p3.npz",
                **arrays,
            )
            report = compare_frozen_outputs(paths, (0.3,), lifetime_points=2)
            profile = report.loc[
                (report["category"] == "template_bank")
                & (report["quantity"] == "metadata:profile")
            ].iloc[0]
            self.assertEqual(profile["status"], "genuine_mismatch")

    def test_csv_roundtrip_tolerance_is_reported_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_artifacts(Path(directory))
            path = paths.current_distance_root / "distance_map_summary.csv"
            table = pd.read_csv(path)
            table.loc[0, "minimum_D_TV"] += 1.0e-15
            table.to_csv(path, index=False)
            report = compare_frozen_outputs(paths, (0.3,), lifetime_points=2)
            row = report.loc[
                (report["category"] == "distance_summary")
                & (report["quantity"] == "minimum_D_TV")
            ].iloc[0]
            self.assertEqual(row["status"], "csv_roundtrip_agreement")
            self.assertGreater(row["max_abs_difference"], 0.0)
            assert_regression_matches(report)


if __name__ == "__main__":
    unittest.main()
