from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pandas as pd

from alp_discrimination.config import PRODUCTION
from alp_discrimination.physics.observable_domains import collect_observable_domains
from alp_discrimination.paths import REPOSITORY_ROOT, profile_output_dir
from alp_discrimination.workflows.scan_ctau_ranges import (
    STAGE_NAME,
    fixed_step_log_bisection_midpoint,
    parse_arguments,
    run_scan_ctau_ranges,
)


class FakeCache:
    def __init__(self, root: Path):
        self.root = root

    def counter_snapshot(self) -> dict[str, int]:
        return {"hits": 0, "misses": 36, "writes": 36, "rejected": 0}


class InverseRateAdapter:
    crossings = {
        "alp_photon_combined": 7.3,
        "alp_su2l": 9.1,
    }

    def __init__(self, cache_root: Path):
        self.cache = FakeCache(cache_root)
        self.calls = []

    def evaluate_model(
        self,
        model_id: str,
        mass_gev: float,
        ctau_m: float,
        model_seed: int,
        stage: str,
    ):
        self.calls.append((model_id, mass_gev, ctau_m, model_seed, stage))
        expected_events = 10.0 * self.crossings[model_id] / ctau_m
        before_ecal = 2.0 * expected_events
        if model_id == "alp_photon_combined":
            source_events = {
                "primary": 0.8 * expected_events,
                "cascade": 0.2 * expected_events,
            }
        else:
            source_events = {"inclusive": expected_events}
        return SimpleNamespace(
            expected_events=expected_events,
            preselection_expected_events=before_ecal,
            coupling_squared_gev_inv2=1.0 / ctau_m,
            source_expected_events=source_events,
            preselection_samples=100,
            accepted_samples=80,
            cache_key=f"fake-{model_id}-{ctau_m:.17g}",
        )


class ScanCtauWorkflowTests(unittest.TestCase):
    def test_cli_defaults_to_production_and_profiles_are_explicit(self):
        self.assertEqual(parse_arguments([]).profile, "production")
        self.assertEqual(parse_arguments(["--profile", "quick"]).profile, "quick")
        self.assertEqual(
            profile_output_dir("validation", STAGE_NAME).parts[-2:],
            ("validation", "scan_ctau_ranges"),
        )

    def test_bisection_has_exactly_fourteen_evaluations_and_unevaluated_return(self):
        evaluated = []

        def rate(ctau_m: float) -> float:
            evaluated.append(ctau_m)
            return 73.0 / ctau_m

        midpoint = fixed_step_log_bisection_midpoint(
            rate,
            6.0,
            12.0,
            threshold=10.0,
            steps=14,
            left_passes=True,
            right_passes=False,
        )
        self.assertEqual(len(evaluated), 14)
        self.assertEqual(len(set(evaluated)), 14)
        self.assertNotIn(midpoint, evaluated)
        final_left = max([6.0, *(value for value in evaluated if rate_value(value) >= 10.0)])
        final_right = min([12.0, *(value for value in evaluated if rate_value(value) < 10.0)])
        self.assertEqual(midpoint, np.sqrt(final_left * final_right))

    def test_saved_scan_drives_interpolated_padded_and_diagnostic_endpoints(self):
        config = replace(
            PRODUCTION,
            masses_gev=(0.3,),
            lifetimes=replace(
                PRODUCTION.lifetimes,
                maximum_ctau_m=20.0,
                coarse_factor=2.0,
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = InverseRateAdapter(root / "cache" / config.name)
            artifacts = run_scan_ctau_ranges(
                config,
                adapter,
                output_dir=root / "outputs" / config.name / STAGE_NAME,
            )

            scan = pd.read_csv(artifacts["scan"])
            domains = pd.read_csv(artifacts["domains"])
            diagnostic = pd.read_csv(artifacts["bisection_diagnostics"])
            self.assertEqual(len(scan), 2 * (4 + 14))
            self.assertEqual(
                scan["scan_point_kind"].value_counts().to_dict(),
                {"bisection_evaluation": 28, "coarse": 8},
            )
            self.assertEqual(len(adapter.calls), len(scan))
            self.assertTrue(all(call[-1] == "ctau" for call in adapter.calls))

            for model_id, crossing in adapter.crossings.items():
                row = domains.loc[domains["model_id"] == model_id].iloc[0]
                self.assertAlmostEqual(row["template_domain_lower_m"], 3.0)
                self.assertAlmostEqual(
                    row["template_domain_upper_m"],
                    crossing,
                    places=13,
                )
                expected_padded_upper = np.exp(
                    np.log(crossing)
                    - 0.002 * (np.log(crossing) - np.log(3.0))
                )
                self.assertAlmostEqual(row["template_grid_lower_m"], 3.0)
                self.assertAlmostEqual(
                    row["template_grid_upper_m"],
                    expected_padded_upper,
                    places=13,
                )
                self.assertEqual(row["number_of_lifetime_templates"], 20)
                model_scan = scan.loc[scan["model_id"] == model_id]
                self.assertNotIn(
                    row["bisection_diagnostic_upper_m"],
                    set(model_scan["ctau_m"]),
                )
                diagnostic_row = diagnostic.loc[
                    diagnostic["model_id"] == model_id
                ].iloc[0]
                self.assertEqual(
                    diagnostic_row["bisection_diagnostic_upper_m"],
                    row["bisection_diagnostic_upper_m"],
                )

            self.assertTrue(artifacts["plot"].is_file())
            manifest_text = artifacts["manifest"].read_text()
            manifest = json.loads(manifest_text)
            self.assertNotIn("/Users/", manifest_text)
            self.assertEqual(manifest["profile"], "production")
            self.assertFalse(
                manifest["returned_bisection_midpoints_saved_to_scan_table"]
            )
            self.assertTrue(
                manifest["returned_bisection_midpoints_saved_to_diagnostic_table"]
            )
            self.assertEqual(
                manifest["template_log_endpoint_padding_fraction"],
                0.002,
            )
            self.assertTrue(
                all(not Path(value).is_absolute() for value in manifest["artifacts"])
            )

def rate_value(ctau_m: float) -> float:
    return 73.0 / ctau_m


if __name__ == "__main__":
    unittest.main()
