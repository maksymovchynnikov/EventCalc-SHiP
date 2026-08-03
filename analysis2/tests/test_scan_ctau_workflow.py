from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pandas as pd

from analysis2.config import PRODUCTION
from analysis2.observable_domains import collect_observable_domains
from analysis2.paths import REPOSITORY_ROOT, profile_output_dir
from analysis2.workflows.scan_ctau_ranges import (
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

    def test_tracked_scan_reproduces_both_frozen_endpoint_conventions(self):
        scan = pd.read_csv(REPOSITORY_ROOT / "analysis" / "ctau_scan" / "ctau_scan.csv")
        domains = collect_observable_domains(scan, threshold=10.0)
        masses = (0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.05)
        expected = {
            "ALP-photon-combined": (
                (
                    275.424157376772,
                    143.585963466417,
                    85.078136106899,
                    54.619466206020,
                    30.849801427726,
                    18.963660286668,
                    14.080490604635,
                    12.108093868065,
                ),
                (
                    275.428606345867,
                    143.588287801849,
                    85.078451997369,
                    54.620301830515,
                    30.849761048645,
                    18.963762353206,
                    14.080658221100,
                    12.107944020699,
                ),
            ),
            "ALP-SU2L": (
                (
                    547.130508785662,
                    401.578196603822,
                    315.339577897365,
                    258.103945672169,
                    200.604236656845,
                    162.903993926246,
                    144.332240734155,
                    136.398974611987,
                ),
                (
                    547.122811337382,
                    401.576714636085,
                    315.342532010714,
                    258.103649736846,
                    200.602041885194,
                    162.902980970563,
                    144.334042507676,
                    136.400861829449,
                ),
            ),
        }
        for model_name, (interpolated, diagnostic) in expected.items():
            actual_interpolated = [
                domains[(model_name, mass)].upper_m for mass in masses
            ]
            actual_diagnostic = [
                domains[(model_name, mass)].bisection_upper_m for mass in masses
            ]
            np.testing.assert_allclose(
                actual_interpolated,
                interpolated,
                rtol=1.0e-12,
                atol=0.0,
            )
            np.testing.assert_allclose(
                actual_diagnostic,
                diagnostic,
                rtol=1.0e-12,
                atol=0.0,
            )


def rate_value(ctau_m: float) -> float:
    return 73.0 / ctau_m


if __name__ == "__main__":
    unittest.main()
