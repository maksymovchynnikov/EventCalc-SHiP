import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from analysis2.frozen_regression import RegressionMismatchError, RegressionPaths
from analysis2.workflows.compare_frozen_results import (
    parse_arguments,
    resolve_paths,
    run_workflow,
    write_regression_outputs,
)


class CompareFrozenResultsWorkflowTests(unittest.TestCase):
    def test_cli_defaults_to_production(self):
        args = parse_arguments([])
        self.assertEqual(args.profile, "production")
        self.assertIsNone(args.current_root)

    def test_default_endpoint_artifacts_come_from_scan_workflow_output(self):
        args = parse_arguments(["--current-root", "portable/current"])
        paths = resolve_paths(args, "production")
        scan_root = Path("portable/current/scan_ctau_ranges")
        self.assertEqual(paths.current_scan_path, scan_root / "ctau_scan.csv")
        self.assertEqual(
            paths.resolved_current_domain_path,
            scan_root / "observable_lifetime_domains.csv",
        )
        self.assertEqual(
            paths.resolved_current_bisection_diagnostic_path,
            scan_root / "bisection_diagnostic_ranges.csv",
        )

    def test_output_paths_are_portable_and_mismatch_is_written_before_raise(self):
        report = pd.DataFrame(
            [
                {
                    "category": "toy",
                    "artifact": "toy.csv",
                    "quantity": "value",
                    "comparison_mode": "csv_roundtrip",
                    "reference_count": 1,
                    "current_count": 1,
                    "max_abs_difference": 1.0,
                    "max_relative_difference": 1.0,
                    "absolute_tolerance": 5.0e-15,
                    "relative_tolerance": 1.0e-14,
                    "status": "genuine_mismatch",
                    "details": "different",
                }
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = RegressionPaths(
                root / "reference.csv",
                root / "frozen",
                root / "current.csv",
                root / "banks",
                root / "distance",
                root / "profiled",
            )
            output = root / "regression"
            csv_path, json_path = write_regression_outputs(
                report,
                output,
                profile="production",
                paths=paths,
            )
            self.assertTrue(csv_path.is_file())
            payload = json.loads(json_path.read_text())
            self.assertEqual(payload["status"], "genuine_mismatch")
            self.assertIn("current_observable_domains", payload["inputs"])
            self.assertIn("current_bisection_diagnostics", payload["inputs"])
            serialized = json.dumps(payload)
            self.assertNotIn(directory, serialized)
            self.assertNotIn("/Users/", serialized)

            with patch(
                "analysis2.workflows.compare_frozen_results.compare_frozen_outputs",
                return_value=report,
            ):
                with self.assertRaises(RegressionMismatchError):
                    run_workflow(
                        profile="production",
                        paths=paths,
                        output_dir=output,
                        masses=(0.3,),
                    )
            self.assertTrue((output / "frozen_numerical_comparison.csv").is_file())


if __name__ == "__main__":
    unittest.main()
