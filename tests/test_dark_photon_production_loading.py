#!/usr/bin/env python3
"""End-to-end loading contract for dark-photon production sources."""

import os
import subprocess
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


class DarkPhotonProductionLoadingTest(unittest.TestCase):
    def test_batch_dry_run_loads_full_cascade_tables_and_reports_source(self):
        command = [
            sys.executable,
            os.path.join(ROOT, "run_batch.py"),
            "--llp", "Dark-photons",
            "--masses", "1.8",
            "--ctaus", "10",
            "--channels", "all",
            "--nevents", "10",
            "--exhad", "off",
            "--dp-production", "cascade",
            "--uncertainty", "central",
            "--dry-run",
        ]
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout)
        self.assertIn("DP production    : cascade", result.stdout)
        self.assertIn("DP uncertainty   : central", result.stdout)
        self.assertIn(
            "Effective per-mass routes:\n"
            "  1.8 GeV: raw-pythia-seed1",
            result.stdout,
        )


if __name__ == "__main__":
    unittest.main()
