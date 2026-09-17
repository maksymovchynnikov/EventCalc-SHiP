#!/usr/bin/env python3
"""End-to-end contracts for requested versus effective generator modes."""

import os
import subprocess
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from funcs import exhadDecays  # noqa: E402

NO_RELEASE = ("these routes need a configured exHad release; without one "
              "every mass falls back to the EventCalc baseline")


def exhad_release_is_configured():
    exhadDecays.set_selection("Dark-photons",
                              os.path.join(ROOT, "Distributions", "Dark-photons"))
    try:
        return exhadDecays.can_use_exhad()
    finally:
        exhadDecays.set_selection(None, None, None)


def run_batch(*arguments):
    env = os.environ.copy()
    env.pop("EXHAD_BENCH", None)
    env.pop("EXHAD_LLP", None)
    return subprocess.run(
        [
            sys.executable,
            os.path.join(ROOT, "run_batch.py"),
            *arguments,
            "--dry-run",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


class EffectiveModeCliTest(unittest.TestCase):
    @unittest.skipUnless(exhad_release_is_configured(), NO_RELEASE)
    def test_dark_photon_mixed_scan_reports_each_effective_route(self):
        """Both windows of the dark photon are exHad routes.

        Below the matched support the release's named exclusive rows are used
        rather than flat phase space, so a mass on either side of the 1.7 GeV
        handoff reports the exHad route: 1.0 GeV through the exclusive rows,
        1.8 GeV through the matched window.
        """
        exhadDecays.set_selection(
            "Dark-photons", os.path.join(ROOT, "Distributions", "Dark-photons"))
        try:
            self.assertTrue(exhadDecays.exclusive_enabled(1.0))
            self.assertFalse(exhadDecays.enabled(1.0))
            self.assertTrue(exhadDecays.enabled(1.8))
        finally:
            exhadDecays.set_selection(None, None, None)
        result = run_batch(
            "--llp", "Dark-photons",
            "--uncertainty", "central",
            "--dp-production", "primary",
            "--masses", "1.0", "1.8",
            "--ctaus", "10",
            "--channels", "all",
            "--nevents", "10",
            "--exhad", "auto",
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout)
        self.assertIn("1 GeV: exhad-dv-seed1", result.stdout)
        self.assertIn("1.8 GeV: exhad-dv-seed1", result.stdout)

    def test_alp_photon_explicit_on_is_a_direct_noop(self):
        result = run_batch(
            "--llp", "ALP-photon",
            "--alp-production", "primary",
            "--masses", "0.1",
            "--ctaus", "10",
            "--channels", "all",
            "--nevents", "10",
            "--exhad", "on",
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout)
        self.assertIn("not applicable", result.stdout)
        self.assertIn(
            "0.1 GeV: eventcalc-direct-seed1", result.stdout
        )

    @unittest.skipUnless(exhad_release_is_configured(), NO_RELEASE)
    def test_zero_rate_hnl_currents_do_not_claim_exhad(self):
        result = run_batch(
            "--llp", "HNL",
            "--mixing", "1", "0", "0",
            "--masses", "1.0",
            "--ctaus", "10",
            "--channels", "all",
            "--nevents", "10",
            "--exhad", "on",
        )
        self.assertEqual(result.returncode, 0, msg=result.stdout)
        effective = result.stdout.split("Effective per-mass routes:", 1)[1]
        self.assertIn("1 GeV: raw-pythia-seed1-src", effective)
        self.assertNotIn("exhad-hnl", effective)


if __name__ == "__main__":
    unittest.main()
