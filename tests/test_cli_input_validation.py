#!/usr/bin/env python3
"""Strict input and path-independence contracts for EventCalc drivers."""

import importlib
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_batch  # noqa: E402
import simulate  # noqa: E402
from funcs import LLP_selection  # noqa: E402


def _grid_args(**overrides):
    values = {
        "masses": [1.0],
        "mass_range": None,
        "ctaus": [1.0],
        "ctau_range": None,
        "ctau_logrange": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ValidationHelperTest(unittest.TestCase):
    def test_event_count_is_strictly_positive(self):
        self.assertEqual(LLP_selection.validate_event_count(1), 1)
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    LLP_selection.validate_event_count(value)

    def test_masses_and_lifetimes_must_be_finite_and_positive(self):
        np.testing.assert_array_equal(
            LLP_selection.validate_positive_finite(
                ("0.02", 5.27), "masses"),
            [0.02, 5.27],
        )
        for values in ((), (0,), (-1,), (math.nan,), (math.inf,)):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    LLP_selection.validate_positive_finite(values, "values")

    def test_hnl_mixing_is_finite_nonnegative_and_normalized(self):
        np.testing.assert_array_equal(
            LLP_selection.normalize_hnl_mixing([2.0, 1.0, 1.0]),
            [0.5, 0.25, 0.25],
        )
        invalid = (
            [1.0, 0.0],
            [math.nan, 0.0, 1.0],
            [math.inf, 0.0, 1.0],
            [-0.1, 0.1, 1.0],
            [0.0, 0.0, 0.0],
        )
        for mixing in invalid:
            with self.subTest(mixing=mixing):
                with self.assertRaises(ValueError):
                    LLP_selection.normalize_hnl_mixing(mixing)
        np.testing.assert_array_equal(
            LLP_selection.normalize_hnl_mixing(
                [sys.float_info.max, sys.float_info.max, 0.0]),
            [0.5, 0.5, 0.0],
        )

    def test_mass_domain_is_closed_and_canonicalizes_decimal_endpoint(self):
        lower = 0.02
        # Reproduce the installed HNL table's decimal representation.
        upper = math.nextafter(5.27, -math.inf)
        masses = LLP_selection.validate_masses_in_domain(
            [lower, 5.27], lower, upper)
        self.assertEqual(masses, [lower, upper])

        below = math.nextafter(
            math.nextafter(lower, -math.inf), -math.inf)
        above = math.nextafter(
            math.nextafter(upper, math.inf), math.inf)
        for value in (below, above):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "inclusive"):
                    LLP_selection.validate_masses_in_domain(
                        [value], lower, upper)

    def test_batch_grid_builders_reject_invalid_ranges(self):
        self.assertEqual(
            run_batch.build_masses(_grid_args(
                masses=None, mass_range=(0.02, 0.04, 0.01))),
            [0.02, 0.03, 0.04],
        )
        self.assertEqual(
            run_batch.build_masses(_grid_args(
                masses=None, mass_range=(1.0, 1.31, 0.2))),
            [1.0, 1.2],
        )
        self.assertEqual(
            run_batch.build_ctaus(_grid_args(
                ctaus=None, ctau_logrange=(0.1, 10.0, 3.0))),
            [0.1, 1.0, 10.0],
        )

        invalid_mass_ranges = (
            (0.0, 1.0, 0.1),
            (1.0, 0.5, 0.1),
            (1.0, 2.0, 0.0),
            (1.0, math.inf, 0.1),
        )
        for mass_range in invalid_mass_ranges:
            with self.subTest(mass_range=mass_range):
                with self.assertRaises(ValueError):
                    run_batch.build_masses(_grid_args(
                        masses=None, mass_range=mass_range))

        invalid_ctau_ranges = (
            {"ctau_range": (0.0, 1.0, 0.1)},
            {"ctau_range": (2.0, 1.0, 0.1)},
            {"ctau_range": (1.0, 2.0, math.nan)},
            {"ctau_logrange": (1.0, 10.0, 2.5)},
            {"ctau_logrange": (10.0, 1.0, 2.0)},
        )
        for update in invalid_ctau_ranges:
            with self.subTest(update=update):
                values = {
                    "ctaus": None,
                    "ctau_range": None,
                    "ctau_logrange": None,
                }
                values.update(update)
                args = _grid_args(**values)
                with self.assertRaises(ValueError):
                    run_batch.build_ctaus(args)


class InteractiveInputTest(unittest.TestCase):
    def test_prompted_masses_accept_both_closed_endpoints(self):
        lower = 0.02
        upper = math.nextafter(5.27, -math.inf)
        with mock.patch(
                "builtins.input",
                side_effect=["0.02 5.27", "1, 10"]):
            masses, c_taus = LLP_selection.prompt_masses_and_c_taus(
                lower, upper)
        self.assertEqual(masses, [lower, upper])
        self.assertEqual(c_taus, [[1.0, 10.0], [1.0, 10.0]])

    def test_prompted_mass_lifetime_and_mixing_reject_bad_values(self):
        invalid_pairs = (
            ("", "1"),
            ("nan", "1"),
            ("1", "0"),
            ("1", "inf"),
        )
        for mass_text, ctau_text in invalid_pairs:
            with self.subTest(mass=mass_text, ctau=ctau_text):
                with mock.patch(
                        "builtins.input",
                        side_effect=[mass_text, ctau_text]):
                    with self.assertRaises(ValueError):
                        LLP_selection.prompt_masses_and_c_taus()

        selection = {"LLP_name": "HNL"}
        for mixing_text in ("1 -1 1", "nan 0 1", "0 0 0"):
            with self.subTest(mixing=mixing_text):
                with mock.patch("builtins.input", return_value=mixing_text):
                    with self.assertRaises(ValueError):
                        LLP_selection.prompt_mixing_pattern(selection)

    def test_combined_production_choices_are_exposed_interactively(self):
        dark_photon = {"LLP_name": "Dark-photons"}
        alp_photon = {"LLP_name": "ALP-photon"}
        with mock.patch("builtins.input", return_value="3"):
            self.assertEqual(
                LLP_selection.prompt_dp_production_mode(dark_photon),
                "combined",
            )
        with mock.patch("builtins.input", return_value="4"):
            self.assertEqual(
                LLP_selection.prompt_dp_production_mode(dark_photon),
                "brem-cascade",
            )
        with mock.patch("builtins.input", return_value="3"):
            self.assertEqual(
                LLP_selection.prompt_alp_production_mode(alp_photon),
                "combined",
            )

    def test_particle_discovery_is_independent_of_caller_cwd(self):
        previous = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                with mock.patch("builtins.input", return_value="1"):
                    selected = LLP_selection.select_particle()
                self.assertTrue(os.path.isabs(selected["particle_path"]))
                self.assertTrue(os.path.isdir(selected["particle_path"]))
                self.assertEqual(
                    os.path.commonpath([
                        selected["particle_path"],
                        LLP_selection.DISTRIBUTIONS_DIR,
                    ]),
                    LLP_selection.DISTRIBUTIONS_DIR,
                )
        finally:
            os.chdir(previous)


class DriverPathAndCliTest(unittest.TestCase):
    def test_importing_simulate_has_no_prompt_or_cwd_side_effect(self):
        previous = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                with mock.patch(
                        "builtins.input",
                        side_effect=AssertionError("unexpected prompt")):
                    importlib.reload(simulate)
                self.assertEqual(
                    Path(os.getcwd()).resolve(), Path(directory).resolve())
                # the tree resolves its inputs and its output root from the
                # project directory, so nothing changes the caller's cwd
                self.assertTrue((ROOT / "Distributions").is_dir())
        finally:
            os.chdir(previous)

    def test_simulate_starts_from_an_unrelated_cwd(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / "simulate.py")],
                cwd=directory,
                input="0\n",
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0, msg=result.stdout)
        self.assertIn("SHiP setup", result.stdout)
        self.assertIn("positive integer", result.stdout)
        self.assertNotIn("No such file or directory: './Distributions'",
                         result.stdout)

    def test_batch_cli_rejects_invalid_values_before_generation(self):
        cases = (
            (
                ["--llp", "ALP-photon", "--nevents", "0",
                 "--masses", "0.1", "--ctaus", "1", "--dry-run"],
                "positive integer",
            ),
            (
                ["--llp", "ALP-photon", "--nevents", "1",
                 "--masses", "nan", "--ctaus", "1", "--dry-run"],
                "finite",
            ),
            (
                ["--llp", "ALP-photon", "--nevents", "1",
                 "--masses", "0.1", "--ctaus", "inf", "--dry-run"],
                "finite",
            ),
        )
        for arguments, expected in cases:
            with self.subTest(arguments=arguments):
                with tempfile.TemporaryDirectory() as directory:
                    result = subprocess.run(
                        [sys.executable, str(ROOT / "run_batch.py"),
                         *arguments],
                        cwd=directory,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                self.assertEqual(result.returncode, 2, msg=result.stdout)
                self.assertIn(expected, result.stdout)

    def test_interactive_seed_comes_from_the_environment(self):
        """An interactive session has no card and no command line.

        EXHAD_SEED is the only way to give one a seed, so it is read, and a
        value that is not a non-negative integer stops the run rather than
        being rounded into one.
        """
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("EXHAD_SEED", None)
            self.assertEqual(simulate._interactive_seed(), simulate.DEFAULT_SEED)
        with mock.patch.dict(os.environ, {"EXHAD_SEED": "37"}):
            self.assertEqual(simulate._interactive_seed(), 37)
        for bad in ("-1", "1.5", "seven", ""):
            with self.subTest(value=bad), \
                    mock.patch.dict(os.environ, {"EXHAD_SEED": bad}), \
                    self.assertRaisesRegex(ValueError, "EXHAD_SEED"):
                simulate._interactive_seed()

    def test_both_drivers_require_the_same_production_choices(self):
        """A flux choice is physics, so neither driver may default it.

        run_batch and simulate are one program entered two ways; a command
        that one refuses cannot be silently completed by the other.
        """
        cases = (
            (["--llp", "Dark-photons", "--dp-production", "primary"],
             ["--model", "Dark-photons", "--dp-production", "primary"],
             "Dark-photons requires 'uncertainty'"),
            (["--llp", "ALP-photon"],
             ["--model", "ALP-photon"],
             "ALP-photon requires 'alp_production_mode'"),
        )
        for batch_arguments, simulate_arguments, expected in cases:
            with self.subTest(expected=expected):
                outputs = []
                for script, arguments, tail in (
                        ("run_batch.py", batch_arguments,
                         ["--nevents", "1", "--masses", "0.3", "--ctaus", "1",
                          "--dry-run"]),
                        ("simulate.py", simulate_arguments,
                         ["--events", "1", "--masses", "0.3", "--c-taus", "1",
                          "--validate-only"])):
                    with tempfile.TemporaryDirectory() as directory:
                        result = subprocess.run(
                            [sys.executable, str(ROOT / script),
                             *arguments, *tail],
                            cwd=directory, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            check=False)
                    outputs.append(result)
                for result in outputs:
                    self.assertEqual(result.returncode, 2, msg=result.stdout)
                    self.assertIn(expected, result.stdout)

    def test_batch_cli_preserves_all_combined_source_aliases(self):
        cases = {
            "--dp-production": (
                "primary", "cascade", "brem-cascade", "combined",
                "primary+cascade", "cascade+primary",
            ),
            "--alp-production": (
                "primary", "cascade", "cascades", "combined",
                "primary+cascade", "cascade+primary",
            ),
        }
        base = [
            "run_batch.py", "--llp", "ALP-photon", "--nevents", "1",
            "--masses", "0.1", "--ctaus", "1",
        ]
        for option, choices in cases.items():
            for choice in choices:
                with self.subTest(option=option, choice=choice):
                    with mock.patch.object(
                            sys, "argv", [*base, option, choice]):
                        _, args = run_batch.parse_args()
                    attribute = option.removeprefix("--").replace("-", "_")
                    self.assertEqual(getattr(args, attribute), choice)


if __name__ == "__main__":
    unittest.main()
