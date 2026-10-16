"""Coupling input must be an alternative parameterization, not new physics."""

import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import run_batch
import simulate
from examples.coupling_scan import coupling_grid, example_card
from funcs.coupling_conventions import (
    COUPLING_LABELS, coupling_to_squared, ctau_from_coupling,
    native_coupling_squared,
)
from funcs.simulation_config import config_from_mapping, config_from_command_line


def card(**changes):
    values = dict(model="ALP-photon", events=10, masses=[0.1, 0.3],
                  alp_production_mode="primary", exhad_mode="off",
                  couplings=[1e-6, 3e-6])
    values.update(changes)
    return values


class CouplingConversionTest(unittest.TestCase):
    def test_native_roundtrip_for_every_model(self):
        for name in COUPLING_LABELS:
            with self.subTest(model=name):
                coefficient = 2.3e-12
                for g in (1e-7, 1e-3, 1.0):
                    ctau = ctau_from_coupling(name, coefficient, g)
                    self.assertEqual(ctau, ctau_from_coupling(
                        name, coefficient, g*g, squared=True))
                    self.assertAlmostEqual(
                        native_coupling_squared(name, coefficient, ctau)/(g*g), 1)

    def test_invalid_and_unrepresentable_values(self):
        for value in (0, -1, float("nan"), float("inf"), True, 1e-300, 1e300):
            with self.subTest(value=value), self.assertRaises(ValueError):
                coupling_to_squared(value)
        for coefficient in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                ctau_from_coupling("HNL", coefficient, 1e-6)
        with self.assertRaises(ValueError):
            ctau_from_coupling("unknown", 1, 1)
        with self.assertRaises(ValueError):
            ctau_from_coupling("HNL", 1e300, 1e-200, squared=True)

    def test_hnl_total_mixing_and_flavour_widths(self):
        from funcs.HNLmerging import get_ctau_func
        widths = (np.array([1., 2.]), np.array([1., 2.])*1e-12,
                  np.array([3., 4.])*1e-12, np.array([5., 6.])*1e-12)
        fractions = np.array([0.2, 0.3, 0.5])
        coefficient = get_ctau_func(widths, fractions)(1.)
        total_u2 = 2e-8
        expected = 1.973269788e-16 / np.dot(fractions*total_u2, [1e-12, 3e-12, 5e-12])
        self.assertAlmostEqual(ctau_from_coupling(
            "HNL", coefficient, total_u2, squared=True)/expected, 1.)

    def test_llp_setter_uses_current_mass_not_stale_coefficient(self):
        from funcs.initLLP import LLP
        llp = LLP.__new__(LLP)
        llp.LLP_name, llp.mass, llp.coupling_metadata = "ALP-photon", 2., None
        llp.get_ctau = lambda m: 8e-12 / m**3
        llp.c_tau_int = 999.
        self.assertEqual(llp.set_coupling(1e-6), 1.)
        llp.set_mass(1.)
        self.assertEqual(llp.set_coupling(1e-12, squared=True), 8.)
        llp.set_mass(None)
        with self.assertRaises(ValueError):
            llp.set_coupling(1e-6)


class CouplingConfigTest(unittest.TestCase):
    def test_shared_and_per_mass_lists_resolve_and_roundtrip(self):
        for inputs in ([1e-6, 2e-6], [[1e-6], [2e-6, 3e-6]]):
            with self.subTest(inputs=inputs):
                config = config_from_mapping(card(couplings=inputs), check_files=False)
                llp = SimpleNamespace(get_ctau=lambda m: m*1e-10)
                resolved = config.resolve_lifetimes(llp)
                for mass, strengths, lifetimes in zip(config.masses,
                        config.couplings_squared, resolved.c_taus):
                    self.assertEqual(lifetimes, tuple(mass*1e-10/s for s in strengths))
                self.assertEqual(config, config_from_mapping(config.as_dict(), check_files=False))
                self.assertNotIn("c_taus", resolved.as_dict())

    def test_lifetime_mode_unchanged(self):
        config = config_from_mapping(card(couplings=None, c_taus=[1, 2]), check_files=False)
        self.assertIs(config.resolve_lifetimes(None), config)
        self.assertEqual(config.c_taus, ((1, 2), (1, 2)))
        self.assertNotIn("couplings_squared", config.as_dict())

    def test_bad_grids_and_conflicting_inputs(self):
        for changes in (dict(c_taus=[1]), dict(couplings_squared=[1e-12]),
                        dict(couplings=None), dict(couplings=[]),
                        dict(couplings=[[1e-6]]), dict(couplings=[True]),
                        dict(couplings=[float("nan")]), dict(couplings=[1e-300])):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                config_from_mapping(card(**changes), check_files=False)

    def test_cli_scan_choice_overrides_card_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"scan.json"
            path.write_text(json.dumps(card(couplings=None, ctaus=[1, 2])))
            with mock.patch("funcs.simulation_config._validate_installed_files"), \
                 mock.patch("funcs.simulation_config.SimulationConfig.selected_decay_indices"):
                config, _ = config_from_command_line([
                    "--card", str(path), "--couplings-squared", "1e-12"])
            self.assertEqual(config.couplings_squared, ((1e-12,), (1e-12,)))

    def test_same_lifetime_grid_reaches_same_generation_loop(self):
        config = config_from_mapping(card(couplings=[[1e-6], [2e-6, 3e-6]]), check_files=False)
        llp = SimpleNamespace(get_ctau=lambda m: m*1e-10,
                             set_mass=mock.Mock(), compute_mass_dependent_properties=mock.Mock())
        runtime = SimpleNamespace(decayProducts=None, output_provenance=SimpleNamespace(
            effective_generator_identity=lambda *a, **k: (None, "raw", None)))
        recorded = []
        def capture(runtime, cfg, llp, channels, mi, ci, n, **kwargs):
            recorded.append((mi, ci, n, cfg.c_taus[mi-1][ci-1]))
        with mock.patch.object(simulate, "_gated_rates", return_value=(None, 1., None)), \
             mock.patch.object(simulate, "run_scan_point", side_effect=capture):
            simulate.run_mass_lifetime_grid(runtime, config, llp, [0])
            coupling_result = recorded.copy()
            recorded.clear()
            resolved = config.resolve_lifetimes(llp)
            lifetime_config = config_from_mapping(card(couplings=None,
                c_taus=resolved.c_taus), check_files=False)
            simulate.run_mass_lifetime_grid(runtime, lifetime_config, llp, [0])
        self.assertEqual(recorded, coupling_result)


class CouplingGridTest(unittest.TestCase):
    def test_lists_and_ranges(self):
        for flag, values, expected in (
            ("couplings", [1e-6, 2e-6], [1e-12, 4e-12]),
            ("couplings_squared", [1e-12, 4e-12], [1e-12, 4e-12]),
            ("coupling_range", [1e-7, 3e-7, 1e-7], [1e-14, 4e-14, 9e-14]),
            ("coupling_squared_range", [1e-14, 3e-14, 1e-14], [1e-14, 2e-14, 3e-14]),
            ("coupling_logrange", [1e-7, 1e-5, 3], [1e-14, 1e-12, 1e-10]),
            ("coupling_squared_logrange", [1e-14, 1e-10, 3], [1e-14, 1e-12, 1e-10]),
        ):
            with self.subTest(flag=flag):
                np.testing.assert_allclose(run_batch.build_couplings(
                    SimpleNamespace(**{flag: values})), expected, atol=0, rtol=1e-14)
        self.assertIsNone(run_batch.build_couplings(SimpleNamespace()))
        for values in ([1e-6, 1e-7, 3], [1e-7, 1e-6, 2.5], [0, 1, 3]):
            with self.assertRaises(ValueError):
                run_batch.build_couplings(SimpleNamespace(coupling_logrange=values))

    def test_batch_cli_rejects_mixed_parameterizations(self):
        args = ["run_batch.py", "--llp", "ALP-photon", "--masses", "0.1",
                "--nevents", "10", "--ctaus", "1", "--couplings", "1e-6"]
        with mock.patch.object(sys, "argv", args), self.assertRaises(SystemExit):
            run_batch.parse_args()

    def test_mass_dependent_edges_and_bias(self):
        lower, upper = [(1, 1e-8), (10, 1e-6)], [(1, 1e-4), (10, 1e-2)]
        uniform = coupling_grid(math.sqrt(10), lower, upper, n=3, alpha=1)
        np.testing.assert_allclose(uniform, [1e-7, 1e-5, 1e-3], atol=0)
        self.assertGreater(coupling_grid(math.sqrt(10), lower, upper, n=3)[1], uniform[1])
        for kwargs in (dict(mass=0.1), dict(alpha=0), dict(n=2.5), dict(lower=upper, upper=lower)):
            arguments = dict(mass=2., lower=lower, upper=upper)
            arguments.update(kwargs)
            with self.assertRaises(ValueError):
                coupling_grid(**arguments)
        config = config_from_mapping(example_card())
        self.assertEqual(len(config.masses), 20)
        self.assertEqual(sum(map(len, config.couplings_squared)), 300)


if __name__ == "__main__":
    unittest.main()
