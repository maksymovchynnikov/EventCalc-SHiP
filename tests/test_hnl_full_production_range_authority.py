#!/usr/bin/env python3
"""Contracts for the complete installed HNL production range and exact W."""

import json
import math
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HNL_DIR = ROOT / "Distributions" / "HNL"
AUTHORITY_PATH = HNL_DIR / "exhad-full-range-authority.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from funcs import exhadDecays  # noqa: E402


def _load_authority():
    with AUTHORITY_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


def _mass_endpoints(path):
    first = None
    last = None
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            try:
                mass = float(fields[0])
            except ValueError as exc:
                raise AssertionError(
                    "%s:%d has a non-numeric parent mass" %
                    (path, line_number)
                ) from exc
            if not math.isfinite(mass):
                raise AssertionError(
                    "%s:%d has a non-finite parent mass" %
                    (path, line_number)
                )
            if first is None:
                first = mass
            if last is not None and mass < last:
                raise AssertionError(
                    "%s parent masses are not nondecreasing" % path
                )
            last = mass
    if first is None or last is None:
        raise AssertionError("%s is empty" % path)
    return first, last


def _particle(px, py, pz, energy, mass, pdg):
    return [px, py, pz, energy, mass, float(pdg), 0.0, 1.0]


def _current_event(pdg_row, hadronic_mass, lepton_energy):
    active = [int(pdg) for pdg in pdg_row if int(pdg) != -999]
    event = []
    event += _particle(
        0.0, 0.0, 0.0, hadronic_mass / 2.0, 0.0, active[0])
    event += _particle(
        0.0, 0.0, 0.0, hadronic_mass / 2.0, 0.0, active[1])
    event += _particle(
        0.0, 0.0, 0.0, lepton_energy, 0.0, active[2])
    return np.asarray([event], dtype=np.float64)


class HNLFullProductionRangeAuthorityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.authority = _load_authority()
        cls.scope = cls.authority["scope"]

    def test_authority_declares_the_installed_closed_production_domain(self):
        self.assertEqual(
            self.authority["schema"],
            "exhad-hnl-full-production-range-v1",
        )
        self.assertEqual(
            self.authority["integration_state"],
            "production-event-provider-connected",
        )
        self.assertEqual(
            self.scope["production_parent_mass_domain_GeV"],
            [0.02, 5.27],
        )
        self.assertEqual(
            self.scope["decay_rate_source_domain_GeV"],
            [0.0002, 40.0],
        )

    def test_all_installed_production_inputs_share_the_declared_endpoints(self):
        expected = tuple(
            self.scope["production_parent_mass_domain_GeV"])
        paths = []
        for item in self.authority["production_inputs"]:
            paths.append(HNL_DIR / item["double_distribution"])
            paths.append(HNL_DIR / item["total_yield"])
        paths.append(HNL_DIR / self.authority["kinematic_endpoint_input"])

        self.assertEqual(len(paths), 7)
        self.assertEqual(len(set(paths)), 7)
        for path in paths:
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file())
                observed = _mass_endpoints(path)
                self.assertTrue(math.isclose(
                    observed[0], expected[0],
                    rel_tol=0.0, abs_tol=1.0e-14))
                self.assertTrue(math.isclose(
                    observed[1], expected[1],
                    rel_tol=0.0, abs_tol=1.0e-12))

    def test_exact_W_is_the_only_fragmentation_composition_coordinate(self):
        ownership = self.authority["exact_W_ownership"]
        self.assertEqual(
            self.scope["composition_coordinate"],
            "hadronic_invariant_mass_W_GeV",
        )
        self.assertTrue(
            self.scope["parent_mass_is_not_a_composition_coordinate"])
        self.assertEqual(ownership["evaluation"], "per-event-exact")
        self.assertEqual(
            ownership["forbidden_composition_coordinates"],
            ["parent_mass_GeV"],
        )
        self.assertEqual(
            ownership["replacement_scope"],
            "replace only the two-parton hadronic subsystem",
        )
        preserved = set(ownership["preserved_quantities"])
        self.assertIn("sampled hadronic invariant mass W", preserved)
        self.assertIn("HNL lifetime", preserved)
        self.assertIn("EventCalc-selected channel count", preserved)

    def test_existing_router_uses_exact_W_above_the_old_parent_cap(self):
        rows = {
            "CC_ud": [2, -1, 11, -999],
            "CC_us": [2, -3, 13, -999],
            "CC_cd": [4, -1, 11, -999],
            "CC_cs": [4, -3, 13, -999],
            "NC_ud": [2, -2, 12, -999],
            "NC_s": [3, -3, 14, -999],
        }
        exact_W = 2.5
        for current, pdgs in rows.items():
            event = _current_event(pdgs, exact_W, lepton_energy=0.4)
            with self.subTest(current=current):
                generator = mock.Mock()
                generator.hadronize_hnl.return_value = [[[0., 0., 0., 1., 1., 211.]]]
                with mock.patch.object(exhadDecays, "_model_generator",
                                       return_value=generator):
                    exhadDecays.process_hnl_with_exhad(event, pdgs, mass=5.27, seed=9)
                call = generator.hadronize_hnl.call_args
                self.assertEqual(call.args[0], 5.27)
                self.assertEqual(call.args[1], pdgs[:3])
                np.testing.assert_array_equal(call.args[2], event)
                pair = np.asarray(call.args[2]).reshape(-1, 8)[:2, :4].sum(axis=0)
                observed_W = np.sqrt(pair[3] ** 2 - np.sum(pair[:3] ** 2))
                self.assertAlmostEqual(observed_W, exact_W, places=13)
                self.assertEqual(call.kwargs, {"seed": 9})


if __name__ == "__main__":
    unittest.main()
