#!/usr/bin/env python3
"""Fail-closed EventCalc integration tests for portable Model 1."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


EVENTCALC = Path(__file__).resolve().parents[1]
if str(EVENTCALC) not in sys.path:
    sys.path.insert(0, str(EVENTCALC))

from funcs import decayProducts, exhadDecays, output_provenance  # noqa: E402

# This module was written against an exHad adapter that carried its own lazily
# imported bridge object.  The adapter of this repository, funcs/exhadDecays.py,
# talks to a configured exHad release instead and defines none of the names
# below, so the module cannot run as written.  The check is by name, so the
# module runs again by itself if the adapter ever defines them.
_ADAPTER_NAMES = ("_bridge",
                  "set_portable_model1_deployment",
                  "portable_model1_generator_provenance")
_ABSENT = [name for name in _ADAPTER_NAMES if not hasattr(exhadDecays, name)]
if _ABSENT:
    raise unittest.SkipTest(
        "funcs/exhadDecays.py does not define " + ", ".join(_ABSENT)
        + "; these tests were written against an adapter that did"
    )



class _FakePortableBridge:
    def __init__(self, card: Path, *, portal="fermion-alp"):
        self.card = card.resolve()
        self.portal = portal
        self.calls = []
        self.signatures = {
            tuple(sorted((21, 21))): "active-gg",
            tuple(sorted((-211, 211))): "external-pipi",
        }

    def portable_model1_eventcalc_deployment_metadata(self, card):
        if Path(card).resolve() != self.card:
            raise RuntimeError("wrong deployment")
        return {
            "deployment_id": "portable-model1-test-deployment-v1",
            "deployment_card_path": str(self.card),
            "deployment_card_sha256": "b" * 64,
            "model_id": "portable-model1-fermion-alp-v1",
            "portal_id": self.portal,
            "benchmark_id": "bc10-fermion-universal-alp",
            "variant_id": "central-1911mev-split",
            "supported_variations": (
                "central", "saturation-off", "p-low", "p-high"),
            "variation_semantics": {
                value: {"effect": "active", "no_effect_reason": None}
                for value in (
                    "central", "saturation-off", "p-low", "p-high")
            },
            "activation_support_gev": (1.911, 5.0),
            "contract_path": "/sealed/fermion-alp.json",
            "contract_sha256": "a" * 64,
            "probability_card_path": "/sealed/fermion-alp-runtime.json",
            "probability_card_sha256": "c" * 64,
            "active_row_id": "model1-active",
            "eventcalc_row_signatures": tuple(
                {"authority_row_id": row, "pdg_signature": signature}
                for signature, row in self.signatures.items()),
            "complete_stable_event_guarantee": True,
            "fixed_mass_complete_pool": True,
        }

    def portable_model1_eventcalc_row_id(self, card, pdgs):
        return self.signatures.get(tuple(sorted(
            int(value) for value in pdgs if int(value) != -999)))

    def portable_model1_eventcalc_authority_probabilities(
            self, card, mass, variation):
        return {"active-gg": 0.6, "external-pipi": 0.4}

    def sample_portable_model1_for_eventcalc(
            self, card, mass, count, seed=1, variation="central"):
        self.calls.append((float(mass), int(count), int(seed), variation))
        # A complete stable six-field event.  Its particular topology is
        # irrelevant to the host-pooling test.
        return [[0.0, 0.0, 0.0, float(mass), 0.0, 22.0]
                for _ in range(int(count))]


class PortableModel1EventCalcIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {}, clear=False)
        self.environment.start()
        for name in (
                "EXHAD_PORTABLE_MODEL1_DEPLOYMENT_CARD",
                "EXHAD_PORTABLE_MODEL1_VARIATION", "EXHAD_BENCH"):
            os.environ.pop(name, None)
        exhadDecays._bridge = None
        exhadDecays._FORCE = None
        exhadDecays.set_selection(None, None, None)
        exhadDecays.set_portable_model1_deployment(None, None)

    def tearDown(self):
        exhadDecays._bridge = None
        exhadDecays._FORCE = None
        exhadDecays.set_selection(None, None, None)
        exhadDecays.set_portable_model1_deployment(None, None)
        self.environment.stop()

    def _selection(self, root: Path, *, variation="p-high", portal=None):
        card = root / "deployment.json"
        card.write_text("{}", encoding="utf-8")
        bridge = _FakePortableBridge(
            card, portal="fermion-alp" if portal is None else portal)
        exhadDecays._bridge = bridge
        exhadDecays.set_portable_model1_deployment(card, variation)
        exhadDecays.set_selection("ALP-fermion", None, None)
        return bridge

    def test_missing_explicit_deployment_fails_closed(self):
        exhadDecays.set_portable_model1_deployment(
            "/definitely/absent/portable-model1.json", "central")
        exhadDecays.set_selection("ALP-fermion", None, None)
        with self.assertRaisesRegex(RuntimeError, "deployment card is missing"):
            exhadDecays.get_bench()

    def test_above_deployment_support_does_not_fall_back_to_raw(self):
        with tempfile.TemporaryDirectory() as directory:
            self._selection(Path(directory), variation='central')
            pdgs=np.array([[21,21,-999],[-211,211,-999]],dtype=int)
            with self.assertRaisesRegex(ValueError,'exceeds.*support'):
                decayProducts._fixed_exhad_context(5.001,pdgs,[0,1],'ALP-fermion')
            self.assertEqual(decayProducts._fixed_exhad_context(
                1.8,pdgs,[0,1],'ALP-fermion'),(None,[]))

    def test_complete_hadronic_pool_is_drawn_once_and_direct_row_survives(self):
        with tempfile.TemporaryDirectory() as directory:
            bridge = self._selection(Path(directory))
            pdgs = np.array([
                [21, 21, -999],
                [-211, 211, -999],
                [-11, 11, -999],
            ], dtype=int)
            ratios = np.array([0.3, 0.2, 0.5])
            selected = [0, 1, 2]
            exhad, rows = decayProducts._fixed_exhad_context(
                2.5, pdgs, selected, "ALP-fermion", None, None)
            self.assertIs(exhad, exhadDecays)
            self.assertEqual(rows, [0, 1])

            pool = decayProducts.prepare_fixed_exhad_pool(
                2.5, 10, pdgs, selected, BrRatio=ratios,
                br_visible_val=1.0, llp_name="ALP-fermion", seed=71)
            self.assertEqual(len(pool), 5)
            self.assertEqual(len(bridge.calls), 1)
            self.assertEqual(bridge.calls[0][1], 5)
            self.assertEqual(bridge.calls[0][3], "p-high")

            events, sizes, labels = decayProducts.simulateDecays_rest_frame(
                2.5, pdgs, ratios, 10, None, selected, 1.0,
                llp_name="ALP-fermion", seed=71,
                fixed_exhad_pool=pool, return_process_labels=True)
            self.assertEqual(len(events), 10)
            self.assertEqual(list(sizes), [3, 2, 5])
            self.assertEqual(labels[:2], ["Jets-matched", "Jets-matched"])
            self.assertIsNone(labels[2])
            self.assertEqual(len(bridge.calls), 1)

    def test_partial_pool_and_rate_mismatch_both_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            self._selection(Path(directory), variation="central")
            pdgs = np.array([
                [21, 21, -999],
                [-211, 211, -999],
                [-11, 11, -999],
            ], dtype=int)
            with self.assertRaisesRegex(ValueError, "complete hadronic row"):
                decayProducts._fixed_exhad_context(
                    2.5, pdgs, [0, 2], "ALP-fermion", None, None)
            with self.assertRaisesRegex(RuntimeError, "branching ratios differ"):
                decayProducts.prepare_fixed_exhad_pool(
                    2.5, 10, pdgs, [0, 1, 2],
                    BrRatio=np.array([0.2, 0.3, 0.5]),
                    br_visible_val=1.0, llp_name="ALP-fermion", seed=1)

    def test_variation_and_frozen_identity_enter_output_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            self._selection(Path(directory), variation="saturation-off")
            provenance = exhadDecays.portable_model1_generator_provenance(
                "portable1")
            validated = output_provenance.validate_generator_provenance(
                provenance)
        self.assertEqual(validated["variation_id"], "saturation-off")
        self.assertEqual(validated["contract_sha256"], "a" * 64)
        self.assertEqual(
            validated["route_ownership"]["outer_owner"],
            "full-portal-hadronic-row-pool")

    def test_hnl_cannot_enter_fixed_mass_portable_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            card = Path(directory) / "deployment.json"
            card.write_text("{}", encoding="utf-8")
            exhadDecays._bridge = _FakePortableBridge(
                card, portal="hnl-cc-ud")
            exhadDecays.set_portable_model1_deployment(card, "central")
            exhadDecays.set_selection("HNL", None, None)
            with self.assertRaisesRegex(RuntimeError, "exact-W"):
                exhadDecays.portable_model1_configuration()


if __name__ == "__main__":
    unittest.main()
