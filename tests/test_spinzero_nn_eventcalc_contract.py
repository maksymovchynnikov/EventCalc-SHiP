#!/usr/bin/env python3
"""Non-MC checks for the spin-zero NN/EventCalc ownership boundary."""

import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DIST = os.path.join(ROOT, "Distributions")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from funcs import exhadDecays  # noqa: E402
from funcs.initLLP import (  # noqa: E402
    DEFAULT_SCALAR_PRESCRIPTION,
    LLP,
    _load_scalar_ctau_json,
    _validate_scalar_branching_vector,
)

SCALAR_DECAY_CARDS = (
    "Scalar-mixing/BrRatio-Scalar-1809.01876.json",
    "Scalar-mixing/BrRatio-Scalar-2407.13587-Central.json",
    "Scalar-mixing/BrRatio-Scalar-2407.13587-Lower.json",
    "Scalar-mixing/BrRatio-Scalar-2407.13587-Upper.json",
)
SCALAR_CTAU_TABLES = (
    "Scalar-mixing/ctau-Scalar-1809.01876.txt",
    "Scalar-mixing/ctau-Scalar-2407.13587-Central.txt",
    "Scalar-mixing/ctau-Scalar-2407.13587-Lower.txt",
    "Scalar-mixing/ctau-Scalar-2407.13587-Upper.txt",
)

EXACT_NN = {(-2212, 2212), (-2112, 2112)}


class SpinZeroNNEventCalcContractTest(unittest.TestCase):
    def test_omitted_scalar_prescription_uses_shared_central_card(self):
        selection = {
            "LLP_name": "Scalar-quartic",
            "particle_path": os.path.join(DIST, "Scalar-quartic"),
        }
        scalar = LLP(None, selection)
        self.assertEqual(
            scalar.scalar_lifetime, DEFAULT_SCALAR_PRESCRIPTION)
        self.assertIn("KLKL", scalar.decayChannels)
        self.assertIn("KSKS", scalar.decayChannels)
        self.assertIn("ppbar", scalar.decayChannels)
        self.assertIn("nnbar", scalar.decayChannels)
        self.assertNotIn("KLKS", scalar.decayChannels)

    def test_unphysical_scalar_probabilities_fail_closed(self):
        channels = np.asarray(["physical", "bad"])
        with self.assertRaisesRegex(ValueError, "negative probabilities"):
            _validate_scalar_branching_vector(
                0.99, "test", channels, [0.5, -0.1])
        with self.assertRaisesRegex(ValueError, "exceed unity"):
            _validate_scalar_branching_vector(
                0.28, "test", channels, [0.8, 0.3])

    def test_alp_owns_exact_nn_as_explicit_persistent_rows(self):
        relative_path = "ALP-fermion/ALP-fermion-decay.json"
        with open(os.path.join(DIST, relative_path)) as stream:
            rows = json.load(stream)
        exact = {
            tuple(sorted(int(value) for value in row[1])): row[0]
            for row in rows
            if tuple(sorted(int(value) for value in row[1])) in EXACT_NN
        }
        self.assertEqual(exact, {
            (-2212, 2212): "ppbar",
            (-2112, 2112): "nnbar",
        })

    def test_scalar_owns_exact_nn_as_explicit_persistent_rows(self):
        for relative_path in SCALAR_DECAY_CARDS:
            with self.subTest(card=relative_path):
                with open(os.path.join(DIST, relative_path)) as stream:
                    rows = json.load(stream)
                exact = {
                    tuple(sorted(
                        int(value) for value in row[1]
                        if int(value) != -999)): row[0]
                    for row in rows
                    if tuple(sorted(
                        int(value) for value in row[1]
                        if int(value) != -999)) in EXACT_NN
                }
                self.assertEqual(exact, {
                    (-2212, 2212): "ppbar",
                    (-2112, 2112): "nnbar",
                })

    def test_scalar_neutral_kaons_are_c_even_identical_pairs(self):
        expected = {
            (130, 130): "KLKL",
            (310, 310): "KSKS",
        }
        for relative_path in SCALAR_DECAY_CARDS:
            with self.subTest(card=relative_path):
                with open(os.path.join(DIST, relative_path)) as stream:
                    rows = json.load(stream)
                neutral_kaons = {
                    tuple(sorted(
                        int(value) for value in row[1]
                        if int(value) != -999)): row[0]
                    for row in rows
                    if all(
                        abs(int(value)) in (130, 310, 999)
                        for value in row[1])
                    and any(int(value) != -999 for value in row[1])
                }
                self.assertEqual(neutral_kaons, expected)

    def test_scalar_lifetime_tables_use_valid_json_pair_grids(self):
        for relative_path in SCALAR_CTAU_TABLES:
            with self.subTest(table=relative_path):
                table = _load_scalar_ctau_json(
                    os.path.join(DIST, relative_path))
                self.assertEqual(tuple(table.columns), (0, 1))
                self.assertGreaterEqual(len(table), 2)
                self.assertTrue((table.iloc[:, 0].diff().dropna() > 0).all())
                self.assertTrue((table.iloc[:, 1] > 0).all())

    def test_scalar_lifetime_parser_rejects_invalid_json_grids(self):
        invalid_tables = (
            [],
            [[1.0, 1.0]],
            [[1.0, 1.0], [1.0, 0.5]],
            [[1.0, 1.0], [2.0, 0.0]],
            [[1.0, 1.0], ["2.0", 0.5]],
            [[1.0, 1.0, 2.0], [2.0, 0.5]],
        )
        for payload in invalid_tables:
            with self.subTest(payload=payload):
                with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".txt") as stream:
                    json.dump(payload, stream)
                    stream.flush()
                    with self.assertRaisesRegex(
                            ValueError, "Scalar lifetime"):
                        _load_scalar_ctau_json(stream.name)

    def test_spinzero_cards_still_delegate_only_an_inclusive_benchmark(self):
        expected = {
            "ALP-fermion/exhad.json": "alp",
            "Scalar-mixing/exhad.json": "hls",
            "Scalar-quartic/exhad.json": "hls",
        }
        for relative_path, benchmark in expected.items():
            with self.subTest(card=relative_path):
                with open(os.path.join(DIST, relative_path)) as stream:
                    card = json.load(stream)
                self.assertEqual(card.get("bench"), benchmark)
                # EventCalc cards must not grow a second BR/width payload.
                self.assertNotIn("R_pp", card)
                self.assertNotIn("R_nn", card)
                self.assertNotIn("nn_constraint", card)

    def test_scalar_table_prescriptions_resolve_matching_benchmarks(self):
        aliases = {
            "1809.01876": "hls-1809",
            "2407.13587-Central": "hls",
            "2407.13587-Lower": "hls-lower",
            "2407.13587-Upper": "hls-upper",
        }
        for folder in ("Scalar-mixing", "Scalar-quartic"):
            path = os.path.join(DIST, folder)
            with open(os.path.join(path, "exhad.json")) as stream:
                card = json.load(stream)
            self.assertEqual(card.get("bench_variants"), aliases)
            for variant, expected in aliases.items():
                with self.subTest(folder=folder, variant=variant), \
                     mock.patch.dict(os.environ, {}, clear=True):
                    exhadDecays.set_force(None)
                    exhadDecays.set_selection(folder, path, variant)
                    self.assertEqual(exhadDecays.get_bench(), expected)

    def test_unknown_scalar_table_prescription_fails_closed(self):
        path = os.path.join(DIST, "Scalar-mixing")
        with mock.patch.dict(os.environ, {}, clear=True):
            exhadDecays.set_force(None)
            exhadDecays.set_selection(
                "Scalar-mixing", path, "future-unmapped-table")
            with self.assertRaisesRegex(RuntimeError, "no benchmark"):
                exhadDecays.get_bench()

    def tearDown(self):
        exhadDecays.set_force(None)
        exhadDecays.set_selection(None, None)


if __name__ == "__main__":
    unittest.main()
