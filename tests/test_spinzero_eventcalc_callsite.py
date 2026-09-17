#!/usr/bin/env python3
"""Call-site contracts for the matched spin-zero EventCalc adapter.

These tests deliberately mock the expensive generator.  The production
spin-zero measure is tested in ``exhad/py/test_spinzero_eventcalc_adapter.py``;
this file checks that EventCalc pools the decay-table rows exactly once,
preserves their aggregate branching fraction, and never sends an owned row
to raw Pythia inside the matched support.
"""

import contextlib
import io
import json
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DIST = os.path.join(ROOT, "Distributions")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Keep this routing contract independent of EventCalc's optional numba
# installation.  The decay kernels are replaced by deterministic mocks below.
for module_name in ("TwoBodyDecay", "ThreeBodyDecay", "FourBodyDecay"):
    module = types.ModuleType("funcs." + module_name)
    module.decay_products = lambda *_args, **_kwargs: None
    sys.modules.setdefault("funcs." + module_name, module)

from funcs import decayProducts, exhadDecays  # noqa: E402

# This module was written against an exHad adapter that carried its own lazily
# imported bridge object.  The adapter of this repository, funcs/exhadDecays.py,
# talks to a configured exHad release instead and defines none of the names
# below, so the module cannot run as written.  The check is by name, so the
# module runs again by itself if the adapter ever defines them.
_ADAPTER_NAMES = ("_bridge",
                  "_load_bridge",
                  "MATCHED_SPINZERO_BACKEND")
_ABSENT = [name for name in _ADAPTER_NAMES if not hasattr(exhadDecays, name)]
if _ABSENT:
    raise unittest.SkipTest(
        "funcs/exhadDecays.py does not define " + ", ".join(_ABSENT)
        + "; these tests were written against an adapter that did"
    )



BACKEND = "matched-spinzero-v1"
ALP_PORTAL = "ALP-fermion"
ALP_VARIANT = "bc10-fermion-universal-alp"
SCALAR_PORTAL = "Scalar-mixing"

ALP_POOLED = (
    "Jets-GG", "Jets-ss", "Jets-cc", "ppbar", "nnbar",
)
ALP_NONHADRONIC = ("ePeM", "muPmuM", "tauPtauM", "2gamma")
SCALAR_POOLED = (
    "PipPim", "2Pi0", "2Kch", "KLKL", "KSKS",
    "ppbar", "nnbar",
    "2Pip2Pim", "PipPim2Pi0",
    "Jets-GG", "Jets-ss", "Jets-cc", "Jets-bb",
)
SCALAR_NONHADRONIC = ("ePeM", "muPmuM", "tauPtauM")

SCALAR_VARIANTS = {
    "2407.13587-Central": ("hls", "2407.13587-Central"),
    "2407.13587-Lower": ("hls-lower", "2407.13587-Lower"),
    "2407.13587-Upper": ("hls-upper", "2407.13587-Upper"),
    "1809.01876": ("hls-1809", "1809.01876"),
}


def _load_decay_rows(relative_path):
    with open(os.path.join(DIST, relative_path)) as stream:
        records = json.load(stream)
    labels = tuple(str(row[0]) for row in records)
    active = tuple(tuple(int(value) for value in row[1]) for row in records)
    width = max(len(row) for row in active)
    pdgs = np.asarray(
        [row + (-999,) * (width - len(row)) for row in active],
        dtype=np.int64,
    )
    return labels, pdgs


def _ratios(labels, pooled, nonhadronic):
    """Give the pool and nonhadronic rows exact, integer event allocations."""
    pooled_weight = 0.60 / len(pooled)
    nonhadronic_weight = 0.40 / len(nonhadronic)
    return [
        pooled_weight if label in pooled
        else nonhadronic_weight if label in nonhadronic
        else 0.0
        for label in labels
    ]


def _unprocessed_events(size, pdgs, mass):
    result = np.zeros((int(size), 8 * len(pdgs)), dtype=np.float64)
    energy = float(mass) / len(pdgs)
    for index, pdg in enumerate(pdgs):
        offset = 8 * index
        result[:, offset + 3] = energy
        result[:, offset + 4] = 0.0
        result[:, offset + 5] = float(pdg)
        result[:, offset + 7] = 1.0
    return result


def _matched_events(count, mass):
    return [
        [0.0, 0.0, 0.0, float(mass), float(mass), 22.0]
        for _ in range(int(count))
    ]


def _ownership(portal, variant, mass, pooled, nonhadronic, branching):
    return SimpleNamespace(
        portal=portal,
        variant=variant,
        mass_gev=float(mass),
        pooled_eventcalc_rows=tuple(pooled),
        nonhadronic_rows=tuple(nonhadronic),
        aggregate_branching_fraction=float(branching),
        replacement_mode="replace-pooled-rows-once",
        call_once_for_aggregate_hadronic_decay=True,
        permits_per_constituent_row_calls=False,
    )


class SpinZeroEventCalcCallSiteTest(unittest.TestCase):
    def tearDown(self):
        exhadDecays.set_force(None)
        exhadDecays.set_selection(None, None)
        exhadDecays._bridge = None
        exhadDecays._card_config.cache_clear()

    def _bridge(self, ownership):
        bridge = mock.Mock()
        bridge.spinzero_eventcalc_row_ownership.return_value = ownership
        bridge.sample_spinzero_for_eventcalc.side_effect = (
            lambda _portal, _variant, mass, count, seed=1:
            _matched_events(count, mass)
        )
        bridge.sample_for_eventcalc.side_effect = AssertionError(
            "matched spin-zero calls must not use the legacy sampler")
        return bridge

    def _assert_one_pooled_batch(
            self, *, llp_name, particle_path, variant, mass, decay_table,
            portal, source_variant, pooled, nonhadronic):
        labels, pdgs = _load_decay_rows(decay_table)
        ratios = _ratios(labels, pooled, nonhadronic)
        selected = list(range(len(labels)))
        # The channel allocation is a multinomial draw from NumPy's global
        # stream, which a driver seeds once per scan point.  Fixing the stream
        # here puts the expected counts and the counts the simulator draws on
        # one sample.
        allocation_seed = 8191
        np.random.seed(allocation_seed)
        expected_sizes = decayProducts.distribute_events(100, ratios)
        pooled_positions = tuple(
            index for index, label in enumerate(labels) if label in pooled)
        nonhadronic_positions = tuple(
            index for index, label in enumerate(labels)
            if label in nonhadronic)
        pooled_branching = sum(ratios[index] for index in pooled_positions)
        # The pool is prepared once for this mass and reused by every lifetime
        # at it, so it holds the largest share the pooled rows can draw rather
        # than the share this one draw gave them.
        expected_pool_size = decayProducts.group_allocation_ceiling(
            100, pooled_branching / sum(ratios))
        ownership = _ownership(
            portal, source_variant, mass, pooled, nonhadronic,
            pooled_branching)
        bridge = self._bridge(ownership)

        raw_signatures = []

        def fake_two_body(
                parent_mass, size, _m1, _m2, pdg1, pdg2, *_args):
            return _unprocessed_events(
                size, (int(pdg1), int(pdg2)), parent_mass)

        def fake_three_body(parent_mass, size, params):
            return _unprocessed_events(
                size, tuple(int(value) for value in params[:3]),
                parent_mass)

        def fake_four_body(parent_mass, params, size):
            return _unprocessed_events(
                size, tuple(int(value) for value in params[:4]),
                parent_mass)

        def fake_raw(events, parent_mass, seed=1):
            del parent_mass, seed
            events = np.asarray(events, dtype=np.float64)
            particles = events.reshape(len(events), -1, 8)
            signature = tuple(
                int(value) for value in particles[0, :, 5]
                if int(value) != -999)
            raw_signatures.append(signature)
            return np.asarray(
                _matched_events(len(events), mass), dtype=np.float64)

        with mock.patch.dict(os.environ, {}, clear=True), (
            mock.patch.object(exhadDecays, "_bridge", bridge)
        ), mock.patch.object(
            exhadDecays, "_load_bridge", return_value=bridge
        ), mock.patch.object(
            decayProducts.TwoBodyDecay, "decay_products",
            side_effect=fake_two_body,
        ), mock.patch.object(
            decayProducts.ThreeBodyDecay, "decay_products",
            side_effect=fake_three_body,
        ), mock.patch.object(
            decayProducts.FourBodyDecay, "decay_products",
            side_effect=fake_four_body,
        ), mock.patch.object(
            decayProducts, "process_events_with_pythia",
            side_effect=fake_raw,
        ), contextlib.redirect_stdout(io.StringIO()):
            pool = decayProducts.prepare_fixed_exhad_pool(
                mass,
                100,
                pdgs,
                selected,
                BrRatio=ratios,
                br_visible_val=1.0,
                llp_name=llp_name,
                particle_path=particle_path,
                exhad_variant=variant,
                seed=73,
                batch_index=4,
            )
            np.random.seed(allocation_seed)
            result, sizes, process_labels = (
                decayProducts.simulateDecays_rest_frame(
                    mass,
                    pdgs,
                    ratios,
                    100,
                    None,
                    selected,
                    1.0,
                    llp_name=llp_name,
                    particle_path=particle_path,
                    exhad_variant=variant,
                    seed=73,
                    batch_index=5,
                    fixed_exhad_pool=pool,
                    return_process_labels=True,
                )
            )

        self.assertEqual(len(pool), expected_pool_size)
        self.assertEqual(len(result), 100)
        np.testing.assert_array_equal(sizes, expected_sizes)

        # The EventCalc table's original pooled BR is retained once.  It is
        # neither renormalized row-by-row nor replaced by a constituent BR.
        self.assertAlmostEqual(
            ownership.aggregate_branching_fraction,
            sum(ratios[index] for index in pooled_positions),
            places=15,
        )
        bridge.spinzero_eventcalc_row_ownership.assert_called_once()
        ownership_call = bridge.spinzero_eventcalc_row_ownership.call_args
        self.assertEqual(
            ownership_call.args[:3],
            (portal, source_variant, mass),
        )
        bridge.sample_spinzero_for_eventcalc.assert_called_once()
        call = bridge.sample_spinzero_for_eventcalc.call_args
        self.assertEqual(
            call.args[:4],
            (portal, source_variant, mass, expected_pool_size),
        )
        self.assertIn("seed", call.kwargs)
        bridge.sample_for_eventcalc.assert_not_called()

        # Every owned row is served from the one pool.  The only permitted
        # raw-Pythia call is the unchanged nonhadronic tau row.
        self.assertEqual(raw_signatures, [(-15, 15)])
        self.assertEqual(
            process_labels,
            [
                "Jets-matched" if label in pooled else None
                for label in labels
            ],
        )
        for position in nonhadronic_positions:
            self.assertIsNone(process_labels[position])

    def test_alp_pools_five_rows_once_and_keeps_nonhadronic_rows(self):
        self._assert_one_pooled_batch(
            llp_name="ALP-fermion",
            particle_path=os.path.join(DIST, "ALP-fermion"),
            variant=None,
            mass=5.0,
            decay_table="ALP-fermion/ALP-fermion-decay.json",
            portal=ALP_PORTAL,
            source_variant=ALP_VARIANT,
            pooled=ALP_POOLED,
            nonhadronic=ALP_NONHADRONIC,
        )

    def test_scalar_pools_all_thirteen_hadronic_rows_once(self):
        self._assert_one_pooled_batch(
            llp_name="Scalar-mixing",
            particle_path=os.path.join(DIST, "Scalar-mixing"),
            variant="2407.13587-Central",
            mass=5.0,
            decay_table=(
                "Scalar-mixing/"
                "BrRatio-Scalar-2407.13587-Central.json"
            ),
            portal=SCALAR_PORTAL,
            source_variant="2407.13587-Central",
            pooled=SCALAR_POOLED,
            nonhadronic=SCALAR_NONHADRONIC,
        )

    def test_mismatched_aggregate_branching_fraction_fails_closed(self):
        labels, pdgs = _load_decay_rows(
            "Scalar-mixing/BrRatio-Scalar-2407.13587-Central.json")
        ratios = _ratios(labels, SCALAR_POOLED, SCALAR_NONHADRONIC)
        selected = list(range(len(labels)))
        pooled_sum = sum(
            ratio for label, ratio in zip(labels, ratios)
            if label in SCALAR_POOLED)
        ownership = _ownership(
            SCALAR_PORTAL,
            "2407.13587-Central",
            3.0,
            SCALAR_POOLED,
            SCALAR_NONHADRONIC,
            pooled_sum + 0.01,
        )
        bridge = self._bridge(ownership)
        with mock.patch.dict(os.environ, {}, clear=True), (
            mock.patch.object(exhadDecays, "_bridge", bridge)
        ), mock.patch.object(
            exhadDecays, "_load_bridge", return_value=bridge
        ), self.assertRaisesRegex(
            (RuntimeError, ValueError),
            r"(?i)(aggregate|branching|pooled)",
        ):
            decayProducts.prepare_fixed_exhad_pool(
                3.0,
                100,
                pdgs,
                selected,
                BrRatio=ratios,
                br_visible_val=1.0,
                llp_name="Scalar-mixing",
                particle_path=os.path.join(DIST, "Scalar-mixing"),
                exhad_variant="2407.13587-Central",
                seed=11,
            )
        bridge.spinzero_eventcalc_row_ownership.assert_called_once()
        bridge.sample_spinzero_for_eventcalc.assert_not_called()

    def test_backend_cards_and_closed_support_boundaries(self):
        self.assertEqual(exhadDecays.MATCHED_SPINZERO_BACKEND, BACKEND)
        cases = [
            (
                "ALP-fermion",
                os.path.join(DIST, "ALP-fermion"),
                None,
                1.911,
            ),
        ]
        for folder in ("Scalar-mixing", "Scalar-quartic"):
            path = os.path.join(DIST, folder)
            cases.extend(
                (folder, path, variant, 2.0)
                for variant in SCALAR_VARIANTS
            )

        bridge = mock.Mock()
        with mock.patch.dict(os.environ, {}, clear=True), (
            mock.patch.object(exhadDecays, "_load_bridge",
                              return_value=bridge)
        ):
            for llp_name, path, variant, lower in cases:
                with self.subTest(llp=llp_name, variant=variant):
                    with open(os.path.join(path, "exhad.json")) as stream:
                        card = json.load(stream)
                    self.assertEqual(card.get("backend"), BACKEND)
                    exhadDecays.set_selection(llp_name, path, variant)
                    self.assertFalse(exhadDecays.enabled(lower - 1.0e-6))
                    self.assertTrue(exhadDecays.enabled(lower))
                    self.assertTrue(exhadDecays.enabled(5.0))
                    self.assertFalse(exhadDecays.enabled(5.0 + 1.0e-6))

    def test_every_scalar_variant_and_alp_use_the_spinzero_sampler(self):
        cases = [
            (
                "ALP-fermion",
                os.path.join(DIST, "ALP-fermion"),
                None,
                "alp",
                ALP_PORTAL,
                ALP_VARIANT,
                1.911,
            ),
        ]
        for folder in ("Scalar-mixing", "Scalar-quartic"):
            path = os.path.join(DIST, folder)
            cases.extend(
                (
                    folder,
                    path,
                    table_variant,
                    bench,
                    SCALAR_PORTAL,
                    source_variant,
                    2.0,
                )
                for table_variant, (bench, source_variant)
                in SCALAR_VARIANTS.items()
            )

        for (
                llp_name, path, table_variant, expected_bench,
                portal, source_variant, mass) in cases:
            with self.subTest(llp=llp_name, variant=table_variant):
                ownership = _ownership(
                    portal,
                    source_variant,
                    mass,
                    ALP_POOLED if portal == ALP_PORTAL else SCALAR_POOLED,
                    (ALP_NONHADRONIC if portal == ALP_PORTAL
                     else SCALAR_NONHADRONIC),
                    0.5,
                )
                bridge = self._bridge(ownership)
                with mock.patch.dict(os.environ, {}, clear=True), (
                    mock.patch.object(exhadDecays, "_bridge", bridge)
                ), mock.patch.object(
                    exhadDecays, "_load_bridge", return_value=bridge
                ), contextlib.redirect_stdout(io.StringIO()):
                    exhadDecays.set_selection(
                        llp_name, path, table_variant)
                    self.assertEqual(
                        exhadDecays.get_bench(), expected_bench)
                    result = exhadDecays.process_events_with_exhad(
                        1, mass, seed=101)

                self.assertEqual(result.shape[0], 1)
                bridge.sample_spinzero_for_eventcalc.assert_called_once_with(
                    portal,
                    source_variant,
                    mass,
                    1,
                    seed=101,
                )
                bridge.sample_for_eventcalc.assert_not_called()


if __name__ == "__main__":
    unittest.main()
