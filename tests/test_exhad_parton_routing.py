#!/usr/bin/env python3
"""Regression tests for fixed-mass EventCalc/exhad channel routing."""

import contextlib
import io
import json
import os
import sys
import types
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# These tests replace the decay kernels.  Lightweight stubs keep the routing
# regression independent of EventCalc's optional numba installation.
for module_name in ("TwoBodyDecay", "ThreeBodyDecay", "FourBodyDecay"):
    module = types.ModuleType("funcs." + module_name)
    module.decay_products = lambda *_args, **_kwargs: None
    sys.modules.setdefault("funcs." + module_name, module)

from funcs import decayProducts, exhadDecays  # noqa: E402

# This module was written against an exHad adapter that carried its own lazily
# imported bridge object and its own HNL helpers.  The adapter of this
# repository, funcs/exhadDecays.py, talks to a configured exHad release instead
# and defines none of the names below, so the module cannot run as written.
# The check is by name, so the module runs again by itself if the adapter ever
# defines them.
_ADAPTER_NAMES = (
    "MATCHED_DARK_PHOTON_BACKEND",
    "_load_bridge",
    "hnl_cc_light_channel",
    "table_edge",
)
_ABSENT = [name for name in _ADAPTER_NAMES if not hasattr(exhadDecays, name)]
if _ABSENT:
    raise unittest.SkipTest(
        "funcs/exhadDecays.py does not define " + ", ".join(_ABSENT)
        + "; these tests were written against an adapter that did"
    )



def _processed_events(n_events):
    result = np.zeros((n_events, 6), dtype=np.float64)
    result[:, 5] = 22.0
    return result


class ExhadPartonRoutingTest(unittest.TestCase):
    def test_fixed_context_propagates_exhad_configuration_errors(self):
        rows = np.asarray([[1, -1]], dtype=np.int64)
        with mock.patch.object(exhadDecays, "set_selection"), \
                mock.patch.object(
                    exhadDecays, "is_hnl_bench",
                    side_effect=RuntimeError("malformed exhad card")):
            with self.assertRaisesRegex(RuntimeError,
                                        "malformed exhad card"):
                decayProducts._fixed_exhad_context(
                    2.0, rows, [0], "Dark-photons", "/tmp/dp", None)

    def test_deployed_dark_photon_uses_frozen_em_backend_and_four_scales(self):
        card_dir = os.path.join(ROOT, "Distributions", "Dark-photons")
        with open(os.path.join(card_dir, "exhad.json")) as stream:
            card = json.load(stream)
        self.assertEqual(card["bench"], "dv")
        self.assertEqual(
            card["backend"], exhadDecays.MATCHED_DARK_PHOTON_BACKEND)
        self.assertEqual(float(card["m_inclusive"]), 1.70)
        self.assertEqual(float(card["m_frag"]), 2.00)
        self.assertEqual(float(card["m_H"]), 4.00)
        self.assertEqual(float(card["m_free"]), 5.00)
        self.assertTrue(card["frozen_em_authority"])

        bridge = mock.Mock()
        bridge.sample_total_em_dark_photon_for_eventcalc.return_value = [
            [0.0, 0.0, 0.0, 1.70, 0.0, 22.0]
        ]
        selection = (
            mock.patch.object(exhadDecays, "_SEL_LLP", "Dark-photons"),
            mock.patch.object(exhadDecays, "_SEL_PATH", card_dir),
            mock.patch.object(exhadDecays, "_SEL_VARIANT", None),
        )
        with selection[0], selection[1], selection[2], (
            mock.patch.object(exhadDecays, "get_bench", return_value="dv")
        ), mock.patch.object(exhadDecays, "_load_bridge",
                             return_value=bridge):
            self.assertFalse(exhadDecays.enabled(1.699999))
            self.assertTrue(exhadDecays.enabled(1.70))
            self.assertTrue(exhadDecays.enabled(1.80))
            self.assertTrue(exhadDecays.enabled(5.00))
            self.assertFalse(exhadDecays.enabled(5.000001))

        with mock.patch.object(exhadDecays, "_SEL_LLP", "Dark-photons"), (
            mock.patch.object(exhadDecays, "_SEL_PATH", card_dir)
        ), mock.patch.object(exhadDecays, "_SEL_VARIANT", None), (
            mock.patch.object(exhadDecays, "enabled", return_value=True)
        ), mock.patch.object(exhadDecays, "get_bench", return_value="dv"), (
            mock.patch.object(exhadDecays, "_bridge", bridge)
        ), mock.patch.object(
            exhadDecays, "frozen_dark_photon_authority_path",
            return_value="/tmp/em_response_authority.json",
        ), contextlib.redirect_stdout(io.StringIO()):
            exhadDecays.process_events_with_exhad(1, 1.70, seed=7)
            exhadDecays.process_events_with_exhad(1, 1.80, seed=8)

        self.assertEqual(
            bridge.sample_total_em_dark_photon_for_eventcalc.call_count, 2)
        for call in (
                bridge.sample_total_em_dark_photon_for_eventcalc.call_args_list):
            self.assertEqual(
                call.args[0], "/tmp/em_response_authority.json")
            self.assertIsNone(call.kwargs["variation"])

    def test_matched_dark_photon_routes_uds_and_charm_but_not_gluons(self):
        card_dir = os.path.join(ROOT, "Distributions", "Dark-photons")
        rows = np.asarray(
            [[1, -1], [2, -2], [3, -3], [4, -4], [21, 21]],
            dtype=np.int64,
        )
        with (
            mock.patch.object(exhadDecays, "_SEL_LLP", "Dark-photons"),
            mock.patch.object(exhadDecays, "_SEL_PATH", card_dir),
            mock.patch.object(exhadDecays, "_SEL_VARIANT", None),
            mock.patch.object(exhadDecays, "_load_bridge",
                              return_value=mock.Mock()),
        ):
            _exhad, positions = decayProducts._fixed_exhad_context(
                4.5, rows, list(range(len(rows))),
                "Dark-photons", card_dir, None)
        self.assertEqual(positions, [0, 1, 2, 3])

    def test_matched_dark_photon_pools_selected_rows_once_without_double_mix(
            self):
        card_dir = os.path.join(ROOT, "Distributions", "Dark-photons")
        rows = np.asarray(
            [[1, -1], [2, -2], [3, -3], [4, -4], [21, 21]],
            dtype=np.int64,
        )
        generated = mock.Mock(
            side_effect=lambda n, _mass, seed=1: _processed_events(n))
        # The four diagonal rows carry 85 % of the rate and must become one
        # call; the gluon allocation is not part of the vector-current
        # total-EM measure.  The pool covers the largest share those rows can
        # draw at this size, since each lifetime at this mass draws its own.
        expected_pool = decayProducts.group_allocation_ceiling(20, 0.85)
        with (
            mock.patch.object(exhadDecays, "_SEL_LLP", "Dark-photons"),
            mock.patch.object(exhadDecays, "_SEL_PATH", card_dir),
            mock.patch.object(exhadDecays, "_SEL_VARIANT", None),
            mock.patch.object(exhadDecays, "enabled", return_value=True),
            mock.patch.object(exhadDecays, "process_events_with_exhad",
                              generated),
        ):
            pool = decayProducts.prepare_fixed_exhad_pool(
                5.0,
                20,
                rows,
                list(range(5)),
                BrRatio=[0.10, 0.20, 0.30, 0.25, 0.15],
                br_visible_val=1.0,
                llp_name="Dark-photons",
                particle_path=card_dir,
                seed=41,
                batch_index=2,
            )
        self.assertEqual(len(pool), expected_pool)
        self.assertGreaterEqual(expected_pool, 17)
        generated.assert_called_once()
        self.assertEqual(generated.call_args.args[:2], (expected_pool, 5.0))

    def test_partonic_predicate_is_narrower_than_pythia_predicate(self):
        self.assertTrue(decayProducts.channel_is_partonic([1, -1]))
        self.assertTrue(decayProducts.channel_is_partonic([21, 21, -999]))
        self.assertFalse(decayProducts.channel_is_partonic([21, -21, -999]))

        for channel in ([15, -15], [323, -323], [2, -1, 11],
                        [2212, -2212], [2112, -2112], []):
            with self.subTest(channel=channel):
                self.assertFalse(decayProducts.channel_is_partonic(channel))

        self.assertTrue(decayProducts.channel_requires_pythia([15, -15]))
        self.assertTrue(decayProducts.channel_requires_pythia([323, -323]))

    def test_fixed_bridge_passes_constrained_composition_through_unchanged(self):
        """EventCalc must not reinterpret exhad's post-constraint events."""
        event_one = [0.0, 0.0, 0.5, 1.1, 0.938, 2212.0,
                     0.0, 0.0, -0.5, 1.1, 0.938, -2212.0]
        event_two = [0.0, 0.0, 0.0, 2.0, 2.0, 22.0]
        bridge = mock.Mock()
        bridge.sample_for_eventcalc.return_value = [event_one, event_two]

        with (
            mock.patch.object(exhadDecays, "enabled", return_value=True),
            mock.patch.object(exhadDecays, "get_bench", return_value="alp"),
            mock.patch.object(exhadDecays, "table_edge", return_value=1.911),
            mock.patch.object(exhadDecays, "_bridge", bridge),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = exhadDecays.process_events_with_exhad(
                2, 2.0, seed=424242)

        bridge.sample_for_eventcalc.assert_called_once_with(
            "alp", 2.0, 2, seed=424242, residual=False)
        self.assertEqual(result.shape, (2, len(event_one)))
        np.testing.assert_array_equal(result[0], event_one)
        np.testing.assert_array_equal(result[1, :len(event_two)], event_two)
        np.testing.assert_array_equal(
            result[1, len(event_two):],
            [0.0, 0.0, 0.0, 0.0, 0.0, -999.0])

    def test_fixed_mass_exhad_replaces_only_partonic_rows(self):
        raw_channels = []

        def fake_two_body(_mass, size, _m1, _m2, pdg1, pdg2, *_args):
            result = np.zeros((size, 16), dtype=np.float64)
            result[:, 5] = pdg1
            result[:, 13] = pdg2
            return result

        def fake_raw(events, _mass, seed=1):
            del seed
            particles = np.asarray(events).reshape(len(events), 2, 8)
            pdgs = tuple(int(pdg) for pdg in particles[0, :, 5])
            raw_channels.append(pdgs)
            result = _processed_events(len(events))
            result[:, 0] = 100.0 if abs(pdgs[0]) == 15 else 200.0
            return result

        def fake_partonic(n, _mass, seed=1):
            del seed
            result = _processed_events(n)
            result[:, 0] = 10.0 + np.arange(n)
            return result

        partonic = mock.Mock(side_effect=fake_partonic)
        # Interleave two fixed-mass partonic rows with two rows that merely
        # require raw Pythia.  The exhad pool must be split back in this order.
        pdg_rows = np.array([[3, -3], [-15, 15], [1, -1], [323, -323]],
                            dtype=np.int64)
        # The channel allocation is a multinomial draw from NumPy's global
        # stream, which a driver seeds once per scan point.  This seed gives
        # the four equal rows two events each, so the routing of the pool back
        # onto the rows is read off a known allocation.
        np.random.seed(30)

        with (
            mock.patch.object(decayProducts.TwoBodyDecay, "decay_products",
                              side_effect=fake_two_body),
            mock.patch.object(decayProducts, "process_events_with_pythia",
                              side_effect=fake_raw),
            mock.patch.object(exhadDecays, "set_selection"),
            mock.patch.object(exhadDecays, "is_hnl_bench", return_value=False),
            mock.patch.object(exhadDecays, "enabled", return_value=True),
            mock.patch.object(exhadDecays, "process_events_with_exhad",
                              partonic),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result, sizes, process_labels = \
                decayProducts.simulateDecays_rest_frame(
                4.0, pdg_rows, [0.25] * 4, 8, None,
                [0, 1, 2, 3], 1.0, return_process_labels=True)

        partonic.assert_called_once()
        self.assertEqual(partonic.call_args.args[:2], (4, 4.0))
        self.assertCountEqual(raw_channels, [(-15, 15), (323, -323)])
        np.testing.assert_array_equal(sizes, [2, 2, 2, 2])
        np.testing.assert_array_equal(
            result[:, 0], [10.0, 11.0, 100.0, 100.0,
                           12.0, 13.0, 200.0, 200.0])
        self.assertEqual(
            process_labels,
            ["Jets-matched", None, "Jets-matched", None])

    def test_effective_provenance_requires_window_and_partonic_selection(self):
        rows = np.array([[3, -3], [211, -211]], dtype=np.int64)
        with (
            mock.patch.object(exhadDecays, "set_selection"),
            mock.patch.object(exhadDecays, "is_hnl_bench", return_value=False),
            mock.patch.object(exhadDecays, "get_bench", return_value="alp"),
            mock.patch.object(exhadDecays, "enabled", return_value=True),
        ):
            bench = decayProducts.effective_exhad_bench(
                2.0, rows, [0], llp_name="ALP-fermion",
                particle_path="/tmp/alp", exhad_variant=None)
            self.assertEqual(bench, "alp")
            self.assertIsNone(decayProducts.effective_exhad_bench(
                2.0, rows, [1], llp_name="ALP-fermion",
                particle_path="/tmp/alp", exhad_variant=None))

        with (
            mock.patch.object(exhadDecays, "set_selection") as selection,
            mock.patch.object(exhadDecays, "is_hnl_bench", return_value=False),
            mock.patch.object(exhadDecays, "enabled", return_value=False),
        ):
            self.assertIsNone(decayProducts.effective_exhad_bench(
                1.8, rows, [0], llp_name="Scalar-mixing",
                particle_path="/tmp/scalar",
                exhad_variant="2407.13587-Lower"))
            selection.assert_called_with(
                "Scalar-mixing", "/tmp/scalar", "2407.13587-Lower")

    def test_effective_provenance_requires_positive_branching_ratio(self):
        fixed_rows = np.array([[3, -3], [211, -211]], dtype=np.int64)
        with (
            mock.patch.object(exhadDecays, "set_selection"),
            mock.patch.object(exhadDecays, "is_hnl_bench",
                              return_value=False),
            mock.patch.object(exhadDecays, "get_bench",
                              return_value="dv"),
            mock.patch.object(exhadDecays, "enabled", return_value=True),
        ):
            self.assertIsNone(decayProducts.effective_exhad_bench(
                2.0, fixed_rows, [0], BrRatio=[0.0, 1.0]))
            self.assertEqual(decayProducts.effective_exhad_bench(
                2.0, fixed_rows, [0], BrRatio=[0.1, 0.9]), "dv")

        hnl_rows = np.array([[2, -2, 12], [211, 11, -999]],
                            dtype=np.int64)
        with (
            mock.patch.object(exhadDecays, "set_selection"),
            mock.patch.object(exhadDecays, "is_hnl_bench",
                              return_value=True),
            mock.patch.object(exhadDecays, "get_bench",
                              return_value="hnl"),
            mock.patch.object(
                exhadDecays,
                "hnl_channel_spec",
                side_effect=lambda pdgs: (
                    {"supported": True}
                    if 2 in tuple(int(value) for value in pdgs)
                    else None
                ),
            ),
        ):
            self.assertIsNone(decayProducts.effective_exhad_bench(
                1.0, hnl_rows, [0, 1], BrRatio=[0.0, 1.0]))
            self.assertEqual(decayProducts.effective_exhad_bench(
                1.5, hnl_rows, [0, 1], BrRatio=[0.2, 0.8]), "hnl")

    def test_direct_sample_detection_is_positive_rate_aware(self):
        rows = np.array([
            [-11, 11, -999],
            [211, -211, 111],
        ], dtype=np.int64)
        self.assertFalse(decayProducts.selected_sample_requires_pythia(
            rows, [1.0, 0.0], [0, 1]))
        self.assertTrue(decayProducts.selected_sample_requires_pythia(
            rows, [0.9, 0.1], [0, 1]))

    def test_precomputed_pool_is_reused_and_checked_for_sufficiency(self):
        def fake_two_body(_mass, size, _m1, _m2, pdg1, pdg2, *_args):
            result = np.zeros((size, 16), dtype=np.float64)
            result[:, 5] = pdg1
            result[:, 13] = pdg2
            return result

        pdg_rows = np.array([[3, -3], [1, -1]], dtype=np.int64)
        pool = _processed_events(5)
        pool[:, 0] = np.arange(5)
        generator = mock.Mock(side_effect=AssertionError(
            "precomputed pool should suppress generation"))

        patches = (
            mock.patch.object(decayProducts.TwoBodyDecay, "decay_products",
                              side_effect=fake_two_body),
            mock.patch.object(exhadDecays, "set_selection"),
            mock.patch.object(exhadDecays, "is_hnl_bench", return_value=False),
            mock.patch.object(exhadDecays, "enabled", return_value=True),
            mock.patch.object(exhadDecays, "process_events_with_exhad",
                              generator),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            with contextlib.redirect_stdout(io.StringIO()):
                result, _ = decayProducts.simulateDecays_rest_frame(
                    4.0, pdg_rows, [0.5, 0.5], 4, None, [0, 1], 1.0,
                    fixed_exhad_pool=pool)
                with self.assertRaisesRegex(AssertionError, "need 4, got 3"):
                    decayProducts.simulateDecays_rest_frame(
                        4.0, pdg_rows, [0.5, 0.5], 4, None,
                        [0, 1], 1.0, fixed_exhad_pool=pool[:3])

        generator.assert_not_called()
        np.testing.assert_array_equal(result[:, 0], [0.0, 1.0, 2.0, 3.0])

    def test_mass_pool_uses_exact_maximum_partonic_allocation(self):
        # At total size 10 these weights allocate [2, 5, 3] events.  Only the
        # first and third rows are fixed-mass partonic, so the reusable pool
        # should contain 5 events, not all 10.
        pdg_rows = np.array([[3, -3], [-15, 15], [1, -1]],
                            dtype=np.int64)
        generator = mock.Mock(side_effect=lambda n, _mass, seed=1:
                              _processed_events(n))
        with (
            mock.patch.object(exhadDecays, "set_selection"),
            mock.patch.object(exhadDecays, "is_hnl_bench", return_value=False),
            mock.patch.object(exhadDecays, "enabled", return_value=True),
            mock.patch.object(exhadDecays, "process_events_with_exhad",
                              generator),
        ):
            pool = decayProducts.prepare_fixed_exhad_pool(
                4.0, 10, pdg_rows, [0, 1, 2],
                BrRatio=[0.2, 0.5, 0.3], br_visible_val=1.0,
                seed=23, batch_index=7)

        self.assertEqual(len(pool), 5)
        generator.assert_called_once()
        self.assertEqual(generator.call_args.args[:2], (5, 4.0))
        self.assertEqual(
            generator.call_args.kwargs["seed"],
            decayProducts.derive_exhad_seed(23, 7, 0))

    def test_hnl_partonic_subsystem_keeps_special_handler(self):
        def fake_three_body(_mass, size, params):
            result = np.zeros((size, 24), dtype=np.float64)
            for index, pdg_id in enumerate(params[:3]):
                result[:, index * 8 + 5] = pdg_id
            return result

        def split_all_low_w(events, _pdgs, parent_mass=None):
            del parent_mass
            return events, np.empty((0, events.shape[1]), dtype=np.float64)

        hnl_handler = mock.Mock(
            side_effect=lambda events, _pdgs, _mass, seed=1:
            _processed_events(len(events)))
        fixed_mass = mock.Mock(side_effect=lambda n, _mass, seed=1:
                               _processed_events(n))
        raw_pythia = mock.Mock(side_effect=lambda events, _mass:
                               _processed_events(len(events)))
        output = io.StringIO()

        with (
            mock.patch.object(decayProducts.ThreeBodyDecay, "decay_products",
                              side_effect=fake_three_body),
            mock.patch.object(decayProducts, "process_events_with_pythia",
                              raw_pythia),
            mock.patch.object(exhadDecays, "set_selection"),
            mock.patch.object(exhadDecays, "is_hnl_bench", return_value=True),
            mock.patch.object(exhadDecays, "hnl_cc_light_channel",
                              return_value=True),
            mock.patch.object(exhadDecays, "hnl_nc_channel",
                              return_value=False),
            mock.patch.object(exhadDecays, "split_by_hadronic_W",
                              side_effect=split_all_low_w),
            mock.patch.object(exhadDecays, "process_hnl_cc_with_exhad",
                              hnl_handler),
            mock.patch.object(exhadDecays, "process_events_with_exhad",
                              fixed_mass),
            contextlib.redirect_stdout(output),
        ):
            decayProducts.simulateDecays_rest_frame(
                2.0, np.array([[2, -1, 11]], dtype=np.int64), [1.0], 2,
                [None], [0], 1.0,
            )

        hnl_handler.assert_called_once()
        fixed_mass.assert_not_called()
        raw_pythia.assert_not_called()
        self.assertIn("Events processed with Pythia: 0", output.getvalue())
        self.assertIn(
            "Events processed with matched exhad: 2", output.getvalue())


if __name__ == "__main__":
    unittest.main()
