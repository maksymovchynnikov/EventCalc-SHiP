#!/usr/bin/env python3
"""Non-Pythia contracts for current-resolved HNL event routing."""

import contextlib
import io
import math
import os
import sys
import types
import unittest
from unittest import mock

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# These tests replace the numerical decay kernels and do not need numba.
# Lightweight stubs keep the routing contract runnable in EventCalc's minimal
# test environment, exactly as the existing fixed-mass routing tests do.
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
    "hadronic_W",
    "split_by_hadronic_W",
    "hnl_nc_flavor",
    "hnl_nc_channel",
    "hnl_cc_light_channel",
    "process_hnl_cc_with_exhad",
    "process_hnl_nc_with_exhad",
    "_rescale_rest_to_mass",
    "_ExactWThresholdError",
)
_ABSENT = [name for name in _ADAPTER_NAMES if not hasattr(exhadDecays, name)]
if _ABSENT:
    raise unittest.SkipTest(
        "funcs/exhadDecays.py does not define " + ", ".join(_ABSENT)
        + "; these tests were written against an adapter that did"
    )



CC_CURRENTS = {
    "CC_ud": (2, 1, True),
    "CC_us": (2, 3, True),
    "CC_cd": (4, 1, True),
    "CC_cs": (4, 3, True),
}

NC_CURRENTS = {
    1: ("NC_ud", "light", True),
    2: ("NC_ud", "light", True),
    3: ("NC_s", "strange", True),
    4: ("NC_c", None, False),
}

W_MIN = {
    "CC_ud": 0.13957 + 0.13498,
    "CC_us": 0.49368 + 0.13498,
    "CC_cd": 1.86966 + 0.13498,
    "CC_cs": 1.86966 + 0.49368,
    "NC_ud": 2.0 * 0.13957,
    "NC_s": 2.0 * 0.49368,
    "NC_c": 2.0 * 1.86966,
}

LEPTON_MASS = {11: 0.000511, 13: 0.105660, 15: 1.77682}


def _particle(px, py, pz, energy, mass, pdg):
    """One synthetic EventCalc eight-field particle record."""

    return [px, py, pz, energy, mass, float(pdg), 0.0, 1.0]


def _three_body_event(pdg_row, hadronic_mass):
    """Build an event whose first two (quark) records have invariant W."""

    active = [int(pdg) for pdg in pdg_row if int(pdg) != -999]
    if len(active) != 3:
        raise ValueError("synthetic current rows must have three products")
    # Back-to-back spatial momenta are unnecessary here: two rest-like
    # synthetic quark records give exactly qq=(0,0,0,W), which isolates the
    # routing boundary from any event-generator details.
    event = []
    event += _particle(0.0, 0.0, 0.0, hadronic_mass / 2.0,
                       0.0, active[0])
    event += _particle(0.0, 0.0, 0.0, hadronic_mass / 2.0,
                       0.0, active[1])
    event += _particle(0.0, 0.0, 0.0, 0.0, 0.0, active[2])
    return np.asarray([event], dtype=np.float64)


def _pole_event(pdg_row):
    active = [int(pdg) for pdg in pdg_row if int(pdg) != -999]
    event = []
    for pdg in active:
        event += _particle(0.0, 0.0, 0.0, 0.5, 0.1, pdg)
    return np.asarray([event], dtype=np.float64)


class HNLCurrentClassifierTest(unittest.TestCase):
    def test_parent_mass_window_is_the_generated_hnl_mass_interval(self):
        """The gate is the interval of HNL mass in which the release generates.

        The upper edge of the release's matched support bounds W, the invariant
        mass of the hadronic system, which an HNL shares with a charged lepton
        or a neutrino; HNL masses above that edge are generated.
        """
        low, high = exhadDecays.generation_window("hnl")
        self.assertEqual((low, high), (0.02, 40.0))
        self.assertFalse(exhadDecays.hnl_parent_mass_supported(0.019999))
        for mass in (0.02, 5.27, 5.28, 6.0, 10.0, 20.0, 40.0):
            with self.subTest(mass=mass):
                self.assertTrue(exhadDecays.hnl_parent_mass_supported(mass))
        self.assertFalse(exhadDecays.hnl_parent_mass_supported(40.000001))

        exhadDecays.require_hnl_parent_masses_supported([0.02, 3.0, 5.28, 40.0])
        for mass in (0.019, 40.1):
            with self.subTest(mass=mass):
                with self.assertRaisesRegex(
                        RuntimeError, r"HNL mass 0\.02 to 40 GeV"):
                    exhadDecays.require_hnl_parent_masses_supported([mass])

    def test_all_cc_ud_us_cd_cs_signs_and_mixings(self):
        for current, (up, down, supported) in CC_CURRENTS.items():
            for lepton in (11, 13, 15):
                cases = (
                    ([up, -down, lepton, -999], lepton, True),
                    ([-up, down, -lepton, -999], -lepton, False),
                )
                for pdgs, expected_lepton, conjugate in cases:
                    with self.subTest(current=current, pdgs=pdgs):
                        spec = exhadDecays.hnl_channel_spec(pdgs)
                        self.assertIsNotNone(spec)
                        self.assertEqual(spec["current"], current)
                        self.assertEqual(spec["kind"], "CC")
                        self.assertEqual(spec["lepton"], expected_lepton)
                        self.assertIs(spec["conjugate"], conjugate)
                        self.assertIs(spec["supported"], supported)

    def test_cc_lepton_sign_must_balance_hadronic_charge(self):
        # These are not exported HNL rows: the charged lepton has the same
        # electric-charge sign as the hadronic current.  A fail-closed
        # classifier must not silently reinterpret them as physical CC rows.
        for current, (up, down, _supported) in CC_CURRENTS.items():
            for lepton in (11, 13, 15):
                with self.subTest(current=current, orientation="particle"):
                    self.assertIsNone(exhadDecays.hnl_channel_spec(
                        [up, -down, -lepton, -999]))
                with self.subTest(current=current, orientation="conjugate"):
                    self.assertIsNone(exhadDecays.hnl_channel_spec(
                        [-up, down, lepton, -999]))

    def test_us_conjugation_comes_from_electric_charge(self):
        # The signed PDG sums are -1 and +1, respectively, and therefore do
        # not encode the hadronic electric charge.  u sbar is +1 and must be
        # conjugated from exhad's charge-minus reference convention.
        particle = exhadDecays.hnl_channel_spec([2, -3, 11, -999])
        antiparticle = exhadDecays.hnl_channel_spec([-2, 3, -11, -999])
        self.assertEqual(particle["current"], "CC_us")
        self.assertIs(particle["conjugate"], True)
        self.assertEqual(antiparticle["current"], "CC_us")
        self.assertIs(antiparticle["conjugate"], False)

    def test_nc_ud_s_c_classifiers_and_support_boundary(self):
        for quark, (current, flavor, supported) in NC_CURRENTS.items():
            for orientation, neutrino in ((quark, 12), (-quark, -12)):
                pdgs = [orientation, -orientation, neutrino, -999]
                with self.subTest(quark=quark, pdgs=pdgs):
                    spec = exhadDecays.hnl_channel_spec(pdgs)
                    self.assertIsNotNone(spec)
                    self.assertEqual(spec["current"], current)
                    self.assertEqual(spec["kind"], "NC")
                    self.assertEqual(spec["lepton"], neutrino)
                    self.assertIs(spec["supported"], supported)
                    self.assertEqual(exhadDecays.hnl_nc_flavor(pdgs), flavor)
                    self.assertIs(exhadDecays.hnl_nc_channel(pdgs),
                                  flavor is not None)

    def test_charged_charm_currents_route_to_their_dedicated_backend(self):
        calls = []

        def generate(current, w_values, seed=1, exact=False):
            calls.append((current, tuple(w_values), seed, exact))
            momentum = 0.5 * float(w_values[0])
            return [[
                momentum, 0.0, 0.0, momentum, 0.0, -211.0,
                -momentum, 0.0, 0.0, momentum, 0.0, 111.0,
            ]]

        bridge = types.SimpleNamespace(hnl_current_decays_at_W=generate)
        for current, pdgs, exact_w in (
                ("CC_cd", [4, -1, 11, -999], 2.2),
                ("CC_cs", [4, -3, 11, -999], 2.5)):
            event = _three_body_event(pdgs, exact_w)
            with self.subTest(current=current), \
                    mock.patch.object(exhadDecays, "_bridge", bridge), \
                    mock.patch.object(exhadDecays, "_load_bridge",
                                      return_value=bridge), \
                    contextlib.redirect_stdout(io.StringIO()):
                result = exhadDecays.process_hnl_cc_with_exhad(
                    event, pdgs, mass=3.0, seed=17)
            self.assertEqual(calls[-1][0], current)
            self.assertEqual(calls[-1][1], (exact_w,))
            # The selected particle-current row needs hadronic charge +1;
            # EventCalc conjugates the bridge's fixed charge-minus convention.
            np.testing.assert_array_equal(
                result[0, 5::6], [11.0, 211.0, 111.0])

    def test_explicit_poles_are_never_classified_or_routed(self):
        pole_rows = (
            [211, 11, -999, -999],       # N -> e pi
            [-321, -13, -999, -999],     # charge conjugate N -> mu K
            [411, 15, -999, -999],       # N -> tau D
            [111, 12, -999, -999],       # N -> nu pi0
            [333, 12, -999, -999],       # N -> nu phi
            [211, 111, 11, -999],        # explicit low-side two-pion row
        )
        for pdgs in pole_rows:
            with self.subTest(pdgs=pdgs):
                self.assertIsNone(exhadDecays.hnl_channel_spec(pdgs))
                self.assertFalse(exhadDecays.hnl_cc_light_channel(pdgs))
                self.assertIsNone(exhadDecays.hnl_nc_flavor(pdgs))
                with self.assertRaisesRegex(
                        ValueError, "not a current-resolved HNL partonic row"):
                    exhadDecays.split_by_hadronic_W(
                        _pole_event(pdgs), pdgs, parent_mass=4.0)


class HNLCurrentKinematicsTest(unittest.TestCase):
    def test_each_current_rejects_w_below_its_physical_minimum(self):
        rows = {
            "CC_ud": [2, -1, 11, -999],
            "CC_us": [2, -3, 11, -999],
            "CC_cd": [4, -1, 11, -999],
            "CC_cs": [4, -3, 11, -999],
            "NC_ud": [2, -2, 12, -999],
            "NC_s": [3, -3, 12, -999],
            "NC_c": [4, -4, 12, -999],
        }
        for current, pdgs in rows.items():
            event = _three_body_event(pdgs, W_MIN[current] - 1.0e-3)
            with self.subTest(current=current), self.assertRaisesRegex(
                    RuntimeError, "below the physical hadronic threshold"):
                exhadDecays.split_by_hadronic_W(
                    event, pdgs, w_cap=10.0)

    def test_e_mu_tau_endpoints_are_parent_mass_minus_lepton_mass(self):
        parent_mass = 3.0
        for lepton, lepton_mass in LEPTON_MASS.items():
            pdgs = [2, -1, lepton, -999]
            endpoint = parent_mass - lepton_mass
            too_high = _three_body_event(pdgs, endpoint + 1.0e-4)
            with self.subTest(lepton=lepton, side="above"), \
                    self.assertRaisesRegex(RuntimeError, "exceeded W endpoint"):
                exhadDecays.split_by_hadronic_W(
                    too_high, pdgs, w_cap=10.0, parent_mass=parent_mass)

            allowed = _three_body_event(pdgs, endpoint - 1.0e-4)
            with self.subTest(lepton=lepton, side="below"):
                low, high = exhadDecays.split_by_hadronic_W(
                    allowed, pdgs, w_cap=10.0, parent_mass=parent_mass)
                self.assertEqual(len(low), 1)
                self.assertEqual(len(high), 0)
                self.assertAlmostEqual(
                    exhadDecays.hadronic_W(allowed, pdgs)[0],
                    endpoint - 1.0e-4,
                    places=12,
                )

    def test_rescale_impossible_state_raises(self):
        impossible = (
            [0.0, 0.0, 0.0, 0.6, 0.6, 211],
            [0.0, 0.0, 0.0, 0.6, 0.6, -211],
        )
        with self.assertRaisesRegex(RuntimeError, "cannot fit exact W"):
            exhadDecays._rescale_rest_to_mass(impossible, 1.0)

    def test_rescale_rejects_any_stable_rest_mass_excess(self):
        target = 1.0
        excess = 5.0e-12
        impossible = (
            [0.0, 0.0, 0.0, 0.5 + excess, 0.5 + excess, 211],
            [0.0, 0.0, 0.0, 0.5, 0.5, -211],
        )
        with self.assertRaises(exhadDecays._ExactWThresholdError):
            exhadDecays._rescale_rest_to_mass(impossible, target)

    def test_cc_ud_just_below_center_threshold_retries_at_exact_w(self):
        exact_w = 1.87631053051
        neutron_mass = 0.939565
        proton_mass = 0.938272
        impossible_center_event = [[
            0.0, 0.0, 0.0, neutron_mass, neutron_mass, 2112.0,
            0.0, 0.0, 0.0, proton_mass, proton_mass, -2212.0,
        ]]
        momentum = 0.2
        pion_mass = 0.13957
        pion0_mass = 0.13498
        physical_exact_event = [[
            momentum, 0.0, 0.0,
            math.sqrt(momentum ** 2 + pion_mass ** 2), pion_mass, -211.0,
            -momentum, 0.0, 0.0,
            math.sqrt(momentum ** 2 + pion0_mass ** 2), pion0_mass, 111.0,
        ]]
        calls = []

        def generate(current, w_values, seed=1, exact=False):
            calls.append((current, tuple(w_values), seed, exact))
            return physical_exact_event if exact else impossible_center_event

        bridge = types.SimpleNamespace(hnl_current_decays_at_W=generate)
        event = _three_body_event([2, -1, 11, -999], exact_w)
        with mock.patch.object(exhadDecays, "_bridge", bridge), \
                mock.patch.object(exhadDecays, "_load_bridge",
                                  return_value=bridge), \
                contextlib.redirect_stdout(io.StringIO()):
            result = exhadDecays.process_hnl_cc_with_exhad(
                event, [2, -1, 11, -999], mass=2.5, seed=29)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], "CC_ud")
        self.assertFalse(calls[0][3])
        self.assertTrue(calls[1][3])
        self.assertEqual(calls[1][1], (exact_w,))
        self.assertEqual(
            calls[1][2],
            exhadDecays._hnl_exact_w_retry_seed(29, 0, 0, "CC_ud"),
        )
        # u dbar has charge +1. The exact-W retry returns the fixed charge -1
        # convention, which the caller conjugates exactly once.
        np.testing.assert_array_equal(result[0, 5::6], [11.0, 211.0, 111.0])
        hadrons = result[0, 6:].reshape(-1, 6)
        self.assertLessEqual(float(hadrons[:, 4].sum()), exact_w)
        np.testing.assert_allclose(
            hadrons[:, :4].sum(axis=0), [0.0, 0.0, 0.0, exact_w],
            rtol=0.0, atol=2.0e-9,
        )

    def test_nc_just_below_center_threshold_uses_same_exact_w_guard(self):
        exact_w = 1.87631053051
        proton_mass = 0.938272
        impossible_center_event = [[
            0.0, 0.0, 0.0, proton_mass, proton_mass, 2212.0,
            0.0, 0.0, 0.0, proton_mass, proton_mass, -2212.0,
        ]]
        momentum = 0.2
        pion_mass = 0.13957
        energy = math.sqrt(momentum ** 2 + pion_mass ** 2)
        physical_exact_event = [[
            momentum, 0.0, 0.0, energy, pion_mass, 211.0,
            -momentum, 0.0, 0.0, energy, pion_mass, -211.0,
        ]]
        calls = []

        def generate(current, w_values, seed=1, exact=False):
            calls.append((current, tuple(w_values), seed, exact))
            return physical_exact_event if exact else impossible_center_event

        bridge = types.SimpleNamespace(hnl_current_decays_at_W=generate)
        event = _three_body_event([2, -2, 12, -999], exact_w)
        with mock.patch.object(exhadDecays, "_bridge", bridge), \
                mock.patch.object(exhadDecays, "_load_bridge",
                                  return_value=bridge), \
                contextlib.redirect_stdout(io.StringIO()):
            result = exhadDecays.process_hnl_nc_with_exhad(
                event, [2, -2, 12, -999], mass=2.5, seed=31)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][0], "NC_ud")
        self.assertEqual(calls[1][1], (exact_w,))
        self.assertTrue(calls[1][3])
        np.testing.assert_array_equal(
            result[0, 5::6], [12.0, 211.0, -211.0])
        hadrons = result[0, 6:].reshape(-1, 6)
        self.assertLessEqual(float(hadrons[:, 4].sum()), exact_w)
        np.testing.assert_allclose(
            hadrons[:, :4].sum(axis=0), [0.0, 0.0, 0.0, exact_w],
            rtol=0.0, atol=2.0e-9,
        )

    def test_exact_w_retry_exhaustion_is_bounded_and_fail_closed(self):
        exact_w = 1.87631053051
        neutron_mass = 0.939565
        proton_mass = 0.938272
        impossible = [[
            0.0, 0.0, 0.0, neutron_mass, neutron_mass, 2112.0,
            0.0, 0.0, 0.0, proton_mass, proton_mass, -2212.0,
        ]]
        calls = []

        def generate(current, w_values, seed=1, exact=False):
            calls.append((current, tuple(w_values), seed, exact))
            return impossible

        bridge = types.SimpleNamespace(hnl_current_decays_at_W=generate)
        event = _three_body_event([2, -1, 11, -999], exact_w)
        pattern = (r"exact-W threshold retry failed at W=1\.87631053051: "
                   r"seed=\d+ attempt=8/8")
        with mock.patch.object(exhadDecays, "_bridge", bridge), \
                mock.patch.object(exhadDecays, "_load_bridge",
                                  return_value=bridge), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, pattern):
            exhadDecays.process_hnl_cc_with_exhad(
                event, [2, -1, 11, -999], mass=2.5, seed=37)
        self.assertEqual(len(calls), 1 + exhadDecays._HNL_EXACT_W_RETRY_LIMIT)
        self.assertTrue(all(call[3] for call in calls[1:]))
        self.assertEqual(
            len({call[2] for call in calls[1:]}),
            exhadDecays._HNL_EXACT_W_RETRY_LIMIT,
        )

    def test_rescale_has_exact_energy_momentum_and_on_shell_closure(self):
        pion_mass = 0.13957
        momentum = 0.3
        energy = math.sqrt(pion_mass ** 2 + momentum ** 2)
        products = (
            [momentum, 0.0, 0.0, energy, pion_mass, 211],
            [-momentum, 0.0, 0.0, energy, pion_mass, -211],
        )
        target = 1.0
        result = exhadDecays._rescale_rest_to_mass(products, target)
        array = np.asarray(result, dtype=np.float64)
        np.testing.assert_allclose(array[:, :3].sum(axis=0), 0.0,
                                   rtol=0.0, atol=2.0e-12)
        self.assertAlmostEqual(float(array[:, 3].sum()), target, places=12)
        np.testing.assert_allclose(
            array[:, 3] ** 2 - np.sum(array[:, :3] ** 2, axis=1),
            array[:, 4] ** 2,
            rtol=1.0e-12,
            atol=1.0e-13,
        )
        # Exercise the downstream invariant as well, without a generator.
        exhadDecays._assert_hadronic_p4(
            array.reshape(-1).tolist(), (0.0, 0.0, 0.0, target), "test")


class HNLEventRecordContractTest(unittest.TestCase):
    def test_mixed_multiplicity_six_field_padding_uses_only_minus999(self):
        one_particle = [0.0, 0.0, 0.0, 0.2, 0.13957, 211.0]
        two_particles = one_particle + [
            0.0, 0.0, 0.0, 0.3, 0.13957, -211.0]

        direct = exhadDecays._pad_six_field_events(
            [one_particle, two_particles])
        self.assertEqual(direct.shape, (2, 12))
        np.testing.assert_array_equal(
            direct[:, 5::6], np.asarray([[211.0, -999.0],
                                         [211.0, -211.0]]))
        self.assertFalse(np.any(direct[:, 5::6] == 0.0))

        concatenated = decayProducts.concatenate_padded_event_arrays((
            np.asarray([one_particle]),
            np.asarray([two_particles]),
        ))
        np.testing.assert_array_equal(concatenated[:, 5::6],
                                      direct[:, 5::6])
        self.assertFalse(np.any(concatenated[:, 5::6] == 0.0))

    def test_hnl_three_body_uses_current_quark_masses_without_pythia(self):
        captured = {}

        def fake_three_body(parent_mass, size, parameters):
            captured["parent_mass"] = parent_mass
            captured["size"] = size
            captured["parameters"] = parameters
            return np.zeros((size, 24), dtype=np.float64)

        fake_processed = np.asarray(
            [[0.0, 0.0, 0.0, 0.13957, 0.13957, 211.0]])
        with mock.patch.object(
                decayProducts, "_fixed_exhad_context",
                return_value=(None, [])), mock.patch.object(
                decayProducts.ThreeBodyDecay, "decay_products",
                side_effect=fake_three_body), mock.patch.object(
                decayProducts, "process_events_with_pythia",
                return_value=fake_processed), contextlib.redirect_stdout(
                io.StringIO()):
            result, sizes = decayProducts.simulateDecays_rest_frame(
                mass=3.0,
                PDGdecay=np.asarray([[2, -3, 13, -999]], dtype=int),
                BrRatio=np.asarray([1.0]),
                size=1,
                Msquared3BodyLLP=[None],
                selected_decay_indices=[0],
                br_visible_val=1.0,
                llp_name="HNL",
            )

        parameters = captured["parameters"]
        self.assertEqual(tuple(parameters[:3]), (2, -3, 13))
        np.testing.assert_allclose(
            parameters[3:6],
            (decayProducts.HNL_CURRENT_QUARK_MASS[2],
             decayProducts.HNL_CURRENT_QUARK_MASS[3], 0.105660),
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_array_equal(result, fake_processed)
        np.testing.assert_array_equal(sizes, [1])


if __name__ == "__main__":
    unittest.main()
