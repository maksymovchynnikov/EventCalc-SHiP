#!/usr/bin/env python3
"""Contracts for EventCalc's manually populated Pythia event record."""

import os
import sys
import types
import unittest
import json
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# This contract exercises only the Pythia-record adapter, not the numerical
# decay kernels.  Keep it runnable in the workspace's minimal Python where
# numba is intentionally absent.
for module_name in ("TwoBodyDecay", "ThreeBodyDecay", "FourBodyDecay"):
    module = types.ModuleType("funcs." + module_name)
    module.decay_products = lambda *_args, **_kwargs: None
    sys.modules.setdefault("funcs." + module_name, module)

from funcs import decayProducts  # noqa: E402


class _Particle:
    def __init__(self, pdg=90):
        self._pdg = int(pdg)
        self._p = [0.0, 0.0, 0.0, 0.0]
        self._m = 0.0
        self._status = 0

    def p(self, px, py, pz, energy):
        self._p = [float(px), float(py), float(pz), float(energy)]

    def m(self, value=None):
        if value is not None:
            self._m = float(value)
        return self._m

    def status(self, value=None):
        if value is not None:
            self._status = int(value)
        return self._status

    def isFinal(self):
        return self._pdg == 22

    def px(self):
        return self._p[0]

    def py(self):
        return self._p[1]

    def pz(self):
        return self._p[2]

    def e(self):
        return self._p[3]

    def id(self):
        return self._pdg


class _Event:
    def __init__(self):
        self.system = _Particle()
        self.particles = [self.system]

    def reset(self):
        self.system = _Particle()
        self.particles = [self.system]

    def __getitem__(self, index):
        return self.particles[index]

    def __iter__(self):
        return iter(self.particles)

    def size(self):
        # EventCalc reads the record with explicit indices bounded by size(),
        # because the Pythia binding's event has no iterator.
        return len(self.particles)

    def append(self, pdg, _status, _m1, _m2, _d1, _d2, _col, _acol,
               px, py, pz, energy, mass):
        particle = _Particle(pdg)
        particle.p(px, py, pz, energy)
        particle.m(mass)
        self.particles.append(particle)


class _Pythia:
    def __init__(self):
        self.event = _Event()

    def readString(self, _setting):
        return True

    def init(self):
        return True

    def forceHadronLevel(self):
        # The regression: reset() leaves this record zeroed.  It must be
        # completed before forceHadronLevel performs its event check.
        assert self.event[0]._p == [0.0, 0.0, 0.0, 2.0]
        assert self.event[0].m() == 2.0
        assert self.event[0].status() == -11
        photon = _Particle(22)
        photon.p(0.0, 0.0, 0.0, 2.0)
        self.event.particles.append(photon)
        return True


class _PythiaModule:
    Pythia = _Pythia


class _RetryPythia(_Pythia):
    """Fake a stochastic hadronization failure that corrupts its record."""

    def __init__(self, failures_before_success):
        super().__init__()
        self.failures_before_success = int(failures_before_success)
        self.force_calls = 0
        self.input_snapshots = []

    def forceHadronLevel(self):
        snapshot = [
            (particle._pdg, tuple(particle._p), particle._m,
             particle._status)
            for particle in self.event.particles
        ]
        self.input_snapshots.append(snapshot)
        self.force_calls += 1

        # A failed Pythia call leaves an inconsistent, mutated record.  The
        # adapter must discard it and rebuild the saved manual input.
        self.event.particles.append(_Particle(999))
        if self.force_calls <= self.failures_before_success:
            return False

        photon = _Particle(22)
        photon.p(0.0, 0.0, 0.0, 2.0)
        self.event.particles.append(photon)
        return True


class _PythiaFactoryModule:
    def __init__(self, instance):
        self.instance = instance

    def Pythia(self):
        return self.instance


class _FinalParticle(_Particle):
    def isFinal(self):
        return True


class _GuardPythia(_Pythia):
    """Return supplied successful records, without correcting their p4."""
    def __init__(self, candidates):
        super().__init__()
        self.candidates = candidates
        self.input_snapshots = []
        self.force_calls = 0

    def forceHadronLevel(self):
        self.input_snapshots.append([
            (p.id(), tuple(p._p), p.m(), p.status()) for p in self.event])
        candidate = self.candidates[min(self.force_calls, len(self.candidates) - 1)]
        self.force_calls += 1
        if candidate is None:
            return False
        for px, py, pz, energy, mass, pdg in candidate:
            particle = _FinalParticle(pdg)
            particle.p(px, py, pz, energy)
            particle.m(mass)
            self.event.particles.append(particle)
        return True


class ManualPythiaRecordTest(unittest.TestCase):
    def test_system_record_is_completed_after_every_reset(self):
        decay = np.asarray([[0.0, 0.0, 0.0, 2.0,
                             0.0, 1.0, 0.0, 1.0]])
        with mock.patch.object(decayProducts, "load_pythia8",
                               return_value=_PythiaModule()):
            result = decayProducts._process_events_with_pythia_sequential(
                decay, mass=2.0)
        self.assertEqual(result.shape, (1, 6))
        self.assertEqual(result[0, 5], 22.0)

    def test_failed_hadronization_rebuilds_identical_input_and_recovers(self):
        decay = np.asarray([[0.0, 0.0, 1.0, 1.0,
                             0.0, 21.0, 0.0, 1.0,
                             0.0, 0.0, -1.0, 1.0,
                             0.0, 21.0, 0.0, 1.0]])
        pythia = _RetryPythia(failures_before_success=1)
        output = StringIO()
        with mock.patch.object(
                decayProducts, "load_pythia8",
                return_value=_PythiaFactoryModule(pythia)), \
                redirect_stdout(output):
            result = decayProducts._process_events_with_pythia_sequential(
                decay, mass=2.0, seed=17)

        self.assertEqual(pythia.force_calls, 2)
        self.assertEqual(pythia.input_snapshots[0],
                         pythia.input_snapshots[1])
        self.assertNotIn(999,
                         [entry[0] for entry in pythia.input_snapshots[1]])
        self.assertEqual(result.shape, (1, 6))
        self.assertEqual(result[0, 5], 22.0)
        self.assertIn("Recovered 1 Pythia event(s)", output.getvalue())
        self.assertIn("after 1 rejected hadronization attempt(s)",
                      output.getvalue())

    def test_persistent_hadronization_failure_remains_fail_closed(self):
        decay = np.asarray([[0.0, 0.0, 1.0, 1.0,
                             0.0, 21.0, 0.0, 1.0,
                             0.0, 0.0, -1.0, 1.0,
                             0.0, 21.0, 0.0, 1.0]])
        pythia = _RetryPythia(failures_before_success=100)
        with mock.patch.object(
                decayProducts, "load_pythia8",
                return_value=_PythiaFactoryModule(pythia)):
            with self.assertRaisesRegex(
                    RuntimeError,
                    r"after 10 attempts.*input_pdgs=\[21, 21\].*"
                    r"max_input_shell_residual=0.000e\+00"):
                decayProducts._process_events_with_pythia_sequential(
                    decay, mass=2.0, seed=19)

        self.assertEqual(
            pythia.force_calls,
            decayProducts.DEFAULT_PYTHIA8_HADRONIZATION_ATTEMPTS)
        self.assertEqual(len(pythia.input_snapshots), 10)
        self.assertTrue(all(
            snapshot == pythia.input_snapshots[0]
            for snapshot in pythia.input_snapshots[1:]))
        self.assertNotIn(999, [
            entry[0] for snapshot in pythia.input_snapshots
            for entry in snapshot
        ])


class RawP4GuardTest(unittest.TestCase):
    # Exact four-pion record captured in CERN's 1.85-GeV gg replay, index8085.
    BAD = np.asarray([
        [-.4635798096493021, -.20118900641216475, .4553129621373381,
         .6943866745671732, .13957, 211],
        [.12393744243479453, .10858704387909042, -.25986970633432144,
         .3378811698008103, .13957, -211],
        [.3106497745863829, -.006629030897499347, .19351300271970112,
         .39175785677538144, .13957, 211],
        [.028992869967427864, .09923097846650153, -.38895622921241013,
         .4259742062775797, .13957, -211],
    ])
    GOOD = np.asarray([[0., 0., .925, .925, 0., 22.],
                       [0., 0., -.925, .925, 0., 22.]])
    INPUT = np.asarray([[.7843295763209025, .2229699524499099, -.4367339190087697,
                         .925, 0., 21., 0., 1.,
                         -.7843295763209025, -.2229699524499099, .4367339190087697,
                         .925, 0., 21., 0., 1.]])

    def run_records(self, candidates, **kwargs):
        pythia = _GuardPythia(candidates)
        with mock.patch.object(decayProducts, "load_pythia8",
                               return_value=_PythiaFactoryModule(pythia)), \
                redirect_stdout(StringIO()):
            result = decayProducts.process_events_with_pythia(
                self.INPUT, 1.85, n_workers=1, seed=855373090, **kwargs)
        return result, pythia

    def test_disabled_guard_preserves_captured_record_byte_for_byte(self):
        implicit, pythia = self.run_records([self.BAD])
        explicit, second = self.run_records([self.BAD], p4_relative_tolerance=None)
        self.assertEqual(implicit.tobytes(), self.BAD.reshape(1, -1).tobytes())
        self.assertEqual(implicit.tobytes(), explicit.tobytes())
        self.assertEqual(pythia.force_calls, 1)
        self.assertEqual(second.force_calls, 1)

    def test_captured_defect_retries_identical_input_without_projecting_daughters(self):
        audit = {}
        output, pythia = self.run_records(
            [self.BAD, self.GOOD], p4_relative_tolerance=1e-7, p4_audit=audit)
        self.assertEqual(pythia.force_calls, 2)
        self.assertEqual(pythia.input_snapshots[0], pythia.input_snapshots[1])
        self.assertEqual(output.tobytes(), self.GOOD.reshape(1, -1).tobytes())
        self.assertEqual(audit["returned_events"], 1)
        self.assertEqual(audit["closure_rejected_attempts"], 1)
        self.assertEqual(audit["force_hadron_level_failures"], 0)
        self.assertEqual(audit["recovered_input_events"], 1)
        self.assertEqual(audit["maximum_attempts_used"], 2)
        self.assertEqual(audit["rejected_attempts"][0]["input_event_index"], 0)
        self.assertEqual(audit["rejected_attempts"][0]["final_pdgs"], [211, -211, 211, -211])
        self.assertAlmostEqual(audit["maximum_finite_rejected_residual_gev"],
                               2.7733930316187205e-7, places=18)
        json.dumps(audit, allow_nan=False)

    def test_guard_does_not_reseed_or_alter_valid_records(self):
        old, old_pythia = self.run_records([self.GOOD])
        audit = {}
        guarded, pythia = self.run_records(
            [self.GOOD], p4_relative_tolerance=1e-7, p4_audit=audit)
        self.assertEqual(old.tobytes(), guarded.tobytes())
        self.assertEqual(old_pythia.input_snapshots, pythia.input_snapshots)
        self.assertEqual(audit["closure_rejected_attempts"], 0)
        self.assertEqual(audit["total_attempts"], audit["returned_events"])

    def test_retry_preserves_all_input_events_and_their_order(self):
        pythia = _GuardPythia([self.GOOD, self.BAD, self.GOOD])
        audit = {}
        inputs = np.repeat(self.INPUT, 2, axis=0)
        with mock.patch.object(decayProducts, "load_pythia8",
                               return_value=_PythiaFactoryModule(pythia)), \
                redirect_stdout(StringIO()):
            output = decayProducts.process_events_with_pythia(
                inputs, 1.85, n_workers=1, p4_relative_tolerance=1e-7, p4_audit=audit)
        self.assertEqual(output.shape[0], 2)
        self.assertEqual(audit["input_events"], audit["returned_events"])
        self.assertEqual(audit["total_attempts"], 3)
        self.assertEqual(audit["rejected_attempts"][0]["input_event_index"], 1)
        self.assertEqual(output.tobytes(), np.repeat(self.GOOD.reshape(1, -1), 2, axis=0).tobytes())

    def test_threshold_uses_same_inclusive_max_component_check_as_worker(self):
        candidate = self.GOOD.copy()
        candidate[0, 0] = 1e-7 * 1.85
        audit = {}
        output, pythia = self.run_records(
            [candidate], p4_relative_tolerance=1e-7, p4_audit=audit)
        self.assertEqual(pythia.force_calls, 1)
        self.assertEqual(audit["closure_rejected_attempts"], 0)
        self.assertEqual(output.tobytes(), candidate.reshape(1, -1).tobytes())

    def test_true_return_with_persistent_bad_p4_fails_after_ten_attempts(self):
        audit = {}
        with self.assertRaisesRegex(RuntimeError, "p4 guard failed.*after 10 attempts"):
            self.run_records([self.BAD], p4_relative_tolerance=1e-7, p4_audit=audit)
        self.assertEqual(audit["returned_events"], 0)
        self.assertEqual(audit["closure_rejected_attempts"], 10)
        self.assertEqual(audit["total_attempts"], 10)

    def test_force_failure_is_distinguished_from_numerical_rejection(self):
        audit = {}
        _, pythia = self.run_records(
            [None, self.BAD, self.GOOD], p4_relative_tolerance=1e-7, p4_audit=audit)
        self.assertEqual(audit["force_hadron_level_failures"], 1)
        self.assertEqual(audit["closure_rejected_attempts"], 1)
        self.assertEqual(audit["total_attempts"], 3)
        self.assertEqual(audit["recovered_input_events"], 1)
        self.assertTrue(all(s == pythia.input_snapshots[0] for s in pythia.input_snapshots))

    def test_empty_and_nonfinite_records_are_not_accepted(self):
        bad = self.GOOD.copy()
        bad[0, 0] = np.nan
        for candidate in ([], bad):
            with self.subTest(candidate=str(candidate)):
                audit = {}
                output, _ = self.run_records(
                    [candidate, self.GOOD], p4_relative_tolerance=1e-7, p4_audit=audit)
                self.assertEqual(output.tobytes(), self.GOOD.reshape(1, -1).tobytes())
                self.assertEqual(audit["nonfinite_or_empty_rejections"], 1)
                json.dumps(audit, allow_nan=False)

    def test_guard_options_fail_closed_outside_sequential_positive_contract(self):
        for tolerance in (0., -1., float("nan"), float("inf")):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                self.run_records([self.GOOD], p4_relative_tolerance=tolerance)
        with self.assertRaisesRegex(ValueError, "requires an enabled guard"):
            self.run_records([self.GOOD], p4_audit={})
        with self.assertRaisesRegex(ValueError, "requires n_workers=1"):
            decayProducts.process_events_with_pythia(
                self.INPUT, 1.85, n_workers=2, p4_relative_tolerance=1e-7)


if __name__ == "__main__":
    unittest.main()
