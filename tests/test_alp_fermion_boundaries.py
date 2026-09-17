#!/usr/bin/env python3
"""Focused regression tests for discrete ALP table ownership."""

import json
import math
import os
import pathlib
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from funcs.alp_fermion import (CHARM_PARTONIC, LIGHT_PARTONIC, NONHADRONIC,
                               PERSISTENT_EXCLUSIVE, build_br_interpolator,
                               load_exhad_boundaries, prepare_decay_rows)
from funcs import PDG, exhadDecays

try:
    import exhad_matched
    from exhad_matched.inputs.alp_fermion_authority import (
        build_alp_fermion_rate_authority)
    # the root of the exHad installation the imported package came from
    EXHAD_ROOT = os.fspath(
        pathlib.Path(exhad_matched.__file__).resolve().parents[3])
except ImportError:
    build_alp_fermion_rate_authority = None
    EXHAD_ROOT = None

NO_MATCHED = (
    "exhad_matched belongs to the exHad installation and is not part of this "
    "repository; put the directory that contains it on PYTHONPATH to run "
    "this test"
)


class ALPFermionBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = os.path.join(ROOT, "Distributions", "ALP-fermion")
        with open(os.path.join(folder, "ALP-fermion-decay.json")) as f:
            rows = json.load(f)
        cls.labels = [row[0] for row in rows]
        cls.pdgs = [row[1] for row in rows]
        cls.tables = [row[2] for row in rows]
        cls.m_start, cls.m_charm = load_exhad_boundaries(folder)
        with open(os.path.join(folder, "ctau-ALP-fermion.txt")) as stream:
            cls.ctau_rows = [
                (round(float(m), 12), float(value))
                for m, value in (line.split() for line in stream
                                 if line.strip())
            ]
        cls.ctau = dict(cls.ctau_rows)
        cls.get_br = staticmethod(build_br_interpolator(
            cls.labels,
            cls.pdgs,
            cls.tables,
            cls.m_start,
            cls.m_charm,
            cls.ctau_rows,
        ))
        cls.idx = {name: i for i, name in enumerate(cls.labels)}
        cls.authority = (
            None if build_alp_fermion_rate_authority is None
            else build_alp_fermion_rate_authority(project_root=EXHAD_ROOT))
        cls.exclusive = (set(cls.labels) - NONHADRONIC - LIGHT_PARTONIC -
                         CHARM_PARTONIC - PERSISTENT_EXCLUSIVE)
        cls.saved_env = {key: os.environ.pop(key, None)
                         for key in ("EXHAD_BENCH", "EXHAD_LLP")}
        # The adapter of this repository resolves the release itself and has
        # no bridge loader to stand in for; where one exists, it is replaced
        # for the duration of the class and restored afterwards.
        cls.saved_loader = getattr(exhadDecays, "_load_bridge", None)
        exhadDecays.set_selection("ALP-fermion", folder)
        exhadDecays.set_force(None)
        if cls.saved_loader is not None:
            exhadDecays._load_bridge = lambda: object()

    @classmethod
    def tearDownClass(cls):
        if cls.saved_loader is not None:
            exhadDecays._load_bridge = cls.saved_loader
        exhadDecays.set_selection(None, None)
        exhadDecays.set_force(None)
        for key, value in cls.saved_env.items():
            if value is not None:
                os.environ[key] = value

    def value(self, mass, label):
        return self.get_br(mass)[self.idx[label]]

    def source_value(self, mass, label):
        table = self.tables[self.idx[label]]
        return next(float(v) for m, v in table if abs(float(m) - mass) < 1e-9)

    def source_interpolated_value(self, mass, label):
        table = [(float(m), float(v)) for m, v in
                 self.tables[self.idx[label]]]
        if mass <= table[0][0]:
            return table[0][1]
        if mass >= table[-1][0]:
            return table[-1][1]
        for (m0, v0), (m1, v1) in zip(table, table[1:]):
            if m0 <= mass <= m1:
                x = (mass - m0) / (m1 - m0)
                return v0 + x * (v1 - v0)
        self.fail("mass is outside the ALP source grid")

    def ctau_value(self, mass):
        if mass <= self.ctau_rows[0][0]:
            return self.ctau_rows[0][1]
        if mass >= self.ctau_rows[-1][0]:
            return self.ctau_rows[-1][1]
        for (m0, v0), (m1, v1) in zip(
                self.ctau_rows, self.ctau_rows[1:]):
            if m0 <= mass <= m1:
                fraction = (mass - m0) / (m1 - m0)
                return v0 + fraction * (v1 - v0)
        self.fail("mass is outside the ALP lifetime grid")

    def reduced_width(self, mass, label):
        return self.value(mass, label) / self.ctau_value(mass)

    def test_low_midpoint_is_exclusive_only(self):
        br = self.get_br(1.9105)
        self.assertTrue(all(br[self.idx[c]] == 0.0
                            for c in LIGHT_PARTONIC | CHARM_PARTONIC))
        source_active = math.fsum(
            self.source_value(1.910, channel)
            for channel in self.exclusive)
        active = math.fsum(
            br[self.idx[channel]] for channel in self.exclusive)
        scale = active / source_active
        for channel in self.exclusive:
            self.assertAlmostEqual(br[self.idx[channel]],
                                   scale * self.source_value(
                                       1.910, channel),
                                   places=15)

    def test_exact_nucleon_rows_persist_across_partonic_handoff(self):
        for channel in PERSISTENT_EXCLUSIVE:
            self.assertEqual(
                self.value(1.910, channel),
                self.source_value(1.910, channel),
            )
            self.assertEqual(
                self.value(1.915, channel),
                self.source_value(1.915, channel),
            )
            values = [
                self.reduced_width(1.910 + 0.005 * index / 100, channel)
                for index in range(101)
            ]
            self.assertTrue(all(value > 0.0 for value in values))
            self.assertTrue(all(
                upper > lower
                for lower, upper in zip(values, values[1:])
            ))
            self.assertEqual(
                self.value(2.0, channel),
                self.source_value(2.0, channel),
            )

    def test_low_boundary_is_partonic_only(self):
        br = self.get_br(self.m_start)
        self.assertTrue(all(br[self.idx[c]] == 0.0 for c in self.exclusive))
        source_active = math.fsum(
            self.source_value(1.915, channel)
            for channel in LIGHT_PARTONIC)
        active = math.fsum(
            br[self.idx[channel]] for channel in LIGHT_PARTONIC)
        scale = active / source_active
        for channel in LIGHT_PARTONIC:
            self.assertAlmostEqual(br[self.idx[channel]],
                                   scale * self.source_value(
                                       1.915, channel),
                                   places=15)

    def test_high_midpoint_has_no_charm(self):
        # The raw-Pythia benchmark reported by a user at m_a=2.5 GeV must
        # contain only the light partonic rows.  Retrying a failed gg
        # hadronization attempt must never be confused with selecting the
        # separate a -> c cbar decay channel.
        self.assertEqual(self.get_br(2.5)[self.idx["Jets-cc"]], 0.0)
        br = self.get_br(3.7405)
        self.assertEqual(br[self.idx["Jets-cc"]], 0.0)
        for channel in LIGHT_PARTONIC:
            self.assertAlmostEqual(br[self.idx[channel]],
                                   self.source_value(3.740, channel), places=14)

    def test_high_boundary_uses_charm_aware_row(self):
        br = self.get_br(self.m_charm)
        self.assertGreater(br[self.idx["Jets-cc"]], 0.0)
        for channel in LIGHT_PARTONIC | CHARM_PARTONIC:
            self.assertAlmostEqual(br[self.idx[channel]],
                                   self.source_value(3.741, channel), places=14)

    def test_deleted_source_gap_bridges_only_the_nucleon_widths(self):
        masses = [round(float(m), 12) for m, _ in self.tables[0]]
        self.assertIn(1.910, masses)
        self.assertIn(1.915, masses)
        self.assertFalse(any(m in masses for m in
                             (1.911, 1.912, 1.913, 1.914)))
        first_complete = {
            channel: self.source_value(1.915, channel)
            for channel in self.labels
        }
        first_hadronic = math.fsum(
            value for channel, value in first_complete.items()
            if channel not in NONHADRONIC
        )
        first_active = math.fsum(
            first_complete[channel] for channel in LIGHT_PARTONIC
        )
        for mass in (self.m_start, 1.912, 1.913, 1.914):
            br = self.get_br(mass)
            self.assertEqual(math.fsum(br), 1.0)
            self.assertAlmostEqual(
                math.fsum(
                    br[index] for index, channel in enumerate(self.labels)
                    if channel not in NONHADRONIC),
                first_hadronic,
                places=15,
            )
            active = math.fsum(
                br[self.idx[channel]] for channel in LIGHT_PARTONIC)
            scale = active / first_active
            for channel in LIGHT_PARTONIC:
                self.assertAlmostEqual(
                    br[self.idx[channel]],
                    scale * first_complete[channel],
                    places=15,
                )
            for channel in NONHADRONIC:
                self.assertEqual(
                    br[self.idx[channel]], first_complete[channel])
        self.assertEqual(
            self.get_br(1.915),
            [first_complete[channel] for channel in self.labels],
        )

    def test_nucleon_width_bridge_matches_both_endpoint_tangents(self):
        step = 1.0e-7
        for channel in PERSISTENT_EXCLUSIVE:
            for mass in (1.910, 1.915):
                left = (
                    self.reduced_width(mass, channel)
                    - self.reduced_width(mass - step, channel)
                ) / step
                right = (
                    self.reduced_width(mass + step, channel)
                    - self.reduced_width(mass, channel)
                ) / step
                self.assertAlmostEqual(
                    left,
                    right,
                    delta=max(abs(left), abs(right)) * 2.0e-5,
                )

    @unittest.skipIf(build_alp_fermion_rate_authority is None, NO_MATCHED)
    def test_eventcalc_and_matched_authority_agree_through_source_gap(self):
        for mass in (1.9105, 1.911, 1.912, 1.913, 1.914, 1.9145, 1.915):
            with self.subTest(mass_gev=mass):
                observed = self.get_br(mass)
                expected = self.authority.point_at(mass)
                for index, channel in enumerate(expected.channels):
                    self.assertAlmostEqual(
                        observed[index],
                        channel.branching_fraction,
                        places=15,
                    )

    def test_production_numpy_label_array_is_supported(self):
        pdgs = np.empty(len(self.pdgs), dtype=object)
        pdgs[:] = [np.asarray(row, dtype=int) for row in self.pdgs]
        tables = np.empty(len(self.tables), dtype=object)
        tables[:] = self.tables
        get_br = build_br_interpolator(
            np.asarray(self.labels, dtype=object),
            pdgs,
            tables,
            self.m_start,
            self.m_charm,
            iter(self.ctau_rows),
        )
        for mass in (1.9105, 1.911, 1.913, 1.9145, 1.915, 2.0):
            self.assertEqual(get_br(mass), self.get_br(mass))

    def test_source_nodes_and_points_outside_the_gap_are_unchanged(self):
        for mass in (1.909, 1.910, 1.915, 1.916, 2.0, 3.741):
            with self.subTest(mass_gev=mass):
                observed = self.get_br(mass)
                for index, channel in enumerate(self.labels):
                    self.assertEqual(
                        observed[index],
                        self.source_value(mass, channel),
                    )

    def test_source_card_is_closed_and_kinematically_supported(self):
        for i in range(len(self.tables[0])):
            self.assertAlmostEqual(
                sum(float(table[i][1]) for table in self.tables),
                1.0, places=14)
        for ids, table in zip(self.pdgs, self.tables):
            threshold = sum(PDG.get_mass(int(pdg)) for pdg in ids)
            for mass, value in table:
                if float(mass) + 1.0e-12 < threshold:
                    self.assertEqual(float(value), 0.0)

    def test_interpolation_inside_each_regime_is_unchanged(self):
        for mass, channel, left, right in (
                (1.5005, "piPpiMpi0", 1.500, 1.501),
                (1.9155, "Jets-GG", 1.915, 1.916),
                (3.7415, "Jets-cc", 3.741, 3.742)):
            expected = 0.5 * (self.source_value(left, channel) +
                              self.source_value(right, channel))
            self.assertAlmostEqual(self.value(mass, channel), expected, places=12)

    def test_runtime_kinematic_gate_blocks_interpolation_below_threshold(self):
        # The source uses an approximate tau threshold, while EventCalc's
        # runtime mass is m_tau=1.77682 GeV.  Neither an open-grid value nor
        # interpolation toward it may create an off-shell tau pair.
        for mass in (3.550, 3.5535):
            self.assertEqual(self.value(mass, "tauPtauM"), 0.0)
            self.assertAlmostEqual(sum(self.get_br(mass)), 1.0, places=14)
        self.assertGreater(self.value(3.554, "tauPtauM"), 0.0)

    def test_lifetime_uses_the_same_physical_lepton_thresholds(self):
        # Below-threshold widths removed from the branching table must also be
        # removed from the total width encoded by ctau.
        self.assertGreater(self.ctau[0.211], 1.0e-3)
        self.assertLess(self.ctau[0.212], 1.0e-5)
        self.assertGreater(self.ctau[3.553], 1.0e-10)
        self.assertLess(self.ctau[3.554], 1.0e-10)

    def test_runtime_gate_uses_the_same_boundaries(self):
        if not exhadDecays.can_use_exhad():
            self.skipTest("no exHad release is configured, so the mass gate "
                          "is closed at every mass")
        self.assertFalse(exhadDecays.enabled(1.9105))
        self.assertTrue(exhadDecays.enabled(1.911))
        self.assertTrue(exhadDecays.enabled(3.7405))
        self.assertTrue(exhadDecays.enabled(3.741))
        self.assertTrue(exhadDecays.enabled(4.999))
        self.assertFalse(exhadDecays.enabled(5.001))

    def test_eventcalc_row_normalization(self):
        folder = os.path.join(ROOT, "Distributions", "ALP-fermion")
        with open(os.path.join(folder, "ALP-fermion-decay.json")) as f:
            rows = json.load(f)
        labels, pdgs, _, matrix_elements = prepare_decay_rows(rows)
        self.assertFalse(any(abs(pdg) >= 999990
                             for row in pdgs for pdg in row))
        self.assertTrue(all(2 <= len(row) <= 4 for row in pdgs))
        by_label = dict(zip(labels, matrix_elements))
        self.assertEqual(by_label["ePeM"], "1.")
        self.assertEqual(by_label["piPpiM2pi0"], "1.")
        self.assertEqual(by_label["KLKSpi0"], "1.")
        self.assertIn("UnitStep", by_label["piPpiMpi0"])

    def test_gluon_source_ids_are_canonical(self):
        folder = os.path.join(ROOT, "Distributions", "ALP-fermion")
        with open(os.path.join(folder, "ALP-fermion-decay.json")) as f:
            rows = json.load(f)
        by_label = {row[0]: [int(pdg) for pdg in row[1]] for row in rows}
        self.assertEqual(by_label["Jets-GG"], [21, 21])
        self.assertFalse(any(pdg == -21 for ids in by_label.values()
                             for pdg in ids))

    def test_three_body_dispatcher_is_rejected(self):
        bad = [["bad", [211, -211, 111], [[1.0, 1.0]],
                'Msquared3BodyLLP("x", "bad", E1, E3, mLLP, "ref")']]
        with self.assertRaisesRegex(ValueError, "dispatcher"):
            prepare_decay_rows(bad)

    def test_negative_gluon_id_is_rejected(self):
        bad = [["Jets-GG", [21, -21], [[1.0, 1.0]], "1."]]
        with self.assertRaisesRegex(ValueError, "gluon PDG id -21"):
            prepare_decay_rows(bad)

    def test_padding_sentinel_is_rejected(self):
        bad = [["bad", [211, -211, 999999], [[1.0, 1.0]], "1."]]
        with self.assertRaisesRegex(ValueError, "padding/sentinel"):
            prepare_decay_rows(bad)


if __name__ == "__main__":
    unittest.main()
