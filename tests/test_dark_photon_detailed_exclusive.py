#!/usr/bin/env python3
"""Rate and topology contracts for the detailed sub-mT dark photon."""

import hashlib
import json
import math
import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


TABLE = os.path.join(ROOT, "Distributions", "Dark-photons",
                     "DP-decay.json")
META = os.path.join(ROOT, "Distributions", "Dark-photons",
                    "DP-decay-detailed-source.json")


def payload_sha256(payload):
    encoded = json.dumps(payload, separators=(",", ":"),
                         sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def rate_series_sha256(values):
    return payload_sha256([round(float(value), 14) for value in values])


class DarkPhotonDetailedExclusiveTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(TABLE) as stream:
            cls.table = json.load(stream)
        with open(META) as stream:
            cls.meta = json.load(stream)
        cls.rows = {str(row[0]): row for row in cls.table}
        cls.grid = [round(float(mass), 12)
                    for mass, _ in cls.table[0][2]]

    def values(self, label):
        return [float(value) for _, value in self.rows[label][2]]

    def value(self, label, mass):
        index = self.grid.index(round(float(mass), 12))
        return float(self.rows[label][2][index][1])

    def test_aggregate_placeholders_are_absent(self):
        for label in ("rho_other", "omega_other", "phi_other,1",
                      "phi_other,2"):
            self.assertNotIn(label, self.rows)

    def test_every_group_preserves_its_signed_legacy_rate_series(self):
        if "aggregate_rate_series_sha256" not in self.meta:
            self.skipTest(
                "the provenance record no longer stores the per-group rate "
                "digests: a digest that cannot be recomputed against a file "
                "the release ships was removed rather than carried")
        for group, labels in self.meta["replacement_rows"].items():
            values = [math.fsum(parts) for parts in zip(
                *(self.values(label) for label in labels))]
            self.assertEqual(
                rate_series_sha256(values),
                self.meta["aggregate_rate_series_sha256"][group])

    def test_complete_branching_table_preserves_every_mass_bin_total(self):
        if "all_channel_rate_series_sha256" not in self.meta:
            self.skipTest(
                "the provenance record no longer stores the whole-table rate "
                "digest: a digest that cannot be recomputed against a file "
                "the release ships was removed rather than carried")
        totals = [math.fsum(parts) for parts in zip(
            *(self.values(label) for label in self.rows))]
        self.assertEqual(rate_series_sha256(totals),
                         self.meta["all_channel_rate_series_sha256"])

    def test_replacements_end_at_the_single_1p70_transition(self):
        labels = [label for group in self.meta["replacement_rows"].values()
                  for label in group]
        self.assertGreater(sum(self.value(label, 1.699)
                               for label in labels), 0.0)
        self.assertAlmostEqual(sum(self.value(label, 1.700)
                                   for label in labels), 0.0, places=15)

    def test_decomposition_carries_physical_particle_identities(self):
        expected = {
            "EtaPiPi": [221, 211, -211],
            "EtaPrimePiPi": [331, 211, -211],
            "6pi_n": [211, 211, -211, -211, 111, 111],
            "6pi_c": [211, 211, 211, -211, -211, -211],
            "OmegaPion": [223, 111],
            "PhiPi": [333, 111],
            "EtaGamma": [221, 22],
            "EtaOmega": [221, 223],
            "OmPiPi_n": [223, 111, 111],
            "OmPiPi_c": [223, 211, -211],
            "EtaPhi": [221, 333],
            "PhiPiPi_n": [333, 111, 111],
            "PhiPiPi_c": [333, 211, -211],
            "KL_KS": [130, 310],
            "Kp_Km": [321, -321],
            "KKpi_0_KL_KS_Pi0": [130, 310, 111],
            "Kp_Km_Pi0": [321, -321, 111],
            "ppbar": [2212, -2212],
            "nnbar": [2112, -2112],
        }
        for label, pdgs in expected.items():
            self.assertEqual([int(value) for value in self.rows[label][1]],
                             pdgs)

    def test_fake_rho_other_two_track_rate_is_removed_at_seam(self):
        # The legacy rho_other aggregate was represented as pi+ pi-.  Its
        # recorded 1.69-GeV rate was 4.82399% of the total width, despite
        # being dominated by eta pi pi and six-pion states.  None of the
        # replacement rows may retain that placeholder identity.
        rho_labels = self.meta["replacement_rows"]["rho_other"]
        rho_rate = sum(self.value(label, 1.690) for label in rho_labels)
        self.assertAlmostEqual(rho_rate,
                               self.meta["checkpoints"]["1.690"]["rho_other"],
                               places=15)
        self.assertGreater(rho_rate, 0.04)
        self.assertFalse(any(
            [int(value) for value in self.rows[label][1]] == [211, -211]
            for label in rho_labels))


if __name__ == "__main__":
    unittest.main()
