#!/usr/bin/env python3
"""Contracts for the flavor-changing HNL CC_cd and CC_cs sources."""

from fractions import Fraction
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
HNL_DIR = ROOT / "Distributions" / "HNL"
AUTHORITY_PATH = HNL_DIR / "exhad-full-range-authority.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from funcs.hnl_source_validation import (  # noqa: E402
    load_canonical_hnl_source,
)


MIXING_INDEX = {"e": 0, "mu": 1, "tau": 2}
LEPTON_PDG = {"e": 11, "mu": 13, "tau": 15}
QUARK_CHARGE = {
    1: Fraction(-1, 3),
    2: Fraction(2, 3),
    3: Fraction(-1, 3),
    4: Fraction(2, 3),
    5: Fraction(-1, 3),
    6: Fraction(2, 3),
}


def _load_authority():
    with AUTHORITY_PATH.open(encoding="utf-8") as stream:
        return json.load(stream)


def _first_positive(table):
    for mass, value in table:
        if value > 0.0:
            return mass
    raise AssertionError("branching table has no positive node")


def _current_from_row(name):
    if name in {"Jets-uuv", "Jets-ddv"}:
        return "NC_ud"
    if name == "Jets-ssv":
        return "NC_s"
    if name == "Jets-ccv":
        return "NC_c"
    if name == "Jets-bbv":
        return "NC_b"
    for stem in ("ud", "us", "cd", "cs", "ub", "cb"):
        if name.startswith("Jets-" + stem):
            return "CC_" + stem
    raise AssertionError("unclassified current-resolved row %s" % name)


def _reference_quantum_numbers(partons):
    charge = sum(
        (1 if pdg > 0 else -1) * QUARK_CHARGE[abs(pdg)]
        for pdg in partons
    )
    baryon = sum(
        Fraction(1 if pdg > 0 else -1, 3) for pdg in partons)
    strangeness = sum(
        (-1 if pdg > 0 else 1)
        for pdg in partons if abs(pdg) == 3
    )
    charm = sum(
        (1 if pdg > 0 else -1)
        for pdg in partons if abs(pdg) == 4
    )
    return {
        "electric_charge": int(charge),
        "baryon_number": int(baryon),
        "strangeness": strangeness,
        "charm": charm,
    }


class HNLCharmCurrentAuthorityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.authority = _load_authority()
        cls.charm = cls.authority["charm_charged_current_sources"]
        cls.source = load_canonical_hnl_source()
        cls.by_name = {row[0]: row for row in cls.source.decay_rows}

    def test_complete_production_range_requires_exactly_six_currents(self):
        production_max = self.authority[
            "scope"]["production_parent_mass_domain_GeV"][1]
        positive_currents = set()
        for row in self.source.decay_rows:
            name = row[0]
            if not name.startswith("Jets-"):
                continue
            if any(
                value > 0.0
                for table in row[2:5]
                for mass, value in table
                if mass <= production_max
            ):
                positive_currents.add(_current_from_row(name))

        self.assertEqual(
            positive_currents,
            set(self.authority[
                "currents_required_through_production_endpoint"]),
        )
        self.assertEqual(
            positive_currents,
            {"CC_ud", "CC_us", "CC_cd", "CC_cs", "NC_ud", "NC_s"},
        )

    def test_charm_source_rows_and_first_positive_nodes_are_exact(self):
        for current, spec in self.charm.items():
            down = abs(spec["reference_partons"][1])
            stem = current.removeprefix("CC_")
            for mixing, row_names in spec["source_rows"].items():
                lepton = LEPTON_PDG[mixing]
                expected_rows = [
                    "Jets-" + stem + mixing,
                    "Jets-" + stem + mixing + "bar",
                ]
                with self.subTest(current=current, mixing=mixing):
                    self.assertEqual(row_names, expected_rows)
                    particle = self.by_name[row_names[0]]
                    conjugate = self.by_name[row_names[1]]
                    self.assertEqual(
                        particle[1], [4, -down, lepton, -999])
                    self.assertEqual(
                        conjugate[1], [-4, down, -lepton, -999])

                    index = MIXING_INDEX[mixing]
                    expected_onset = spec[
                        "first_positive_parent_mass_GeV"][mixing]
                    self.assertAlmostEqual(
                        _first_positive(particle[2 + index]),
                        expected_onset,
                        places=12,
                    )
                    self.assertAlmostEqual(
                        _first_positive(conjugate[2 + index]),
                        expected_onset,
                        places=12,
                    )

    def test_charm_sources_carry_the_declared_additive_quantum_numbers(self):
        for current, spec in self.charm.items():
            with self.subTest(current=current):
                observed = _reference_quantum_numbers(
                    spec["reference_partons"])
                self.assertEqual(
                    observed, spec["reference_quantum_numbers"])
                conjugate = _reference_quantum_numbers(
                    spec["conjugate_partons"])
                self.assertEqual(
                    conjugate,
                    {key: -value for key, value in observed.items()},
                )

    def test_charm_residuals_cannot_reuse_the_neutral_em_ccbar_source(self):
        for current, spec in self.charm.items():
            source = spec["conditional_event_source"]
            with self.subTest(current=current):
                self.assertEqual(
                    source["kind"],
                    "flavor-changing-single-open-charm-current",
                )
                self.assertTrue(source["evaluate_at_exact_W"])
                self.assertTrue(source["must_veto_explicit_owners"])
                self.assertTrue(
                    source["must_conserve_reference_quantum_numbers"])
                self.assertFalse(
                    source["reuse_neutral_electromagnetic_ccbar_source"])
                self.assertEqual(
                    source["provider_state"], "connected-and-tested")

    def test_explicit_D_and_Ds_rows_remain_separate_EventCalc_owners(self):
        owned = set()
        for current, spec in self.charm.items():
            explicit = set(spec["explicit_EventCalc_owners"])
            with self.subTest(current=current):
                self.assertTrue(explicit)
                self.assertTrue(explicit.isdisjoint(owned))
                self.assertTrue(explicit.issubset(self.by_name))
                self.assertTrue(all(
                    not name.startswith("Jets-") for name in explicit))
            owned.update(explicit)

    def test_NC_charm_does_not_open_inside_the_production_grid(self):
        production_max = self.authority[
            "scope"]["production_parent_mass_domain_GeV"][1]
        declared = self.authority[
            "currents_inactive_through_production_endpoint"]["NC_c"]
        self.assertGreater(
            declared["first_positive_parent_mass_GeV"], production_max)
        for mixing_index in range(3):
            table = self.by_name["Jets-ccv"][2 + mixing_index]
            first = _first_positive(table)
            self.assertAlmostEqual(
                first, declared["first_positive_parent_mass_GeV"],
                places=12,
            )


if __name__ == "__main__":
    unittest.main()
