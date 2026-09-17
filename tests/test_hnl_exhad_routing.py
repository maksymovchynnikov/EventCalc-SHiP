#!/usr/bin/env python3
"""What EventCalc may hand the sealed release for an HNL, and how it routes.

Two contracts are checked here.  The first is the invariant mass of the parton
pair: the release validates it event by event and refuses a pair below the
lightest state it builds for that current, so a sample that reaches lower kills
the whole scan point rather than one event.  The second is the routing itself:
each signed current is hadronized on its own, the explicit pole rows are never
sent to the hadronizer, and every row comes back with the number of events it
asked for.
"""

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from funcs import ThreeBodyDecay  # noqa: E402
from funcs import decayProducts, exhadDecays, initLLP, seeding, thresholds  # noqa: E402


#: The smallest hadronic invariant mass the sealed release builds for each
#: current, as ``compact/exhad/hnl.py`` states it: PRIMARY_W_MIN for the six
#: installed currents and NC_c, PYTHIA_W_MIN for the heavy ones it showers.
#: ``test_the_release_states_these_same_minima`` checks this table against the
#: installed release whenever one is configured.
RELEASE_W_MIN = {
    (2, -1): 0.27455,     # CC_ud, pi+ pi0
    (2, -3): 0.62866,     # CC_us, K+ pi0
    (4, -1): 2.004430,    # CC_cd, D0 pi+
    (4, -3): 2.358540,    # CC_cs, D0 K+
    (2, -5): 5.419600,    # CC_ub, showered
    (4, -5): 7.155000,    # CC_cb, showered
    (1, -1): 0.279140,    # NC_ud, pi+ pi-
    (2, -2): 0.279140,    # NC_ud
    (3, -3): 0.987360,    # NC_s, K+ K-
    (4, -4): 3.739240,    # NC_c, D+ D-
    (5, -5): 10.560000,   # NC_b, showered
}

#: The release compares W against its minimum with this slack.
RELEASE_W_TOLERANCE = 2e-6


def _parton_pair(pdg_list):
    """The current a row's two partons make, keyed as in ``RELEASE_W_MIN``.

    A row and its charge conjugate are the same current, so the key is written
    with the up-type quark positive.  ``None`` says the row is not a quark
    pair.
    """
    partons = [int(code) for code in pdg_list
               if abs(int(code)) in thresholds.PARTON_PDGS]
    if len(partons) != 2 or any(abs(code) == 21 for code in partons):
        return None
    first, second = (abs(code) for code in partons)
    if first == second:
        return (first, -first)
    up = next(abs(code) for code in partons if abs(code) in (2, 4))
    down = next(abs(code) for code in partons if abs(code) in (1, 3, 5))
    return (up, -down)


class SamplingFloorTests(unittest.TestCase):
    """The floor the sampler applies, against what the release accepts."""

    def test_every_parton_pair_reaches_its_release_minimum(self):
        for pair, needed in RELEASE_W_MIN.items():
            with self.subTest(pair=pair):
                floor = ThreeBodyDecay.parton_pair_sampling_floor(pair)
                self.assertGreaterEqual(floor, needed - RELEASE_W_TOLERANCE)

    def test_the_floor_is_never_below_the_physical_two_hadron_bound(self):
        for pair in RELEASE_W_MIN:
            with self.subTest(pair=pair):
                self.assertGreaterEqual(
                    ThreeBodyDecay.parton_pair_sampling_floor(pair),
                    thresholds.parton_system_threshold(pair))

    def test_a_parton_with_no_hadronizer_floor_is_refused(self):
        with self.assertRaises(ValueError):
            ThreeBodyDecay.parton_pair_sampling_floor((6, -6))

    def test_the_release_states_these_same_minima(self):
        root = os.environ.get("EXHAD_ROOT")
        if not root:
            self.skipTest("EXHAD_ROOT is not configured")
        if root not in sys.path:
            sys.path.insert(0, root)
        from exhad.hnl import CC_PAIRS, NC_CURRENTS, PRIMARY_W_MIN, PYTHIA_W_MIN
        stated = {}
        for name, (up, down) in CC_PAIRS.items():
            stated[(abs(up), -abs(down))] = PRIMARY_W_MIN.get(
                name, PYTHIA_W_MIN.get(name))
        for flavour, name in NC_CURRENTS.items():
            stated[(flavour, -flavour)] = PRIMARY_W_MIN.get(
                name, PYTHIA_W_MIN.get(name))
        self.assertEqual(set(stated), set(RELEASE_W_MIN))
        for pair, value in stated.items():
            with self.subTest(pair=pair):
                self.assertAlmostEqual(value, RELEASE_W_MIN[pair], places=6)


class SampledInvariantMassTests(unittest.TestCase):
    """Drawn events, not just the analytic bound."""

    @classmethod
    def setUpClass(cls):
        cls.llp = initLLP.LLP(
            mass=2.0,
            particle_selection={"particle_path": "Distributions/HNL",
                                "LLP_name": "HNL"},
            mixing_pattern=np.array([1 / 3., 1 / 3., 1 / 3.]))
        cls.llp.import_particle()

    def test_no_drawn_event_falls_below_the_release_minimum(self):
        llp = self.llp
        rows = [(index, [int(code) for code in np.asarray(row).ravel()
                         if int(code) != -999])
                for index, row in enumerate(llp.PDGs)]
        checked = 0
        for mass in (1.5, 2.0, 3.0, 4.0, 5.27):
            llp.set_mass(mass)
            llp.compute_mass_dependent_properties()
            gated, _ = thresholds.gate_rates(
                mass, llp.PDGs, np.asarray(llp.BrRatios_distr, dtype=float),
                hnl=True)
            for index, pdg_list in rows:
                if len(pdg_list) != 3 or gated[index] <= 0.:
                    continue
                pair = _parton_pair(pdg_list)
                if pair is None:
                    continue
                needed = RELEASE_W_MIN[pair]
                masses = thresholds.daughter_masses(pdg_list, hnl=True)
                seeding.seed_all(11)
                events = np.asarray(ThreeBodyDecay.block_random_energies(
                    mass, masses[0], masses[1], masses[2], 20000,
                    llp.Matrix_elements[index],
                    pdg_list[0], pdg_list[1], pdg_list[2]))
                lepton_is_first = abs(pdg_list[0]) not in thresholds.PARTON_PDGS
                energy = events[:, 0] if lepton_is_first else events[:, 1]
                lepton_mass = masses[0] if lepton_is_first else masses[2]
                w = np.sqrt(np.maximum(
                    mass ** 2 + lepton_mass ** 2 - 2 * mass * energy, 0.))
                checked += 1
                with self.subTest(channel=llp.decayChannels[index], mass=mass):
                    self.assertGreaterEqual(
                        float(w.min()), needed - RELEASE_W_TOLERANCE)
        self.assertGreater(checked, 0)

    def test_every_row_that_carries_rate_still_has_phase_space(self):
        """The floor must not silently empty a row the tables give a rate."""
        llp = self.llp
        rows = [(index, [int(code) for code in np.asarray(row).ravel()
                         if int(code) != -999])
                for index, row in enumerate(llp.PDGs)]
        empty = []
        for mass in np.linspace(llp.m_min_tabulated, llp.m_max_tabulated, 301):
            llp.set_mass(float(mass))
            llp.compute_mass_dependent_properties()
            gated, _ = thresholds.gate_rates(
                float(mass), llp.PDGs,
                np.asarray(llp.BrRatios_distr, dtype=float), hnl=True)
            for index, pdg_list in rows:
                if len(pdg_list) != 3 or gated[index] <= 0.:
                    continue
                pair = _parton_pair(pdg_list)
                if pair is None:
                    continue
                masses = thresholds.daughter_masses(pdg_list, hnl=True)
                (low1, high1), (low3, high3) = ThreeBodyDecay.dalitz_energy_ranges(
                    float(mass), masses[0], masses[1], masses[2],
                    pdg_list[0], pdg_list[1], pdg_list[2])
                if high1 <= low1 or high3 <= low3:
                    empty.append((llp.decayChannels[index], float(mass)))
        self.assertEqual(empty, [])


class RoutingTests(unittest.TestCase):
    """One hadronizer call per signed current; poles are EventCalc's own."""

    @classmethod
    def setUpClass(cls):
        cls.llp = initLLP.LLP(
            mass=2.0,
            particle_selection={"particle_path": "Distributions/HNL",
                                "LLP_name": "HNL"},
            mixing_pattern=np.array([1., 0., 0.]))
        cls.llp.import_particle()
        cls.llp.set_mass(2.0)
        cls.llp.compute_mass_dependent_properties()

    def _selection(self):
        """Two conjugate partonic rows and one explicit pole row."""
        names = list(self.llp.decayChannels)
        return [names.index("Jets-ude"), names.index("Jets-udebar"),
                names.index("Pie")]

    def test_signed_rows_are_hadronized_one_by_one_and_poles_are_not(self):
        llp = self.llp
        selected = self._selection()
        branching = np.zeros(len(llp.decayChannels))
        branching[selected] = [0.3, 0.3, 0.4]

        calls = []

        def spy(decay_events, pdg_list, mass, seed=1):
            calls.append([int(code) for code in pdg_list])
            return np.zeros((len(decay_events), 6), dtype=float)

        with mock.patch.object(exhadDecays, "set_selection"), \
                mock.patch.object(exhadDecays, "is_hnl_bench",
                                  return_value=True), \
                mock.patch.object(exhadDecays, "exclusive_enabled",
                                  return_value=False), \
                mock.patch.object(exhadDecays,
                                  "require_hnl_parent_masses_supported"), \
                mock.patch.object(exhadDecays, "hnl_channel_spec",
                                  side_effect=lambda pdg_list: {
                                      "current": "CC_ud", "supported": True}), \
                mock.patch.object(exhadDecays, "process_hnl_with_exhad", spy):
            events, sizes, labels = decayProducts.simulateDecays_rest_frame(
                2.0, llp.PDGs, branching, 10, llp.Matrix_elements,
                selected, float(branching[selected].sum()),
                llp_name="HNL", particle_path="Distributions/HNL", seed=19,
                return_process_labels=True)

        self.assertEqual(calls, [[2, -1, 11], [-2, 1, -11]])
        self.assertEqual(len(events), 10)
        self.assertEqual(int(np.sum(sizes)), 10)
        self.assertEqual(len(sizes), 3)
        # No HNL row is pooled: each keeps its own physical channel label.
        self.assertEqual(labels, [None, None, None])

    def test_a_route_that_returns_the_wrong_number_of_events_is_refused(self):
        llp = self.llp
        selected = self._selection()
        branching = np.zeros(len(llp.decayChannels))
        branching[selected] = [0.3, 0.3, 0.4]

        def short(decay_events, pdg_list, mass, seed=1):
            return np.zeros((max(len(decay_events) - 1, 0), 6), dtype=float)

        with mock.patch.object(exhadDecays, "set_selection"), \
                mock.patch.object(exhadDecays, "is_hnl_bench",
                                  return_value=True), \
                mock.patch.object(exhadDecays, "exclusive_enabled",
                                  return_value=False), \
                mock.patch.object(exhadDecays,
                                  "require_hnl_parent_masses_supported"), \
                mock.patch.object(exhadDecays, "hnl_channel_spec",
                                  side_effect=lambda pdg_list: {
                                      "current": "CC_ud", "supported": True}), \
                mock.patch.object(exhadDecays, "process_hnl_with_exhad", short), \
                self.assertRaisesRegex(RuntimeError, "returned"):
            decayProducts.simulateDecays_rest_frame(
                2.0, llp.PDGs, branching, 10, llp.Matrix_elements,
                selected, float(branching[selected].sum()),
                llp_name="HNL", particle_path="Distributions/HNL", seed=19)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
