import unittest

import numpy as np

from analysis.ECAL import diphoton_ecal_acceptance
from analysis2.cache import cache_key
from analysis2.eventcalc_adapter import MotherSample
from analysis2.selections import (
    DIPHOTON_ECAL_ALGORITHM_VERSION,
    DiphotonECALSelection,
    ECALGeometry,
    SelectionContext,
    rectangular_ecal_hit_mask,
)


def fixed_mother_sample() -> MotherSample:
    mass = 1.0
    energy = np.array([2.0, 10.0, 100.0, 3.0])
    px = np.array([0.0, 0.1, 0.2, -0.1])
    py = np.array([0.0, -0.05, 0.1, 0.2])
    pz = np.sqrt(energy**2 - mass**2 - px**2 - py**2)
    return MotherSample(
        px_gev=px,
        py_gev=py,
        pz_gev=pz,
        energy_gev=energy,
        decay_probability=np.array([0.1, 0.2, 0.3, 0.4]),
        x_m=np.array([0.0, 0.5, -0.2, 1.0]),
        y_m=np.array([0.0, -0.2, 0.3, 0.1]),
        z_m=np.array([50.0, 70.0, 80.0, 94.0]),
        mass_gev=mass,
    )


def one_mother(*, energy: float, x_m: float = 0.0) -> MotherSample:
    mass = 1.0
    pz = np.sqrt(energy**2 - mass**2)
    return MotherSample(
        px_gev=np.array([0.0]),
        py_gev=np.array([0.0]),
        pz_gev=np.array([pz]),
        energy_gev=np.array([energy]),
        decay_probability=np.array([1.0]),
        x_m=np.array([x_m]),
        y_m=np.array([0.0]),
        z_m=np.array([95.0]),
        mass_gev=mass,
    )


def legacy_table(sample: MotherSample) -> np.ndarray:
    return np.column_stack(
        (
            sample.px_gev,
            sample.py_gev,
            sample.pz_gev,
            sample.energy_gev,
            np.full(len(sample), sample.mass_gev),
            np.full(len(sample), 22.0),
            sample.decay_probability,
            sample.x_m,
            sample.y_m,
            sample.z_m,
        )
    )


class DiphotonECALSelectionTests(unittest.TestCase):
    def test_fixed_events_match_legacy_ecal_exactly(self):
        sample = fixed_mother_sample()
        context = SelectionContext(source_seed=123, true_sample_seed=124)
        current = DiphotonECALSelection().details(sample, context)
        legacy = diphoton_ecal_acceptance(
            legacy_table(sample),
            seed=125,
            return_details=True,
        )

        field_names = (
            "event_mask",
            "photon_1_hit_mask",
            "photon_2_hit_mask",
            "photon_1_four_momentum",
            "photon_2_four_momentum",
            "photon_1_x_ecal_m",
            "photon_1_y_ecal_m",
            "photon_2_x_ecal_m",
            "photon_2_y_ecal_m",
        )
        for field_name in field_names:
            np.testing.assert_array_equal(
                getattr(current, field_name),
                getattr(legacy, field_name),
            )
        np.testing.assert_array_equal(
            current.event_mask,
            np.array([False, True, True, True]),
        )
        self.assertEqual(np.count_nonzero(current.event_mask) / len(sample), 0.75)
        self.assertEqual(
            sample.decay_probability[current.event_mask].sum()
            / sample.decay_probability.sum(),
            0.9,
        )

    def test_both_photons_must_hit_the_ecal(self):
        selection = DiphotonECALSelection()
        context = SelectionContext(source_seed=0, true_sample_seed=1)

        inside = selection.details(one_mother(energy=10.0), context)
        np.testing.assert_array_equal(inside.photon_1_hit_mask, [True])
        np.testing.assert_array_equal(inside.photon_2_hit_mask, [True])
        np.testing.assert_array_equal(inside.event_mask, [True])

        one_downstream = selection.details(one_mother(energy=1.0), context)
        self.assertEqual(
            int(one_downstream.photon_1_hit_mask[0])
            + int(one_downstream.photon_2_hit_mask[0]),
            1,
        )
        np.testing.assert_array_equal(one_downstream.event_mask, [False])

        outside = selection.details(one_mother(energy=10.0, x_m=10.0), context)
        np.testing.assert_array_equal(outside.photon_1_hit_mask, [False])
        np.testing.assert_array_equal(outside.photon_2_hit_mask, [False])
        np.testing.assert_array_equal(outside.event_mask, [False])

    def test_rectangle_edges_are_inclusive(self):
        mask = rectangular_ecal_hit_mask(
            np.array([-2.0, 2.0, -2.0001, 0.0]),
            np.array([3.0, -3.0, 0.0, 3.0001]),
            np.ones(4, dtype=bool),
        )
        np.testing.assert_array_equal(mask, [True, True, False, False])

    def test_context_is_required_and_resolves_source_seed_plus_two(self):
        selection = DiphotonECALSelection()
        context = SelectionContext(source_seed=42, true_sample_seed=43)
        self.assertEqual(selection.selection_seed(context), 44)
        with self.assertRaises(ValueError):
            selection.mask(fixed_mother_sample())

    def test_cache_identity_changes_with_geometry_and_ecal_seed(self):
        context = SelectionContext(source_seed=42, true_sample_seed=43)
        selection = DiphotonECALSelection()
        identity = selection.cache_identity(context)
        self.assertEqual(identity["algorithm_version"], DIPHOTON_ECAL_ALGORITHM_VERSION)
        self.assertEqual(identity["selection_seed"], 44)

        changed_geometry = DiphotonECALSelection(
            geometry=ECALGeometry(width_x_m=4.0001)
        )
        changed_seed = SelectionContext(source_seed=43, true_sample_seed=44)
        changed_offset = DiphotonECALSelection(seed_offset=3)
        keys = {
            cache_key(identity),
            cache_key(changed_geometry.cache_identity(context)),
            cache_key(selection.cache_identity(changed_seed)),
            cache_key(changed_offset.cache_identity(context)),
        }
        self.assertEqual(len(keys), 4)


if __name__ == "__main__":
    unittest.main()
