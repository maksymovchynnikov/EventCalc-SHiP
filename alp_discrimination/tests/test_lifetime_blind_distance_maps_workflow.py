from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from alp_discrimination.cache import CacheStore
from alp_discrimination.config import QUICK
from alp_discrimination.plotting.lifetime import distance_map_interval_blocks
from alp_discrimination.templates.lifetime_banks import LifetimeTemplateBank
from alp_discrimination.workflows.lifetime_blind_distance_maps import (
    cached_distance_matrix,
    discover_template_bank_masses,
    distance_products,
    run_distance_map_workflow,
)


def _bank():
    common = {
        "n_events": np.array([11.0, 12.0]),
        "n_events_before_ecal": np.array([15.0, 16.0]),
        "epsilon_ecal_weighted": np.array([11.0 / 15.0, 0.75]),
        "total_n_eff": np.array([100.0, 100.0]),
        "interval_m": np.array([3.0, 30.0]),
    }
    return LifetimeTemplateBank(
        mass_gev=0.3,
        energy_edges_gev=np.array([0.3, 3.0, 400.0]),
        minimum_bin_n_eff=100.0,
        jeffreys_alpha=0.5,
        event_threshold=10.0,
        template_seed_offset=0,
        template_base_seed=54_321,
        photon_ctau_m=np.array([3.0, 20.0]),
        photon_probabilities=np.array([[0.8, 0.2], [0.6, 0.4]]),
        su2_ctau_m=np.array([4.0, 25.0]),
        su2_probabilities=np.array([[0.5, 0.5], [0.1, 0.9]]),
        profile="quick",
        selection_name="diphoton_ecal",
        **{
            f"photon_{name}": value.copy() for name, value in common.items()
        },
        **{f"su2_{name}": value.copy() for name, value in common.items()},
    )


def _disconnected_bank():
    common = {
        "n_events": np.array([11.0, 12.0, 13.0, 14.0]),
        "n_events_before_ecal": np.array([15.0, 16.0, 17.0, 18.0]),
        "epsilon_ecal_weighted": np.array([0.7, 0.75, 0.76, 0.77]),
        "total_n_eff": np.full(4, 100.0),
    }
    return LifetimeTemplateBank(
        mass_gev=0.3,
        energy_edges_gev=np.array([0.3, 3.0, 400.0]),
        minimum_bin_n_eff=100.0,
        jeffreys_alpha=0.5,
        event_threshold=2.3,
        template_seed_offset=0,
        template_base_seed=54_321,
        photon_ctau_m=np.array([1.0, 4.0, 25.0, 100.0]),
        photon_interval_index=np.array([1, 1, 0, 0]),
        photon_allowed_intervals_m=np.array([[25.0, 100.0], [1.0, 4.0]]),
        photon_interval_m=np.array([1.0, 100.0]),
        photon_probabilities=np.array(
            [[0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]]
        ),
        su2_ctau_m=np.array([0.5, 2.0, 20.0, 80.0]),
        su2_interval_index=np.array([1, 1, 0, 0]),
        su2_allowed_intervals_m=np.array([[20.0, 80.0], [0.5, 2.0]]),
        su2_interval_m=np.array([0.5, 80.0]),
        su2_probabilities=np.array(
            [[0.55, 0.45], [0.45, 0.55], [0.35, 0.65], [0.25, 0.75]]
        ),
        profile="quick",
        selection_name="diphoton_ecal",
        **{f"photon_{name}": value.copy() for name, value in common.items()},
        **{f"su2_{name}": value.copy() for name, value in common.items()},
    )


def _save_bank(bank, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **bank.arrays())


class LifetimeBlindDistanceMapWorkflowTests(unittest.TestCase):
    def test_exact_distance_products(self):
        bank = _bank()
        distances = np.array([[0.3, 0.7], [0.1, 0.5]])
        products = distance_products(bank, distances)
        self.assertEqual(products.summary["minimum_D_TV"], 0.1)
        self.assertEqual(products.summary["minimum_photon_lifetime_index"], 1)
        self.assertEqual(products.summary["minimum_su2_lifetime_index"], 0)
        self.assertEqual(len(products.distance_table), 4)
        self.assertAlmostEqual(
            products.minimum_pair_table["D_TV_bin_contribution"].sum(),
            0.1,
        )

    def test_disconnected_distance_products_retain_interval_provenance(self):
        bank = _disconnected_bank()
        distances = np.full((4, 4), 0.5)
        distances[2, 1] = 0.05
        distances[0, 3] = 0.95

        products = distance_products(bank, distances)

        self.assertEqual(products.summary["minimum_photon_interval_index"], 0)
        self.assertEqual(products.summary["minimum_su2_interval_index"], 1)
        self.assertEqual(products.summary["maximum_photon_interval_index"], 1)
        self.assertEqual(products.summary["maximum_su2_interval_index"], 0)
        minimum_row = products.distance_table.loc[
            np.isclose(products.distance_table["D_TV"], 0.05)
        ].iloc[0]
        self.assertEqual(int(minimum_row["photon_interval_index"]), 0)
        self.assertEqual(int(minimum_row["su2_interval_index"]), 1)
        self.assertTrue(
            np.all(
                products.minimum_pair_table["photon_interval_index"] == 0
            )
        )
        self.assertTrue(
            np.all(products.minimum_pair_table["su2_interval_index"] == 1)
        )

    def test_distance_plot_blocks_do_not_span_excluded_gaps(self):
        bank = _disconnected_bank()
        distances = np.arange(16, dtype=float).reshape(4, 4) / 16.0

        blocks = distance_map_interval_blocks(bank, distances)

        self.assertEqual(len(blocks), 4)
        self.assertEqual(
            {(photon, su2) for photon, su2, *_ in blocks},
            {(0, 0), (0, 1), (1, 0), (1, 1)},
        )
        for photon, su2, x_edges, y_edges, values in blocks:
            np.testing.assert_allclose(
                [x_edges[0], x_edges[-1]],
                bank.photon_allowed_intervals_m[photon],
            )
            np.testing.assert_allclose(
                [y_edges[0], y_edges[-1]],
                bank.su2_allowed_intervals_m[su2],
            )
            self.assertEqual(values.shape, (2, 2))
            self.assertFalse(x_edges[0] < 4.0 and x_edges[-1] > 25.0)
            self.assertFalse(y_edges[0] < 2.0 and y_edges[-1] > 20.0)

    def test_distance_cache_reuses_portable_bank_fingerprint(self):
        bank = _bank()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank_path = root / "template_bank_ma_0p3.npz"
            _save_bank(bank, bank_path)
            cache = CacheStore("quick", root / "cache")
            first = cached_distance_matrix(
                cache=cache,
                bank_path=bank_path,
                bank=bank,
            )
            second = cached_distance_matrix(
                cache=cache,
                bank_path=bank_path,
                bank=bank,
            )
            np.testing.assert_array_equal(first, second)
            np.testing.assert_allclose(first, [[0.3, 0.7], [0.1, 0.5]])
            self.assertEqual(cache.counter_snapshot()["writes"], 1)
            self.assertEqual(cache.counter_snapshot()["hits"], 1)

    def test_workflow_discovers_extended_masses_from_input_banks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "banks"
            _save_bank(
                replace(_bank(), mass_gev=1.0),
                input_dir / "template_bank_ma_1.npz",
            )
            _save_bank(
                replace(_bank(), mass_gev=2.5),
                input_dir / "template_bank_ma_2p5.npz",
            )

            self.assertEqual(
                discover_template_bank_masses(input_dir),
                (1.0, 2.5),
            )

            summary = run_distance_map_workflow(
                config=QUICK,
                cache=CacheStore("quick", root / "cache"),
                input_dir=input_dir,
                output_dir=root / "distance",
                requested_masses=(1.0, 2.5),
                make_plots=False,
            )

            self.assertEqual(summary["mass_GeV"].tolist(), [1.0, 2.5])

    def test_workflow_writes_tables_summary_and_portable_manifest(self):
        bank = _bank()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "banks"
            bank_path = input_dir / "template_bank_ma_0p3.npz"
            _save_bank(bank, bank_path)
            output_dir = root / "distance"
            cache = CacheStore("quick", root / "cache")
            summary = run_distance_map_workflow(
                config=QUICK,
                cache=cache,
                input_dir=input_dir,
                output_dir=output_dir,
            )
            self.assertAlmostEqual(summary.loc[0, "minimum_D_TV"], 0.1)
            self.assertTrue((output_dir / "distance_map_summary.csv").is_file())
            self.assertTrue(
                (output_dir / "tables" / "distance_map_ma_0p3.csv").is_file()
            )
            self.assertTrue(
                (output_dir / "plots" / "distance_map_ma_0p3.pdf").is_file()
            )
            self.assertTrue(
                (output_dir / "plots" / "minimum_pair_spectra_ma_0p3.png").is_file()
            )
            payload = json.loads((output_dir / "manifest.json").read_text())
            self.assertEqual(payload["profile"], "quick")
            self.assertTrue(payload["interval_aware_domains"])
            serialized = json.dumps(payload)
            self.assertNotIn("/Users/", serialized)
            self.assertNotIn(directory, serialized)
            with self.assertRaises(FileExistsError):
                run_distance_map_workflow(
                    config=QUICK,
                    cache=cache,
                    input_dir=input_dir,
                    output_dir=output_dir,
                )


if __name__ == "__main__":
    unittest.main()
