import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from analysis2.cache import CacheStore
from analysis2.config import QUICK
from analysis2.lifetime_template_banks import LifetimeTemplateBank
from analysis2.workflows.lifetime_blind_distance_maps import (
    cached_distance_matrix,
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
