import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from alp_discrimination.cache import CacheStore, cache_key, canonical_json, file_fingerprint
from alp_discrimination.paths import REPOSITORY_ROOT, portable_path, profile_cache_dir


class CacheTests(unittest.TestCase):
    def test_key_is_deterministic_and_sensitive(self):
        identity = {"mass_gev": 0.3, "ctau_m": 10.0, "seed": 1, "selection": "mother_level", "samples": 20}
        self.assertEqual(cache_key(identity), cache_key(dict(reversed(list(identity.items())))))
        for field, value in (("mass_gev", 0.4), ("ctau_m", 11.0), ("seed", 2), ("selection", "ecal"), ("samples", 21)):
            changed = {**identity, field: value}
            self.assertNotEqual(cache_key(identity), cache_key(changed))

    def test_profiles_use_separate_namespaces(self):
        self.assertNotEqual(profile_cache_dir("smoke"), profile_cache_dir("production"))
        self.assertNotEqual(profile_cache_dir("quick"), profile_cache_dir("validation"))

    def test_atomic_write_read_and_corrupt_metadata_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CacheStore("smoke", Path(directory))
            identity = {"kind": "toy", "mass": 1.0}
            store.save("spectrum", identity, {"values": np.array([1.0, 2.0])}, {"label": "ok"})
            loaded = store.load("spectrum", identity)
            np.testing.assert_array_equal(loaded[0]["values"], [1.0, 2.0])
            _, metadata_path, _ = store.paths("spectrum", identity)
            metadata_path.write_text("{corrupt")
            self.assertIsNone(store.load("spectrum", identity))

    def test_semantically_corrupt_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CacheStore("smoke", Path(directory))
            identity = {"mass_gev": 0.3}
            store.save("spectrum", identity, {"values": np.array([1.0])}, {"mass_gev": 0.3})
            _, metadata_path, _ = store.paths("spectrum", identity)
            metadata = json.loads(metadata_path.read_text())
            metadata["mass_gev"] = 0.4
            metadata_path.write_text(json.dumps(metadata))
            self.assertIsNone(store.load("spectrum", identity))

    def test_repository_paths_are_serialized_portably(self):
        path = REPOSITORY_ROOT / "alp_discrimination" / "config.py"
        serialized = canonical_json({"input": path})
        fingerprint = file_fingerprint(path)

        self.assertIn('"input":"alp_discrimination/config.py"', serialized)
        self.assertEqual(fingerprint["path"], "alp_discrimination/config.py")
        self.assertNotIn(str(REPOSITORY_ROOT), serialized)
        self.assertNotIn(str(REPOSITORY_ROOT), json.dumps(fingerprint))

    def test_external_paths_are_explicitly_tagged(self):
        with tempfile.TemporaryDirectory() as directory:
            external = Path(directory) / "input.dat"
            external.write_bytes(b"test")
            serialized = portable_path(external)
            fingerprint_path = file_fingerprint(external)["path"]
            self.assertTrue(serialized.startswith("external:"))
            self.assertTrue(serialized.endswith(":input.dat"))
            self.assertEqual(fingerprint_path, serialized)
            self.assertNotIn(str(external.parent), serialized)
            self.assertNotIn("/Users/", serialized)

    def test_cache_counter_snapshot_records_reuse_and_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CacheStore("quick", Path(directory))
            identity = {"kind": "counter-test"}
            self.assertIsNone(store.load("toy", identity))
            store.save("toy", identity, {"values": np.array([1.0])}, {})
            self.assertIsNotNone(store.load("toy", identity))
            _, metadata_path, _ = store.paths("toy", identity)
            metadata_path.write_text("{corrupt")
            self.assertIsNone(store.load("toy", identity))

            self.assertEqual(
                store.counter_snapshot(),
                {"hits": 1, "misses": 2, "writes": 1, "rejected": 1},
            )
            snapshot = store.counter_snapshot()
            snapshot["hits"] = 100
            self.assertEqual(store.counter_snapshot()["hits"], 1)


if __name__ == "__main__":
    unittest.main()
