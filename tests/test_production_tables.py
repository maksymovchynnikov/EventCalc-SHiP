"""Storage changes must preserve values, ordering and contiguous columns."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from funcs.production_tables import distribution_path, pack_distribution, read_distribution


class ProductionTableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'DoubleDistr-test.txt'
        axes = ([0.2, 0.35], [0.001, 0.1, 0.4], [1.5, 3.0])
        xyz = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
        density = np.linspace(1e-90, 0.12345678901234567, len(xyz))
        np.savetxt(self.path, np.column_stack((xyz, density)), delimiter='\t', fmt='%.17g')

    def test_lossless_values_order_layout_and_text_precedence(self):
        old = pd.read_csv(self.path, sep='\t', header=None)
        packed = pack_distribution(self.path)
        new = read_distribution(packed)
        np.testing.assert_array_equal(old.to_numpy().view(np.uint64), new.to_numpy().view(np.uint64))
        self.assertEqual(old.to_numpy().strides, new.to_numpy().strides)
        self.assertTrue(all(new[i].to_numpy().flags.c_contiguous for i in range(4)))
        pd.testing.assert_frame_equal(old, new, check_exact=True)
        self.assertEqual(distribution_path(self.path), self.path)
        self.path.unlink()
        self.assertEqual(distribution_path(self.path), packed)
        pd.testing.assert_frame_equal(read_distribution(self.path), new, check_exact=True)

    def test_replace(self):
        packed = pack_distribution(self.path, replace_text=True)
        self.assertFalse(self.path.exists())
        self.assertTrue(packed.is_file())

    def test_no_overwrite(self):
        packed = pack_distribution(self.path)
        with self.assertRaises(FileExistsError):
            pack_distribution(self.path, replace_text=True)
        self.assertTrue(self.path.is_file())

    def test_bad_grid_retains_source(self):
        data = pd.read_csv(self.path, sep='\t', header=None).to_numpy()[::-1]
        np.savetxt(self.path, data, delimiter='\t')
        with self.assertRaisesRegex(ValueError, 'grid order'):
            pack_distribution(self.path, replace_text=True)
        self.assertTrue(self.path.exists())
        self.assertFalse(self.path.with_suffix('.npz').exists())

    def test_bad_binary_is_not_silently_replaced_with_text(self):
        packed = self.path.with_suffix('.npz')
        np.savez_compressed(packed, mass=np.array([1.], dtype='float32'),
                            theta=np.array([0.1]), energy=np.array([2.]),
                            density=np.ones((1, 1, 1)))
        with self.assertRaisesRegex(ValueError, 'Invalid production grid'):
            read_distribution(packed)

    def test_shape_validation(self):
        packed = self.path.with_suffix('.npz')
        np.savez_compressed(packed, mass=np.array([1.]), theta=np.array([0.1]),
                            energy=np.array([2.]), density=np.ones((2, 1, 1)))
        with self.assertRaisesRegex(ValueError, 'Invalid production grid'):
            read_distribution(packed)

    def test_photon_protection(self):
        with self.assertRaisesRegex(ValueError, 'ALP-photon'):
            pack_distribution(self.path.with_name('DoubleDistr-ALP-photon_primary.txt'))


if __name__ == '__main__':
    unittest.main()
