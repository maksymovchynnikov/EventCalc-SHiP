"""Exact scalar/vectorized boost agreement, including nontrailing padding."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from funcs.boost import tab_boosted_decay_products


def scalar_reference(m, momentum, daughters):
    velocity = momentum[:, :3] / momentum[:, 3, None]
    gamma = momentum[:, 3] / m
    v2 = velocity[:, 0]**2 + velocity[:, 1]**2 + velocity[:, 2]**2
    gf = (gamma - 1) / np.where(v2 == 0, 1e-12, v2)
    rows = daughters.reshape(len(momentum), -1, 6)
    output = np.zeros_like(rows)
    output[:, :, 5] = -999
    for i in range(len(rows)):
        dest = 0
        vx, vy, vz = velocity[i]
        for px, py, pz, energy, mass, pid in rows[i]:
            if pid == -999:
                continue
            dot = vx * px + vy * py + vz * pz
            output[i, dest] = [
                px + gamma[i] * vx * energy + gf[i] * vx * dot,
                py + gamma[i] * vy * energy + gf[i] * vy * dot,
                pz + gamma[i] * vz * energy + gf[i] * vz * dot,
                gamma[i] * (energy + vx * px + vy * py + vz * pz), mass, pid]
            dest += 1
    return output.reshape(daughters.shape)


class BoostExecutionTests(unittest.TestCase):
    def test_exact_scalar_replay(self):
        rng = np.random.default_rng(21324)
        for mass in (.5, 2., 3.):
            for n, k in ((1, 1), (300, 30)):
                with self.subTest(mass=mass, n=n, k=k):
                    mom = rng.normal(size=(n, 4)) * 100
                    mom[:, 3] = np.sqrt(mass**2 + (mom[:, :3]**2).sum(1))
                    mom[0] = [0., 0., 0., mass]
                    d = rng.normal(size=(n, k, 6))
                    d[:, :, 3] = np.sqrt(d[:, :, 4]**2 + (d[:, :, :3]**2).sum(2))
                    d[:, :, 5] = rng.choice([22, 211, -999], size=(n, k))
                    d = d.reshape(n, k*6)
                    self.assertTrue(np.array_equal(scalar_reference(mass, mom, d),
                        tab_boosted_decay_products(mass, mom, d)))

    def test_empty_and_invalid_shapes(self):
        self.assertEqual(tab_boosted_decay_products(2., np.empty((0, 4)),
            np.empty((0, 12))).shape, (0, 12))
        for mom in (np.zeros((1, 3)), np.zeros((2, 4))):
            with self.assertRaises(ValueError):
                tab_boosted_decay_products(2., mom, np.zeros((1, 12)))


if __name__ == '__main__':
    unittest.main()
