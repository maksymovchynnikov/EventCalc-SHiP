"""Absolute weights must retain negative lobes and valid rejection bounds."""
import unittest
import numpy as np
from funcs.sampling_bounds import FirstOrderModel
from funcs.ThreeBodyDecay import block_random_energies


class AbsoluteMatrixWeightsTest(unittest.TestCase):
    def test_real_and_complex_modulus_enclosures(self):
        for low, high in [(-3., -1.), (-2., 3.), (.2, 4.)]:
            x = FirstOrderModel.variable(low, high, 1)
            y = FirstOrderModel.variable(-.7, .4, 3)
            gx, gy = np.meshgrid(np.linspace(low, high, 101), np.linspace(-.7, .4, 101))
            for enclosed, values in [(np.abs(x), abs(gx)),
                                     (abs(x + 1j*y), abs(gx + 1j*gy))]:
                self.assertTrue(enclosed.is_real)
                self.assertTrue(np.all(values >= enclosed.real_low))
                self.assertTrue(np.all(values <= enclosed.real_high))

    def test_abs_sampling_keeps_negative_source_lobe(self):
        # For massless daughters at m=2, the allowed E3 interval has width E1.
        # Integrating E1*abs(E1-1/2) gives fraction 1/6 below E1=1/2;
        # zero-clamping would instead give zero.
        def weight(m, e1, e3, *masses):
            return np.abs(e1 - .5)
        state = np.random.get_state()
        try:
            np.random.seed(743210)
            energies = block_random_energies(2., 0., 0., 0., 20000, weight, 22, 22, 22)
        finally:
            np.random.set_state(state)
        self.assertAlmostEqual(float(np.mean(energies[:, 0] < .5)), 1./6., delta=.012)


if __name__ == '__main__':
    unittest.main()
