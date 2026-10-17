"""Combined production keeps source grids, yields and point-dependent bounds."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from funcs import kinematics
from eventcalc.runtime_generator import (
    RuntimeModelScan, _ProductionInverseCDF, _production_cdf,
)


def source(label, energies, density, rate, angles=(0.001, 0.01)):
    table = pd.DataFrame([(m, t, e, float(density)) for m in (1., 3.)
                          for t in angles for e in energies])
    maximum = pd.DataFrame([(m, t, energies[-1]) for m in (1., 3.) for t in angles])
    return label, table, maximum, rate


def mixture_grid(mass=2., c_tau=100., *, zero=False):
    mixture = kinematics.ProductionMixture([
        source('primary', (3., 5.), 2., lambda m: m-1),
        source('cascade', (10., 20.), 0. if zero else 0.5, lambda m: 10-m,
               angles=(0.001, 0.004, 0.01)),
    ])
    return kinematics.Grids(mixture, None, 1, mass, c_tau)


class ToyLLP:
    m_min_tabulated, m_max_tabulated = 1., 3.
    Energy_distr = None

    def __init__(self):
        self.Distr = kinematics.ProductionMixture([
            source('primary', (3., 8.), 2., lambda m: m-1),
            source('cascade', (10., 30.), .5, lambda m: 10-m),
        ])

    def set_mass(self, mass):
        self.mass = mass

    def compute_mass_dependent_properties(self):
        self.Yield = self.Distr.total_yield(self.mass)

    def set_c_tau(self, c_tau):
        self.c_tau_input = c_tau

    def kinematics_cache_token(self):
        return ('combined-test', id(self.Distr))


class CombinedRuntimeTests(unittest.TestCase):
    def test_yield_times_acceptance_not_yield_alone(self):
        grid = mixture_grid()
        cdf = _production_cdf(grid)
        # Constant densities have exact integrals even on different grids.
        integrals = np.array([2.*(5.-3.)*.009, .5*(20.-10.)*.009])
        weights = integrals*np.array([1., 8.])
        self.assertAlmostEqual(cdf.integral, weights.sum()/9., places=14)
        np.testing.assert_allclose(np.diff(np.r_[0., cdf.cdf]), weights/weights.sum(), rtol=1e-14)
        theta, energy = cdf.sample(100000, np.random.default_rng(7))
        first = energy < 6.
        self.assertAlmostEqual(first.mean(), weights[0]/weights.sum(), delta=.004)
        self.assertAlmostEqual(energy[first].mean(), 4., delta=.025)
        self.assertAlmostEqual(energy[~first].mean(), 15., delta=.04)
        self.assertTrue(np.all((theta >= .001) & (theta <= .01)))

    def test_normalization_and_component_cells_match_standalone(self):
        grid = mixture_grid(c_tau=.1)
        cdf = _production_cdf(grid)
        expected = []
        for (_, rate, component), (_, sampler) in zip(grid._mixture_components, cdf.components):
            reference = _ProductionInverseCDF(component)
            np.testing.assert_array_equal(sampler.density, reference.density)
            np.testing.assert_array_equal(sampler.cdf, reference.cdf)
            expected.append(rate*reference.integral)
        self.assertEqual(cdf.integral, sum(expected)/grid._mixture_total_yield)

    def test_parent_stream_does_not_depend_on_batch_boundaries(self):
        cdf = _production_cdf(mixture_grid())
        whole = cdf.sample(1000, np.random.default_rng(13))
        rng = np.random.default_rng(13)
        chunks = [cdf.sample(n, rng) for n in (0, 1, 31, 200, 768)]
        for i in range(2):
            np.testing.assert_array_equal(whole[i], np.concatenate([part[i] for part in chunks]))

    def test_zero_rate_and_zero_acceptance_are_distinct(self):
        grid = mixture_grid(mass=1.)  # primary yield vanishes
        cdf = _production_cdf(grid)
        self.assertEqual([label for label, _ in cdf.components], ['cascade'])
        component = _ProductionInverseCDF(grid._mixture_components[0][2])
        for actual, expected in zip(cdf.sample(100, np.random.default_rng(8)),
                                    component.sample(100, np.random.default_rng(8))):
            np.testing.assert_array_equal(actual, expected)
        grid = mixture_grid(zero=True)  # cascade yield is nonzero but acceptance vanishes
        cdf = _production_cdf(grid)
        self.assertEqual([label for label, _ in cdf.components], ['primary'])
        self.assertAlmostEqual(cdf.integral, .036/9.)
        with self.assertRaisesRegex(ValueError, 'zero polar acceptance'):
            _production_cdf(mixture_grid(mass=1., zero=True))

    def test_corrupt_component_is_not_silently_dropped(self):
        grid = mixture_grid()
        grid._mixture_components[1][2].source_grids.distr[:] = np.nan
        with self.assertRaises(ValueError):
            _production_cdf(grid)

    def test_scan_refreshes_mass_lifetime_and_active_sources_without_refilling(self):
        scan = RuntimeModelScan(SimpleNamespace(model='ALP-photon'))
        scan._initialized = True
        scan.runtime = SimpleNamespace(kinematics=kinematics)
        scan._llp_template = ToyLLP()
        with patch.object(kinematics, '_fill_distr_3D', wraps=kinematics._fill_distr_3D) as fill:
            _, first_grid, first = scan._compiled_point(1., 100.)
            _, second_grid, second = scan._compiled_point(2., .1)
            _, _, third = scan._compiled_point(2., 100.)
            _, _, cached = scan._compiled_point(1., 100.)
        self.assertIs(first, cached)
        self.assertEqual(fill.call_count, 2)
        self.assertEqual(len(first.components), 1)
        self.assertEqual(len(second.components), 2)
        self.assertNotEqual(second.integral, third.integral)
        self.assertEqual(first_grid._mixture_components[0][2].m, 1.)
        self.assertEqual(first_grid._mixture_components[0][2].c_tau, 100.)
        for _, _, component in second_grid._mixture_components:
            self.assertEqual(component.m, 2.)
            self.assertEqual(component.c_tau, .1)
        for grid, actual in ((first_grid, first), (second_grid, second)):
            direct = _production_cdf(kinematics.Grids(
                grid.Distr, None, 1, grid.m, grid.c_tau,
                theta_max_sim=scan.experiment.theta_max_rad,
                survival_energy_floor=scan.experiment.z_min_m/scan.experiment.survival_cutoff))
            self.assertEqual(direct.integral, actual.integral)


if __name__ == '__main__':
    unittest.main()
