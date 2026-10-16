"""Analytic and independent vertex-quadrature checks of lifetime scans."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from sensitivity import (DecayBatch as _DecayBatch, Geometry, Selection, prepare, integrate,
                         scan_replicas, threshold_intervals, ship_selections)


def DecayBatch(*args, **kwargs):
    # These analytic straight-line fixtures describe electrically neutral daughters.
    kwargs.setdefault('charges', np.zeros(np.asarray(args[1]).shape[:2]))
    return _DecayBatch(*args, **kwargs)


def sample(n=80, seed=137):
    rng = np.random.default_rng(seed)
    m = 3.
    p = rng.normal(size=(n, 3))*[.9, .9, 0.]
    p[:, 2] = rng.uniform(20, 180, n)
    mother = np.column_stack((p, np.sqrt(m*m+(p*p).sum(1))))
    # Three massless particles with equal rest energies, rotated independently
    # for each event. The third can stand for an invisible daughter.
    d = np.zeros((n, 3, 4))
    ring = np.column_stack((np.cos(np.arange(3)*2*np.pi/3),
                            np.sin(np.arange(3)*2*np.pi/3), np.zeros(3)))
    for i in range(n):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        d[i, :, :3] = ring@q
    d[:, :, 3] = 1.
    visible = np.ones((n, 3), bool)
    visible[::2, 2] = False
    return DecayBatch(mother, d, visible, np.full(n, 1/n), m,
                     rng.normal(size=(n, 3))*[.03, .03, .2], 'rest')


def numeric(batch, ctaus, geometry, selections, steps=65536):
    """Independent midpoint integration in laboratory z, without interval code."""
    (p, d, vis, w, orig), mass = batch.arrays()
    dz = (geometry.z_max-geometry.z_min)/steps
    z = geometry.z_min+(np.arange(steps)+.5)*dz
    result = np.zeros((len(ctaus), len(selections)))
    for i in range(len(p)):
        if p[i, 2] <= 0: continue
        xyz = orig[i]+(z-orig[i, 2])[:, None]*p[i, :3]/p[i, 2]
        widths = np.array([geometry.x_in, geometry.y_in])+(z[:, None]-geometry.z_min)/(geometry.z_max-geometry.z_min)*np.array([geometry.x_out-geometry.x_in, geometry.y_out-geometry.y_in])
        in_dv = ((np.abs(xyz[:, :2]) < widths/2).all(1)) & (z >= orig[i, 2])
        decay_density = mass/p[i, 2]/ctaus[None, :]*np.exp(-(z[:, None]-orig[i, 2])*mass/p[i, 2]/ctaus[None, :])
        for j, s in enumerate(selections):
            if s.min_particles == 0:
                mask = in_dv
            else:
                forward = d[i, :, 2] > 0
                slopes = np.divide(d[i, :, :2], d[i, :, 2, None], out=np.zeros_like(d[i, :, :2]), where=forward[:, None])
                hit = xyz[:, None, :2]+(geometry.aperture_z-z)[:, None, None]*slopes
                mask_hits = (np.abs(hit) < np.array([geometry.aperture_x, geometry.aperture_y])/2).all(2) & forward & vis[i] & (d[i, :, 3] > s.energy_min)
                count = mask_hits.sum(1)
                mask = in_dv & (count >= s.min_particles)
                if s.all_visible: mask &= count == vis[i].sum()
                if s.ip_max is not None:
                    summed = mask_hits@d[i, :, :3]
                    slope = np.divide(summed[:, :2], summed[:, 2, None], out=np.zeros_like(summed[:, :2]), where=summed[:, 2, None] > 0)
                    ip = np.linalg.norm(xyz[:, :2]-(z-geometry.target_z)[:, None]*slope, axis=1)
                    mask &= (summed[:, 2] > 0) & (ip <= s.ip_max)
            result[:, j] += w[i]*dz*(decay_density*mask[:, None]).sum(0)
    return result


class SensitivityTests(unittest.TestCase):
    def test_cached_correlated_variations(self):
        b = sample(120)
        c = np.geomspace(.002, 1e16, 90)
        prepared = prepare(b)
        factors = np.column_stack([np.ones(120), np.linspace(.2, 3., 120)])
        fast = prepared.response(factors).evaluate(c)
        for j in range(2):
            weighted = replace(b, weights=b.weights*factors[:, j])
            exact = prepare(weighted).evaluate(c)
            self.assertLess(np.max(np.abs(fast[j]-exact))/exact.max(), 3e-6)
        b.origins[:, 2] = 50.
        a = prepare(b)
        exact = a.evaluate(c)
        fast = a.response().evaluate(c)[0]
        self.assertLess(np.max(np.abs(fast-exact))/exact.max(), 3e-6)
        empty = prepare(b, selections=(Selection('impossible', energy_min=1e9),))
        self.assertEqual(empty.response().evaluate(c).sum(), 0.)

    def test_axis_exponential_and_extreme_lifetimes(self):
        p = np.array([[0., 0., 100., np.hypot(1., 100.)]])
        rest = np.array([[[.5, 0., 0., .5], [-.5, 0., 0., .5]]])
        batch = DecayBatch(p, rest, np.ones((1, 2), bool), np.array([2.]), 1., frame='rest')
        ct = np.array([1e-3, .1, 1., 1e3, 1e16])
        a = prepare(batch, geometry=Geometry()).evaluate(ct)
        exact = 2*np.exp(-.32/ct)*(-np.expm1(-.5/ct))
        np.testing.assert_allclose(a, np.broadcast_to(exact[:, None], a.shape), rtol=1e-13)

    def test_three_body_invisible_displaced_matches_quadrature(self):
        b = sample(28); c = np.array([.05, .5, 5., 500.])
        cuts = ship_selections()+(Selection('tight_ip', energy_min=3., ip_max=.04),)
        exact = prepare(b, geometry=Geometry(), selections=cuts).evaluate(c)
        reference = numeric(b, c, Geometry(), cuts)
        np.testing.assert_allclose(exact, reference, rtol=3e-4, atol=3e-8)
        self.assertGreater(exact[:, 2].sum(), exact[:, 4].sum())

    def test_rest_lab_and_streaming_normalization(self):
        b = sample(); c = [1., 100.]
        arrays, m = b.arrays()
        p, d, v, w, o = arrays
        lab = DecayBatch(p, d, v, w, m, o)
        expected = prepare(b).evaluate(c)
        np.testing.assert_allclose(prepare(lab).evaluate(c), expected, rtol=2e-13)
        chunks = [DecayBatch(p[s], d[s], v[s], w[s], m, o[s]) for s in (slice(0, 17), slice(17, None))]
        actual = integrate(chunks, c, exposure=50., production_scales=[2., 3.])
        np.testing.assert_allclose(actual, expected*np.array([100., 150.])[:, None], rtol=2e-13)

    def test_energy_and_visibility_are_explicit(self):
        b = sample(4)
        cuts = (Selection('impossible', energy_min=1e6), Selection('one', min_particles=1, energy_min=0.))
        a = prepare(b, selections=cuts).evaluate([5.])
        self.assertEqual(a[0, 0], 0.)
        self.assertGreater(a[0, 1], 0.)
        b.visible[:] = False
        self.assertEqual(prepare(b, selections=cuts).evaluate([5.]).sum(), 0.)

    def test_translation_and_backward_mother(self):
        b = sample(12)
        a = prepare(b).evaluate([1.])
        b.origins[:, 2] += 200
        g = Geometry(z_min=232., z_max=282., aperture_z=297.32, target_z=200., magnet_z_min=286., magnet_z_max=291.)
        np.testing.assert_allclose(prepare(b, geometry=g).evaluate([1.]), a, rtol=1e-12)
        # A mother starting downstream can enter the volume while moving back.
        p = np.array([[0., 0., -10., np.hypot(1., 10.)]])
        b = DecayBatch(p, np.array([[[.5, 0, 0, .5], [-.5, 0, 0, .5]]]), np.ones((1, 2), bool), np.ones(1), 1., np.array([[0., 0., 100.]]), 'rest')
        a = prepare(b, geometry=Geometry()).evaluate([1.])[0, 0]
        self.assertAlmostEqual(a, np.exp(-1.8)-np.exp(-6.8), places=13)

    def test_replicas_and_empty_stream(self):
        b = sample(3)
        a = prepare(b).evaluate([3.])
        mean, error = scan_replicas(lambda i: [replace(b, weights=b.weights*(i+1))], [3.])
        np.testing.assert_allclose(mean, 1.5*a)
        np.testing.assert_allclose(error, .5*a)
        self.assertEqual(integrate([], [3.]).sum(), 0.)

    def test_invalid_inputs_fail_before_scan(self):
        b = sample(3)
        for invalid in (replace(b, visible=b.visible.astype(int)),
                        replace(b, daughters=b.daughters[:, :2]),
                        replace(b, weights=np.array([np.nan, 0., 1.]))):
            with self.assertRaises(ValueError): prepare(invalid)
        with self.assertRaises(ValueError): prepare(b).evaluate([0.])
        with self.assertRaises(ValueError): integrate([], [1.], production_scales=[-1.])
        with self.assertRaises(ValueError): scan_replicas(lambda i: [b], [1.], replicas=1)

    def test_contours_preserve_disconnected_and_open_regions(self):
        a = threshold_intervals([1., 10., 100., 1000., 10000.], [1., 9., 1., 9., 9.], threshold=3.)
        np.testing.assert_allclose(a[0], [np.sqrt(10.), np.sqrt(1000.)])
        self.assertAlmostEqual(a[1][0], np.sqrt(100000.))
        self.assertIsNone(a[1][1])
        self.assertEqual(threshold_intervals([1., 10.], [3., 3.]), [(None, None)])
        self.assertEqual(threshold_intervals([1., 10.], [0., 0.]), [])

    def test_streamed_response_precision(self):
        b=sample(300);ct=np.geomspace(.002,1e7,101)
        exact=integrate([b],ct)
        response=integrate([b],ct,response_nodes=8192)
        np.testing.assert_allclose(response,exact,rtol=1e-3,atol=1e-10)
        from sensitivity import ResponseAccumulator
        acc=ResponseAccumulator(ship_selections(),Geometry(),distance_range=(1e-20,1e-19))
        with self.assertRaises(ValueError):acc.add(prepare(b))

    def test_npz_card_matches_api(self):
        from scan_sensitivity import run
        b = sample(6)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for i in range(2):
                np.savez(base/f'r{i}.npz', mother=b.mother, daughters=b.daughters,
                         visible=b.visible, weights=b.weights*(i+1), mass=b.mass,
                         origins=b.origins, frame=b.frame, charges=b.charges)
            card = dict(response_nodes=None, parameters=[1e-10, 1e-8], ctau_m=[10., .1], production_scales=[1e-10, 1e-8], exposure=1e12,
                        source=dict(kind='npz', replicas=['r0.npz', 'r1.npz'], chunk_size=2))
            result, meta = run(card, base=base)
            expected = integrate([b], card['ctau_m'], production_scales=card['production_scales'], exposure=card['exposure'])
            np.testing.assert_allclose(result['events'], 1.5*expected)
            np.testing.assert_allclose(result['integration_error_events'], .5*expected)
            json.dumps(meta)


if __name__ == '__main__':
    unittest.main()
