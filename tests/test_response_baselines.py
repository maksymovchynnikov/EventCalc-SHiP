from pathlib import Path
import unittest
import numpy as np
from eventcalc.sensitivity import DecayBatch, integrate
from eventcalc.experiments.ship import ShipResponse
from eventcalc.experiments.na62 import NA62Response, NA62Config

FIXTURES=Path(__file__).with_name('fixtures')


def load(path):
    with np.load(path,allow_pickle=False) as archive:
        a={k:archive[k] for k in archive.files}
    b=DecayBatch(a['mother'],a['daughters'],a['visible'],a['weights'],float(a['mass']),
        a['origins'],str(a['frame']),a['charges'],a.get('pdg_ids'))
    return b,a


class ResponseBaselineTests(unittest.TestCase):
    def test_original_ship(self):
        b,a=load(FIXTURES/'ship_baseline.npz')
        np.testing.assert_allclose(ShipResponse().prepare(b).evaluate(a['ctau_m']),a['expected'],rtol=2e-13,atol=1e-18)

    def test_frozen_na62_leptons(self):
        b,a=load(FIXTURES/'na62_leptons.npz')
        prepared=NA62Response().prepare(b)
        z=b.origins[prepared.event_index,2]+prepared.lower*b.mother[prepared.event_index,2]/b.mass
        self.assertTrue(np.all(z>=105.-1e-12))
        actual=prepared.evaluate(a['ctau_m'])
        np.testing.assert_allclose(actual[:,:6],a['expected'],rtol=2e-13,atol=1e-18)
        np.testing.assert_array_equal(actual[:,6:],0)
        self.assertGreater(actual[:,[2,5]].max(),0)

    def test_six_na62_masses_and_doubled_vertex_grid(self):
        paths=list((FIXTURES/'na62').glob('mass_*.npz'))
        self.assertEqual(len(paths),6)
        for path in paths:
            with self.subTest(mass=path.stem):
                b,a=load(path)
                response=NA62Response()
                actual=integrate([b],a['ctau_m'],experiment=response)
                np.testing.assert_allclose(actual,a['expected'],rtol=2e-12,atol=1e-18)
                cutflow=response.cutflow(b,a['ctau_m'])[0]
                np.testing.assert_allclose(cutflow,a['expected_cutflow'],rtol=2e-12,atol=1e-18)
                np.testing.assert_allclose(actual[:,6:],cutflow[0,:,-1,:],rtol=2e-12,atol=1e-18)
                fine=NA62Response(NA62Config(vertex_cells=256)).prepare(b).evaluate(a['ctau_m'])
                np.testing.assert_allclose(fine,a['expected_256'],rtol=2e-12,atol=1e-18)
