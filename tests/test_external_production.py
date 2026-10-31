import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from eventcalc.production.external import ExternalProduction, ParentBatch, density_to_probabilities


def manifest(kind='spectrum'):
    return dict(schema_version=1,kind=kind,frame='lab',momentum_order=['px','py','pz','E'],
        energy_unit='GeV',length_unit='m',angle_unit='rad',coordinate_frame='target at (0,0,0); beam +z',
        normalization='produced_parents_per_incident',mass_GeV=1.,arrays='arrays.npz',
        production_probability=.4,origin_m=[0,0,0])


class ExternalProductionTests(unittest.TestCase):
    def test_spectrum_normalization_chunk_identity_and_sample_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'input.json';meta=manifest()
            path.write_text(json.dumps(meta))
            np.savez(root/'arrays.npz',theta_edges=[0,.01,.02],energy_edges=[20,40,60],bin_probabilities=[[.1,.2],[.3,.4]])
            src=ExternalProduction(path,mass=1.)
            def collect(size):
                batches=list(src.batches(events=10000,seed=42,chunk_size=size))
                return [np.concatenate([getattr(b,k) for b in batches]) for k in ('momenta','origins','weights')]
            a,b=collect(10000),collect(97)
            for x,y in zip(a,b): np.testing.assert_array_equal(x,y)
            self.assertAlmostEqual(a[2].sum(),.4)
            meta['kind']='parents';path.write_text(json.dumps(meta))
            np.savez(root/'arrays.npz',momenta=a[0],origins=a[1],weights=a[2])
            supplied=ExternalProduction(path,mass=1.)
            np.testing.assert_array_equal(supplied.parents.momenta,a[0])
            with self.assertRaises(ValueError): ExternalProduction(path,mass=1.1)
            meta['normalization']='unknown';path.write_text(json.dumps(meta))
            with self.assertRaises(ValueError): ExternalProduction(path,mass=1.)

    def test_signed_weights_and_invalid_parents(self):
        p=np.array([[0,0,2,np.sqrt(5)]]*2)
        b=ParentBatch(p,np.zeros((2,3)),np.array([.7,-.2]),1.).validate()
        self.assertAlmostEqual(sum(c.weights.sum() for c in b.chunks(1)),.5)
        with self.assertRaises(ValueError): ParentBatch(p,np.zeros((2,3)),[1,np.nan],1.).validate()
        with self.assertRaises(ValueError): ParentBatch(p,np.zeros((2,3)),[1,1],2.).validate()
        with self.assertRaises(ValueError): ParentBatch([[0,0,0,1]],[[0,0,0]],[1],1.).validate()

    def test_density_measure(self):
        p=density_to_probabilities([[1,1],[1,1]],[0,.1,.3],[1,2,5])
        np.testing.assert_allclose(p,np.array([[1,3],[2,6]])/12)
        for edges in ([0,0,.1],[0,np.nan,.1],[-1,0,1]):
            with self.assertRaises(ValueError): density_to_probabilities(np.ones((2,2)),edges,[1,2,3])

    def test_bad_spectrum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'input.json';path.write_text(json.dumps(manifest()))
            for probs in ([[1,1]],[[np.nan,0]],[[1.1,-.1]]):
                np.savez(root/'arrays.npz',theta_edges=[0,.1],energy_edges=[1,2,3],bin_probabilities=probs)
                with self.assertRaises(ValueError): ExternalProduction(path,mass=1.)
