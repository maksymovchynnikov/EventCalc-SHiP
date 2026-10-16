"""Compact sampling preserves normalization and works without the EOS dataset."""
from pathlib import Path
from unittest.mock import patch
import tempfile, unittest, numpy as np
from funcs.simulation_config import config_from_mapping
from eventcalc.runtime_generator import ExperimentCard, RuntimeModelScan
from eventcalc.production.dark_photon.compact import Density,CompactSource
from eventcalc.production.dark_photon.datasets import resolve_full_data,install_archive
ROOT=Path(__file__).resolve().parents[1]

def config(model='Dark-photons',**extra):
    v=dict(model=model,events=100,masses=[20 if model=='Inelastic-DM' else 1.8],c_taus=[1],seed=7,exhad_mode='auto' if model=='Inelastic-DM' else 'off')
    v.update(extra);return config_from_mapping(v,project_root=ROOT)

class CompactProductionTests(unittest.TestCase):
    def test_bilinear_inverse_cdf_and_clipped_integral(self):
        t=np.array([0.,.1,.2]);e=np.array([1.,2.,3.]);d=(1+t[:,None]*5)*(1+e[None,:])
        sampler=Density(t,e,d,1.,.15)
        self.assertAlmostEqual(sampler.integral,1.2375)
        a,b=sampler.sample(150000,np.random.default_rng(12))
        self.assertTrue(np.all((a>=0)&(a<=.15)&(b>=1)&(b<=3)))
        self.assertAlmostEqual(a.mean(),(.15**2/2+5*.15**3/3)/(.15+2.5*.15**2),delta=.0004)
        self.assertAlmostEqual(b.mean(),(4+26/3)/6,delta=.004)
    def test_default_source_does_not_load_full_data(self):
        with patch('eventcalc.production.dark_photon.ProductionModel',side_effect=AssertionError('full data accessed')):
            scan=RuntimeModelScan(config());a=scan._collision_source(1.8)
            self.assertIs(a,scan._collision_source(1.8))
            self.assertEqual(a.metadata['origin_convention'],'point source at target front face')
    def test_model_mass_and_absolute_normalization(self):
        for model,delta in [('Dark-photons',.1),('Inelastic-DM',.1),('Inelastic-DM',.05)]:
            with self.subTest(model=model,delta=delta):
                c=config(model,idm_delta=delta);m=c.masses[0]
                source=CompactSource(c,m,ExperimentCard.ship());sample=source.sampler(1.,'attempted')
                self.assertAlmostEqual(sample.integral,1.,places=10)
                out=sample.sample(10000,np.random.default_rng(4),np.random.default_rng(5),4900022)
                np.testing.assert_allclose(out.momentum[:,3]**2-np.sum(out.momentum[:,:3]**2,axis=1),source.mass**2,rtol=1e-8,atol=1e-9)
                self.assertTrue(np.all(out.origins==0));self.assertTrue(np.all(out.mother[:,6]>=0))
                self.assertEqual(source.mass,m*(1+delta)/3 if model=='Inelastic-DM' else m)
    def test_nonbenchmark_splitting_is_explicit(self):
        with self.assertRaisesRegex(ValueError,'Compact iDM'):config('Inelastic-DM',idm_delta=.2)
        self.assertEqual(config('Inelastic-DM',idm_delta=.2,production_backend='correlated').production_backend,'correlated')
    def test_momenta_and_vertices_do_not_depend_on_batch_size(self):
        source=CompactSource(config(),1.8,ExperimentCard.ship())
        sampler=source.sampler(1.,'fiducial')
        whole=sampler.sample(1000,np.random.default_rng(12),np.random.default_rng(13),4900022)
        parent=np.random.default_rng(12);vertex=np.random.default_rng(13)
        parts=[sampler.sample(n,parent,vertex,4900022) for n in (101,7,892)]
        np.testing.assert_array_equal(whole.mother,np.concatenate([p.mother for p in parts]))
        np.testing.assert_array_equal(whole.categories,np.concatenate([p.categories for p in parts]))
    def test_missing_dataset_and_corrupted_archive(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(FileNotFoundError,'datasets fetch'):resolve_full_data(d)
            p=Path(d)/'bad.tar.gz';p.write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError,'checksum'):install_archive(p,Path(d)/'output')
    def test_short_lifetime_response_has_no_cancellation_tail(self):
        from eventcalc.sensitivity import LifetimeResponse
        distance=np.geomspace(1e-12,1e12,8192)
        weights=np.zeros((1,1,len(distance)))
        weights[0,0,4500:4510]=np.linspace(.1,1.,10)
        weights[0,0,4600:4610]=-np.linspace(.1,1.,10)
        response=LifetimeResponse(distance,weights,np.zeros((1,1)))
        self.assertEqual(float(response.evaluate([1e-8])[0,0,0]),0.)
    def test_compact_output_tag_is_parseable(self):
        from funcs.selecting_processing import parse_event_filename
        r=parse_event_filename('Dark-photons_1.8_1_combined-central_eventcalc-direct-seed1-compact-source-combined-sf8eaa5e0de16_channels-all_data.dat')
        self.assertEqual(r.full_source,'combined')
if __name__=='__main__':unittest.main()
