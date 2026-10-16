"""Physical normalization, trajectory and launcher contracts for full sources."""
import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from production_model.model import ProductionBatch
from production_model.runtime import CollisionSource, trajectory_intervals, uses_full_source
from runtime_generator import ExperimentCard, RuntimeModelScan
from funcs.simulation_config import config_from_mapping, MODEL_SPECS
from funcs import LLP_selection


def config(model="Dark-photons", **changes):
    values = dict(model=model, events=32, masses=[20 if model == "Inelastic-DM" else 1.8],
                  c_taus=[1], seed=7, production_power=10,
                  exhad_mode="auto" if model == "Inelastic-DM" else "off")
    values.update(changes)
    return config_from_mapping(values)


def synthetic_source():
    # Two crossing trajectories share p but start at different depths; a third
    # misses entirely. A negative component must survive importance sampling.
    momentum = np.array([[0,0,10,np.sqrt(101)]]*4, float)
    origins = np.array([[.1,0,1.],[-.2,0,2.],[100,0,0],[0,0,2.]], float)
    b = ProductionBatch(momentum, origins, np.array([2.,3.,4.,-.5]), 1., "primary", "brem")
    return CollisionSource([b], ExperimentCard.ship())


class PhysicalSourceTest(unittest.TestCase):
    def test_vertices_start_at_the_correlated_origin_and_decay_distance(self):
        source = synthetic_source()
        sample = source.sampler(2., "fiducial").sample(5000, np.random.default_rng(3), np.random.default_rng(4), 4900022)
        np.testing.assert_allclose(sample.mother[:,:4], sample.momentum)
        np.testing.assert_allclose(sample.mother[:,7:9], sample.origins[:,:2])
        self.assertTrue(ExperimentCard.ship().contains(*sample.mother[:,7:10].T).all())
        expected = np.exp(-(32-sample.origins[:,2])/20)*(-np.expm1(-50/20))
        np.testing.assert_allclose(abs(sample.mother[:,6]), expected, rtol=1e-14)
        # Its true proper flight distance differs from target-front propagation.
        self.assertGreater(np.max(abs(expected-np.exp(-32/20)*(-np.expm1(-50/20)))), .01)
        for origin in source.origins[source.valid]:
            self.assertTrue(np.any(np.all(sample.origins==origin, axis=1)))
        self.assertTrue(np.any(sample.mother[:,6]<0))

    def test_signed_normalization_and_attempted_fiducial_agree(self):
        source = synthetic_source(); ctau=2.
        p = np.exp(-source.low/(ctau*10))*(-np.expm1(-(source.high-source.low)/(ctau*10)))
        expected = np.dot(source.weights, p)
        for mode in ("attempted", "fiducial"):
            sampler=source.sampler(ctau,mode)
            sample=sampler.sample(300000,np.random.default_rng(5),np.random.default_rng(6),4900022)
            estimate=source.total_yield*sampler.integral*sample.mother[:,6].mean()
            error=source.total_yield*sampler.integral*sample.mother[:,6].std(ddof=1)/np.sqrt(len(sample.mother))
            self.assertLess(abs(estimate-expected),5*error)
            self.assertEqual(np.count_nonzero(sample.mother[:,6]==0),np.count_nonzero(~sample.inside_volume))

    def test_side_entry_backwards_ray_and_zero_momentum(self):
        experiment=ExperimentCard("box",10,20,2,2,2,2)
        # First enters x=-1 at z=15; second enters from the downstream plane.
        p=np.array([[.2,0,1,2],[0,0,-1,2],[0,0,0,1]],float)
        o=np.array([[-4,0,0],[0,0,30],[0,0,0]],float)
        direction,low,high,valid=trajectory_intervals(p,o,experiment)
        self.assertEqual(valid.tolist(),[True,True,False])
        np.testing.assert_allclose(o[:2]+direction[:2]*low[:2,None],[[-1,0,15],[0,0,20]])
        np.testing.assert_allclose(o[:2]+direction[:2]*high[:2,None],[[0,0,20],[0,0,10]])

    def test_hepmc_reads_back_with_correlated_origin(self):
        import importlib.util
        if importlib.util.find_spec("pyhepmc") is None:
            self.skipTest("optional pyhepmc is not installed")
        import pyhepmc
        import tempfile
        from hepmc_export import HepMC3Writer
        from runtime_generator import EventRecord
        mother=np.array([0,0,10,np.sqrt(101),1,4900102,.2,.1,.2,32.])
        event=EventRecord(0,mother,np.array([[0,0,10,np.sqrt(101),1,4900101]]),
                          "test",True,.2,production_origin=np.array([.1,.2,1.]))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"event.hepmc3"
            with HepMC3Writer(path) as writer:writer.write(event)
            with pyhepmc.open(path) as reader:events=list(reader)
        self.assertEqual(len(events),1)
        out=events[0]
        parent=next(p for p in out.particles if p.pid==4900102)
        self.assertAlmostEqual(parent.end_vertex.position.t,31000*np.sqrt(101)/10)
        for axis,value in zip("xyz",[100.,200.,1000.]):
            attribute=out.attributes[f"eventcalc_production_{axis}_mm"]
            self.assertAlmostEqual(attribute.astype(float),value)
        self.assertEqual(out.weights,[.2])

    def test_hepmc_clock_starts_at_collision(self):
        from hepmc_export import HepMC3Writer
        mother=np.array([0,0,10,np.sqrt(101),1,4900022,1,0,0,32.])
        self.assertAlmostEqual(HepMC3Writer._decay_time_mm(mother,[0,0,2]),
                               30000*np.sqrt(101)/10)

    def test_source_reused_across_lifetimes(self):
        scan=RuntimeModelScan(config())
        source=synthetic_source()
        with patch.object(CollisionSource,"build",return_value=source) as build, \
             patch("production_model.ProductionModel"):
            self.assertIs(scan._collision_source(1.8),scan._collision_source(1.8))
            self.assertIs(scan._collision_source(1.8).sampler(1.,"fiducial").source,
                          scan._collision_source(1.8).sampler(100.,"fiducial").source)
        self.assertEqual(build.call_count,1)


class ModelConfigurationTest(unittest.TestCase):
    def test_explicit_legacy_route_and_default_full_source(self):
        self.assertTrue(uses_full_source(config()))
        self.assertFalse(uses_full_source(config(dp_production_mode="legacy-primary",uncertainty="upper")))
        self.assertTrue(uses_full_source(config("Inelastic-DM",idm_delta=.05)))
        for change in (dict(idm_delta=1),dict(idm_delta=.0001),dict(exhad_mode="off"),
                       dict(dp_production_mode="legacy-primary"),dict(production_power=3)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                config("Inelastic-DM",**change)

    def test_analysis_recognizes_full_source_output_tags(self):
        from funcs.selecting_processing import parse_event_filename
        for name,base in (("Dark-photons","combined-central_eventcalc-direct"),
                          ("Inelastic-DM","exhad-inelastic-current")):
            tag=base+"-seed1-full-source-combined-p15-sf8eaa5e0de16"
            if name=="Inelastic-DM":tag+="-delta0.05"
            record=parse_event_filename(f"{name}_20_1_{tag}_channels-all_data.dat")
            self.assertEqual(record.full_source,"combined")
            self.assertEqual(record.production_power,15)
            self.assertEqual(record.seed,1)
            self.assertEqual(record.idm_delta,.05 if name=="Inelastic-DM" else None)

    def test_menu_selects_idm_and_its_parameters(self):
        installed=[s for s in MODEL_SPECS if (ROOT/'Distributions'/s.directory).is_dir()]
        index=next(i+1 for i,s in enumerate(installed) if s.name=="Inelastic-DM")
        with patch.object(LLP_selection,"_interactive_config",None), \
             patch("builtins.input",side_effect=["32",str(index),"3",".05"]), \
             contextlib.redirect_stdout(io.StringIO()):
            values=LLP_selection.configure_interactively()
        self.assertEqual(values['particle_selection']['LLP_name'],"Inelastic-DM")
        self.assertEqual(values['idm_delta'],.05)
        self.assertEqual(values['dp_production_mode'],"combined")

    def test_idm_mass_and_coupling_roundtrip(self):
        cfg=config("Inelastic-DM", idm_delta=.05, c_taus=None, couplings_squared=[1e-6])
        from funcs.inelastic_llp import InelasticLLP
        llp=InelasticLLP(cfg);llp.set_mass(20);llp.compute_mass_dependent_properties()
        self.assertEqual(llp.mass,7.)
        self.assertEqual(llp.mediator_mass,20.)
        self.assertEqual(cfg.resolve_lifetimes(llp).c_taus[0][0],llp.c_tau_int/1e-6)


if __name__ == '__main__':
    unittest.main()
