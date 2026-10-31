from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.integrate import quad

from eventcalc.sensitivity import DecayBatch, integrate
from eventcalc.experiments.bebc import BEBCConfig, BEBCResponse, Box, PairCuts, transverse_mass
from eventcalc.experiments.charm import CHARMConfig, CHARMResponse, Plane, daughter_intervals


def batch():
    p=np.array([[0.,0.,1.,np.sqrt(5)]])
    d=np.array([[[1.,0.,.5,np.sqrt(1.25)],[-1.,0.,.5,np.sqrt(1.25)]]])
    return DecayBatch(p,d,np.ones((1,2),bool),np.array([-.5]),2.,
        origins=np.zeros((1,3)),charges=np.array([[-1.,1.]]),pdg_ids=np.array([[11,-11]]))


class BEBCTests(unittest.TestCase):
    def config(self,**kwargs):
        return BEBCConfig(lower=(-1,-1,10),upper=(1,1,12),**kwargs)

    def test_box_axes_and_translation(self):
        box=Box((0,0,10),(2,2,12))
        a,b=box.intervals([[1,1,0],[3,1,0],[1,1,11],[1,1,13],[1,1,0]],
                          [[0,0,1],[0,0,1],[0,0,1],[0,0,-1],[0,0,0]])
        np.testing.assert_allclose(a,[10,0,0,1,0]);np.testing.assert_allclose(b,[12,0,1,3,0])
        c=self.config();sample=batch();ct=[1,20,1000]
        expected=integrate([sample],ct,experiment=BEBCResponse(c))
        shift=np.array([3,-4,23.07]);sample.origins+=shift
        moved=replace(c,lower=tuple(np.array(c.lower)+shift),upper=tuple(np.array(c.upper)+shift))
        np.testing.assert_allclose(integrate([sample],ct,experiment=BEBCResponse(moved)),expected)

    def test_signed_yield_efficiency_and_independent_integral(self):
        b=batch();c=self.config(efficiency=.96)
        actual=integrate([b],[20],experiment=BEBCResponse(c),exposure=2,production_scales=[3])[0]
        expected=-.5*quad(lambda z:2/20*np.exp(-z*2/20),10,12)[0]*6
        np.testing.assert_allclose(actual,[expected,expected*.96,expected*.96,0,0,expected*.96],rtol=1e-13)
        approximate=integrate([b],[20],experiment=BEBCResponse(c),response_nodes=65536)
        np.testing.assert_allclose(approximate*6,actual[None,:],rtol=2e-5,atol=1e-14)

    def test_no_downstream_aperture_or_forward_daughter_cut(self):
        b=batch();b.daughters=np.array([[[0,0,-1,1],[0,0,1,1]]],float);b.frame='rest'
        response=BEBCResponse(self.config(daughter_energy_min_GeV=0))
        actual=response.prepare(b).evaluate([20])[0]
        self.assertNotEqual(actual[1],0);self.assertEqual(actual[0],actual[1])

    def test_pair_energy_and_track_momentum_are_independent(self):
        d=[[[0,0,.6,.7],[0,0,.6,.7],[0,0,20,21]]];q=[[1,-1,0]];v=[[True,True,False]]
        self.assertTrue(PairCuts(pair_energy_min=1).select(d,q,v)[0])
        self.assertFalse(PairCuts(track_p_min=1).select(d,q,v)[0])
        self.assertFalse(PairCuts(pair_energy_min=1.4).select(d,q,v)[0])
        self.assertFalse(PairCuts().select(d,q,[[True,True,True]])[0])

    def test_explicit_transverse_mass_cut(self):
        self.assertAlmostEqual(transverse_mass([[3,0,0,5]],[0,0,1])[0],8.)
        with self.assertRaises(ValueError):self.config(transverse_mass_max_GeV=1.)
        config=self.config(transverse_mass_max_GeV=1.,transverse_axis=(0,0,1))
        values=BEBCResponse(config).prepare(batch()).evaluate([20])[0]
        self.assertNotEqual(values[0],0);np.testing.assert_array_equal(values[1:],0)

    def test_threshold_and_species_selection(self):
        b=batch();ct=[20]
        values=BEBCResponse(self.config(daughter_energy_min_GeV=np.sqrt(1.25))).prepare(b).evaluate(ct)[0]
        self.assertNotEqual(values[0],0);np.testing.assert_array_equal(values[1:],0)
        b.pdg_ids=np.array([[211,-211]])
        values=BEBCResponse(self.config()).prepare(b).evaluate(ct)[0]
        self.assertNotEqual(values[1],0);np.testing.assert_array_equal(values[2:],0)


class CHARMTests(unittest.TestCase):
    def test_off_axis_volume(self):
        c=CHARMConfig()
        a,b=c.intervals([[0,0,0],[0,0,0]],[[0,0,1],[5,0,500]])
        self.assertEqual(b[0],0);self.assertGreater(b[1],a[1])

    def test_plane_intervals_against_independent_vertex_integral(self):
        box=Box((-1,-1,10),(1,1,12));plane=Plane(15,(-.4,-1),(.4,1))
        a,b=daughter_intervals(box,[plane],[[0,0,0]],[[0,0,1]],[[[.1,0,1],[-.1,0,1]]],[[True,True]])
        np.testing.assert_allclose([a[0],b[0]],[11,12])
        expected=quad(lambda z:np.exp(-z/20)/20*(abs((15-z)*.1)<=.4),10,12,points=[11])[0]
        self.assertAlmostEqual(float(np.exp(-a[0]/20)*-np.expm1(-(b[0]-a[0])/20)),expected,places=13)
        c,d=daughter_intervals(box,[plane,Plane(16,(-.45,-1),(.45,1))],
            [[0,0,0]],[[0,0,1]],[[[.1,0,1]]],[[True]])
        np.testing.assert_allclose([c[0],d[0]],[11.5,12])

    def test_response_transport_and_translation(self):
        b=batch();c=CHARMConfig(lower=(-1,-1,10),upper=(1,1,12),
            planes=(Plane(15,(-8,-1),(8,1)),),efficiency=.6)
        actual=CHARMResponse(c).prepare(b).evaluate([20])[0]
        lo=11.;hi=12.;expected=-.5*np.exp(-lo*2/20)*-np.expm1(-(hi-lo)*2/20)*.6
        np.testing.assert_allclose(actual[1:],[expected,expected,0,0,expected],rtol=1e-13)
        shift=np.array([3,-4,23.07]);b.origins+=shift
        moved=replace(c,lower=tuple(np.array(c.lower)+shift),upper=tuple(np.array(c.upper)+shift),
            planes=tuple(Plane(p.z+shift[2],tuple(np.array(p.lower)+shift[:2]),tuple(np.array(p.upper)+shift[:2])) for p in c.planes))
        np.testing.assert_allclose(CHARMResponse(moved).prepare(b).evaluate([20])[0],actual)
        blocked=replace(c,planes=(Plane(15,(-.1,-1),(.1,1)),))
        self.assertEqual(CHARMResponse(blocked).prepare(batch()).evaluate([20])[0,1],0)

    def test_invisible_daughter_does_not_need_plane_hit(self):
        box=Box((-1,-1,10),(1,1,12));plane=Plane(15,(-.4,-1),(.4,1))
        d=[[[.1,0,1],[50,0,-1]]]
        a,b=daughter_intervals(box,[plane],[[0,0,0]],[[0,0,1]],d,[[True,False]])
        self.assertGreater(b[0],a[0])
        self.assertEqual(daughter_intervals(box,[plane],[[0,0,0]],[[0,0,1]],d,[[True,True]])[1][0],0)


class HistoricalScanTests(unittest.TestCase):
    def test_complete_invisible_daughter_is_retained_but_not_selected(self):
        d=np.array([[[1.,0.,.5,np.sqrt(1.25)],[-1.,0.,.5,np.sqrt(1.25)],
                     [0.,0.,-.2,.2]]])
        p=d.sum(axis=1);mass=float(np.sqrt(p[0,3]**2-np.dot(p[0,:3],p[0,:3])))
        b=DecayBatch(p,d,np.array([[True,True,False]]),np.array([1.]),mass,
            origins=np.zeros((1,3)),charges=np.array([[-1.,1.,0.]]),pdg_ids=np.array([[11,-11,4900101]]))
        for response in (BEBCResponse(BEBCConfig(lower=(-1,-1,10),upper=(1,1,12))),
                         CHARMResponse(CHARMConfig(lower=(-1,-1,10),upper=(1,1,12),
                             planes=(Plane(15,(-8,-1),(8,1)),)))):
            self.assertGreater(response.prepare(b).evaluate([20])[0,-1],0)
            extra_visible=replace(b,visible=np.ones((1,3),bool))
            values=response.prepare(extra_visible).evaluate([20])[0]
            self.assertGreater(values[0],0);np.testing.assert_array_equal(values[1:],0)

    def test_scan_cards_and_resolved_metadata(self):
        from eventcalc.scan_sensitivity import run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);b=batch()
            for i in range(2):
                np.savez(root/f'r{i}.npz',mother=b.mother,daughters=b.daughters,visible=b.visible,
                    weights=b.weights,mass=b.mass,origins=b.origins,charges=b.charges,pdg_ids=b.pdg_ids)
            for name in ('bebc','charm'):
                c=dict(lower=[-1,-1,10],upper=[1,1,12])
                if name=='charm':c['planes']=[dict(z=15,lower=[-8,-1],upper=[8,1])]
                card=dict(experiment=dict(name=name,config=c),parameters=[1.,2.],ctau_m=[10.,20.],
                    production_scales=[1.,1.],exposure=1.,replicas=2,response_nodes=None,
                    source=dict(kind='npz',replicas=['r0.npz','r1.npz']))
                a,meta=run(card,base=root)
                self.assertEqual(a['events'].shape,(2,6));self.assertIn('resolved_experiment_config',meta)
                self.assertEqual(meta['resolved_experiment_config']['efficiency'],1.)
                json.dumps(meta)

    def test_invalid_configs_and_event_contract(self):
        for response in (BEBCResponse(),CHARMResponse()):
            b=batch();b.pdg_ids=None
            with self.assertRaises(ValueError):response.prepare(b)
            b=batch();b.mother=np.array([[0,0,0,2.]])
            b.daughters=np.array([[[0,0,-1,1],[0,0,1,1]]],float)
            with self.assertRaises(ValueError):response.prepare(b)
        for cls in (BEBCConfig,CHARMConfig):
            with self.assertRaises(ValueError):cls(efficiency=1.1)
            with self.assertRaises(ValueError):cls(lower=(0,0,2),upper=(0,1,3))
        with self.assertRaises(ValueError):CHARMConfig(planes=[dict(z=500,lower=[0,0],upper=[1,1])])
