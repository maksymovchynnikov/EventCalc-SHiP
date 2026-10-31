from dataclasses import dataclass, replace
import unittest
import numpy as np
from eventcalc.sensitivity import PreparedAcceptance, Selection, integrate, prepare
from eventcalc.experiments.ship import ShipResponse
from test_sensitivity import sample


class ExperimentResponseTests(unittest.TestCase):
    def test_ship_compatibility(self):
        b=sample()
        legacy=prepare(b);extracted=ShipResponse().prepare(b)
        for key in ('lower','upper','weights','selected','event_index'):
            np.testing.assert_array_equal(getattr(legacy,key),getattr(extracted,key))
        np.testing.assert_array_equal(integrate([b],[1.,10.]),integrate([b],[1.,10.],experiment=ShipResponse()))

    def test_generic_disjoint_intervals_signed_scaling_and_chunking(self):
        @dataclass(frozen=True)
        class Label:
            name: str
        class Response:
            geometry=None
            selections=(Label('accepted'),)
            def prepare(self,batch):
                w=np.asarray(batch,float)
                return PreparedAcceptance(np.tile([1.,3.],len(w)),np.tile([2.,4.],len(w)),
                    np.repeat(w,2),np.zeros(2*len(w),np.int64),self.selections,
                    event_index=np.repeat(np.arange(len(w)),2))
        ct=np.array([.1,1,10]);r=Response()
        expected=.5*(np.exp(-1/ct)-np.exp(-2/ct)+np.exp(-3/ct)-np.exp(-4/ct))
        a=integrate([[.7,-.2]],ct,experiment=r)
        np.testing.assert_allclose(a[:,0],expected,rtol=1e-13)
        np.testing.assert_allclose(a,integrate([[.7],[-.2]],ct,experiment=r),rtol=1e-13)
        np.testing.assert_allclose(a*6,integrate([[.7,-.2]],ct,experiment=r,exposure=2,production_scales=[3]*3))
        np.testing.assert_allclose(integrate([[1.,-1.]],ct,experiment=r),0,atol=1e-15)
        np.testing.assert_array_equal(integrate([[]],ct,experiment=r),np.zeros((3,1)))
        approximate=integrate([[.7,-.2]],ct,experiment=r,response_nodes=65536)
        np.testing.assert_allclose(approximate,a,rtol=2e-5,atol=1e-14)

    def test_na62_requires_pdg_and_charges(self):
        from eventcalc.experiments.na62 import NA62Response
        with self.assertRaises(ValueError):NA62Response().prepare(sample())
