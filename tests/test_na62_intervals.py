import unittest
import numpy as np
from eventcalc.experiments._na62_hadrons import prepare
from eventcalc.experiments._na62_cutflow import evaluate,ORIGIN
from na62_fixtures import pack
from eventcalc.sensitivity import DecayBatch


class ResponseTests(unittest.TestCase):
    def test_merged_cells_match_direct_cutflow(self):
        en=np.sqrt(.12**2+.13957**2);m=2*en+.02
        events=[[[0,.12,0,en,.13957,211],[0,-.12,0,en,.13957,-211],[.01,0,0,.01,0,22],[-.01,0,0,.01,0,22]]]
        d,ids,v,q=pack(events);p=np.array([[.2,0,np.sqrt(100**2-m*m-.04),100.]])
        batch=DecayBatch(p,d,v,[.7],m,origins=ORIGIN[None,:],frame='rest',charges=q)
        ctaus=np.geomspace(.01,100,15);groups=np.array([1])
        for nz in [16,64,128]:
            expected=evaluate(batch,ids,groups,ctaus,pmins=(1.,),nz=nz)[0][0,:,-1,:]
            actual=prepare(batch,ids,groups,nz=nz).evaluate(ctaus)
            np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=1e-18)
            self.assertGreater(actual.max(),0)

    def test_neutral_only_empty_response(self):
        d,ids,v,q=pack([[[.15,0,0,.15,0,22],[-.15,0,0,.15,0,22]]])
        p=np.array([[.2,0,np.sqrt(100**2-.3**2-.04),100.]])
        batch=DecayBatch(p,d,v,[1.],.3,origins=ORIGIN[None,:],frame='rest',charges=q)
        result=prepare(batch,ids,[0]).evaluate([1,10])
        np.testing.assert_array_equal(result,np.zeros((2,6)))


if __name__=='__main__':unittest.main()
