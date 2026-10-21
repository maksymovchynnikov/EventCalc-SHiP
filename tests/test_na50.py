"""NA50 identity/fit contracts and stable signed lifetime integration."""
import hashlib,json,unittest
from types import SimpleNamespace
import numpy as np
from eventcalc.production.dark_photon.calibration import DATA,NA50Calibration
from eventcalc.sensitivity import LifetimeResponse


class NA50Tests(unittest.TestCase):
    def test_fit_and_compact_proton_mask(self):
        md=json.loads((DATA/'manifest.json').read_text())
        for name,sha in md['sha256'].items():
            self.assertEqual(hashlib.sha256((DATA/name).read_bytes()).hexdigest(),sha)
        fit=json.loads((DATA/'fit.json').read_text())
        c=NA50Calibration.__new__(NA50Calibration)
        c.rows={tuple(row['label']):row for row in fit['calibrations']}
        labels=[row['label'] for row in fit['calibrations']]
        values=c.coefficients(labels)
        self.assertEqual(values.shape,(3,243))
        self.assertAlmostEqual(values[0,0],1.2347030711343996,places=13)
        self.assertTrue(np.all(values[1]<=values[0]) and np.all(values[0]<=values[2]))
        np.testing.assert_equal(c.coefficients(labels[::-1]),values[:,::-1])
        with self.assertRaises(ValueError):c.coefficients(labels[:-1])
        with np.load(DATA/'proton_events.npz') as z:mask=z['proton']
        self.assertEqual(mask.dtype,np.bool_);self.assertEqual(len(mask),655535)
        self.assertEqual(int(mask.sum()),167314)

    def test_proton_selection_preserves_event_identity(self):
        c=NA50Calibration.__new__(NA50Calibration)
        c.proton=np.array([False,True,False,True,False])
        c.ids={('primary',0):np.array([4,1,2]),('primary',1):np.array([3,0])}
        inp=SimpleNamespace(current_dy_pool=np.array([1,0,1,0]),current_dy_row=np.array([0,2,1,1]))
        np.testing.assert_equal(c.sampled_protons(inp,'primary'),[True,False,False,True])

    def test_signed_response_matches_direct_intervals(self):
        d=np.exp(np.linspace(-12.,24.,8192));ct=np.geomspace(1e-5,1e16,400)
        h=np.zeros((3,1,len(d)));direct=np.zeros((3,len(ct)))
        rng=np.random.default_rng(7139)
        for row in range(3):
            lo=rng.integers(2400,3000,600);hi=lo+rng.integers(1,1000,600);w=np.exp(rng.uniform(-15,0,len(lo)))
            if row==2:w[::3]*=-.3
            np.add.at(h[row,0],lo,w);np.add.at(h[row,0],hi,-w)
            direct[row]=np.sum(w[:,None]*np.exp(-d[lo,None]/ct)*(-np.expm1(-(d[hi,None]-d[lo,None])/ct)),axis=0)
        result=LifetimeResponse(d,h,np.zeros((3,1))).evaluate(ct)[:,:,0]
        np.testing.assert_allclose(result,direct,rtol=2e-11,atol=1e-290)
        h[:]=0;h[0,0,2400]=-1.;h[0,0,2600]=1.
        expected=-np.exp(-d[2400]/ct)*(-np.expm1(-(d[2600]-d[2400])/ct))
        np.testing.assert_allclose(LifetimeResponse(d,h,np.zeros((3,1))).evaluate(ct)[0,:,0],expected,rtol=2e-12,atol=1e-290)

    def test_zero_lower_endpoint(self):
        d=np.array([.01,.1,1.,10.]);h=np.array([[[0.,0.,0.,-2.]]]);ct=np.geomspace(1e-12,1e15,100)
        actual=LifetimeResponse(d,h,np.array([[2.]])).evaluate(ct)[0,:,0]
        np.testing.assert_allclose(actual,2*(-np.expm1(-10/ct)),rtol=1e-14)

if __name__=='__main__':unittest.main()
