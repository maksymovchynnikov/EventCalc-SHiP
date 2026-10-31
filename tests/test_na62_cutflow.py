import unittest
import numpy as np
from scipy.integrate import quad
from eventcalc.experiments.na62 import evaluate_cutflow as evaluate,ORIGIN,hadron_pointing,fv_pass
from na62_fixtures import pack
from eventcalc.experiments.na62 import ElectronFV,project_tracks,Dipole,pointing
from eventcalc.experiments.na62 import ZMAG,FIELD,PLANE
from eventcalc.sensitivity import DecayBatch


class CutflowTests(unittest.TestCase):
    def fixture(self,photons=False):
        en=np.sqrt(.12**2+.13957**2)
        event=[[0,.12,0,en,.13957,211],[0,-.12,0,en,.13957,-211]]
        if photons:event += [[.01,0,0,.01,0,22],[-.01,0,0,.01,0,22]]
        m=sum(row[3] for row in event);d,ids,v,q=pack([event])
        p=np.array([[.2,0,np.sqrt(100**2-m*m-.04),100.]])
        return DecayBatch(p,d,v,[.7],m,origins=ORIGIN[None,:],frame='rest',charges=q),ids

    def test_cylinder_integral_and_ordering(self):
        batch,ids=self.fixture();ct=np.array([.1,1.,10.]);values,_,_=evaluate(batch,ids,np.array([0]),ct,nz=256)
        rate=batch.mass/batch.mother[0,2]/ct
        exact=.7*np.exp(-(105-ORIGIN[2])*rate)*(-np.expm1(-75*rate))
        np.testing.assert_allclose(values[0,:,0,0],exact,rtol=1e-12)
        self.assertTrue(np.all(np.diff(values,axis=2)<=1e-14))

    def test_missing_soft_photons_are_not_automatic_veto(self):
        batch,ids=self.fixture(True)
        val,phot,control=evaluate(batch,ids,np.array([1]),[1.],nz=512)
        self.assertGreater(val[0,0,-1,1],0)
        np.testing.assert_array_equal(val[:,:,7],val[:,:,6])
        self.assertGreater(phot[0,0,0],0)
        np.testing.assert_array_equal(control,0)

    def test_fv_agrees_with_existing_digitization(self):
        fv=ElectronFV.digitized(centre_xy=(0,0));verts=np.asarray(fv.vertices)
        for z in [105.,150.,156.9,157.,175.]:
            for radius in [.1,.4,.6,.8]:
                self.assertEqual(fv_pass(z,radius,0.,verts),bool(fv(np.array([z]),np.array([[radius,0]]))[0]))

    def test_pointing_uses_reconstructed_momentum(self):
        p=np.array([.2,0,100.]);z=150.;xy=ORIGIN[:2]+(z-ORIGIN[2])*p[:2]/p[2]
        self.assertTrue(hadron_pointing(*xy,z,p))
        self.assertFalse(hadron_pointing(*xy,z,np.array([.01,0,100.])))

    def test_final_against_independent_vertex_quadrature(self):
        batch,ids=self.fixture(True);(p,d,v,w,o),mass=batch.arrays()
        fv=ElectronFV.digitized(centre_xy=(0,0));mag=Dipole(*ZMAG,FIELD)
        rate=mass/p[0,2];q=batch.charges[0];tracks=np.flatnonzero(q);gammas=np.flatnonzero(ids[0]==22)
        def integrand(z):
            vertex=o[0]+(z-o[0,2])*p[0,:3]/p[0,2]
            if not fv(np.array([z]),vertex[None,:2])[0]:return 0.
            vertices=np.broadcast_to(vertex,(2,3))
            final,ok=project_tracks(vertices,d[0,tracks,:3],q[tracks],PLANE,dipole=mag)
            radii=np.linalg.norm(final,axis=1)
            if not np.all(ok & (radii>.08)&(radii<1.15)):return 0.
            straw,_=project_tracks(vertices,d[0,tracks,:3],q[tracks],183.)
            lkr,_=project_tracks(vertices,d[0,tracks,:3],q[tracks],241.495,dipole=mag)
            if np.linalg.norm(straw[0]-straw[1])<.02 or np.linalg.norm(lkr[0]-lkr[1])<.2:return 0.
            rec=d[0,tracks,:3].sum(0)
            for j in gammas:
                hit,ok=project_tracks(vertex[None,:],d[0,j,None,:3],[0.],241.495)
                r=np.linalg.norm(hit[0])
                if ok[0] and d[0,j,3]>5 and .08<r<1.15:rec+=d[0,j,:3]
            ztax,cda,ok=pointing(vertex[None,:],rec[None,:],beam_xy=ORIGIN[:2])
            if not ok[0] or ((ztax[0]-23)/23)**2+(cda[0]/.04)**2>=1:return 0.
            return .7*rate*np.exp(-(z-o[0,2])*rate)
        expected=quad(integrand,105,180,points=np.linspace(105,180,100),epsabs=1e-10,limit=1000)[0]
        actual=evaluate(batch,ids,np.array([1]),[1.],nz=4096)[0][0,0,-1,1]
        self.assertGreater(expected,0)
        self.assertLess(abs(actual/expected-1),.01)


if __name__=='__main__':unittest.main()
