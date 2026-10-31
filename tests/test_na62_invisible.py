import unittest
import numpy as np
from eventcalc.sensitivity import DecayBatch
from eventcalc.experiments.na62 import (
    ElectronFV, ORIGIN, FIELD, ZMAG, PLANE, pointing, signal_region, prepare_leptons)
from eventcalc.transport import dipole_displacement


class InvisiblePairTests(unittest.TestCase):
    def check_invisible(self,channel):
        n=24;mass=2.2;missing_px=np.linspace(.001,.08,n)
        d=np.zeros((n,3,4))
        d[:,0,0]=1-missing_px/2;d[:,0,3]=d[:,0,0]
        d[:,1,0]=-1-missing_px/2;d[:,1,3]=-d[:,1,0]
        d[:,2,0]=missing_px;d[:,2,3]=.2
        p=np.tile([.4,0.,200.,np.sqrt(200**2+.4**2+mass**2)],(n,1))
        b=DecayBatch(p,d,np.tile([True,True,False],(n,1)),np.full(n,1/n),mass,
            np.tile(ORIGIN,(n,1)),'rest',np.tile([-1.,1.,0.],(n,1)),
            np.tile([11 if channel=='ee' else 13,-11 if channel=='ee' else -13,4900101],(n,1)))
        prepared=prepare_leptons(b,b.pdg_ids);ctaus=np.array([.2,2.,20.])
        actual=prepared.evaluate(ctaus)
        (p,d,v,w,o),mass=b.arrays()
        nz=65536;dz=75/nz;z=105+(np.arange(nz)+.5)*dz
        expected=np.zeros((3,2));fv=ElectronFV.digitized(centre_xy=(0,0))
        for i in range(len(p)):
            vertex=o[i]+(z-o[i,2])[:,None]*p[i,:3]/p[i,2]
            pair=d[i,:2,:3].sum(axis=0)
            zca,cda,valid=pointing(vertex,np.broadcast_to(pair,vertex.shape),beam_xy=ORIGIN[:2])
            base=(np.linalg.norm(vertex[:,:2],axis=1)<1)&fv(z,vertex[:,:2])&signal_region(zca,cda,channel=channel)
            base &= np.all(np.linalg.norm(d[i,:2,:3],axis=1)>10.)
            detector=base.copy()
            for j in range(2):
                dx,dy,ok=dipole_displacement(d[i,j,:3],b.charges[i,j],*ZMAG,FIELD,PLANE)
                if not ok:detector[:]=False;continue
                hit=vertex[:,:2]+(PLANE-z)[:,None]*d[i,j,:2]/d[i,j,2]+[dx,dy]
                radius=np.linalg.norm(hit,axis=1)
                detector &= (radius>.08)&(radius<1.15)
            rate=mass/p[i,2]/ctaus
            probability=np.exp(-(z[:,None]-o[i,2])*rate)*(-np.expm1(-dz*rate))*w[i]
            # Midpoint density paired with exact cell width probability has O(dz) error.
            expected[:,0]+=probability[base].sum(axis=0)
            expected[:,1]+=probability[detector].sum(axis=0)
        self.assertGreater(expected[:,0].max(),0)
        np.testing.assert_allclose(actual[:,[1,2] if channel=='ee' else [4,5]],expected,rtol=2e-3,atol=2e-7)

    def test_electron_pointing_intervals_against_vertex_quadrature(self):
        self.check_invisible('ee')

    def test_muon_pointing_intervals_against_vertex_quadrature(self):
        self.check_invisible('mumu')


if __name__=='__main__':unittest.main()
