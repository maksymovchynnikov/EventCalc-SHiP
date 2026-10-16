import unittest
import numpy as np
from scipy.integrate import quad
from funcs import inelasticDM as d
class CurrentTests(unittest.TestCase):
 def test_small_splitting_limit(self):
  m1=1.;m2=1.001;ma=3.;delta=m2-m1
  def f(s):return d.differential_width([s],m1,m2,ma,widths=[d.ALPHA*np.sqrt(s)/3])[0,0]
  exact=quad(f,0,delta**2,epsabs=1e-32)[0]
  limit=4*d.ALPHA*.1*delta**5/(15*np.pi*ma**4)
  self.assertLess(abs(exact/limit-1),.002)
 def test_width_quadrature(self):
  for ma in [.1,1.,5.,10.,20.,30.,40.]:
   a=d.spectrum(ma,nodes=1537)[1][-1].sum();b=d.spectrum(ma,nodes=3073)[1][-1].sum()
   self.assertLess(abs(a/b-1),.002)
 def test_angular_normalization(self):
  c=np.linspace(-1,1,20001)
  for kind in ['lepton','pipi','threepi','radiative']:
   w=d.angular_weight(kind,c,.4,20/3,22/3,.105)
   self.assertAlmostEqual(np.trapezoid(w,c)/2,1,places=7)
 def test_decay_conservation_and_endpoint(self):
  a=d.generate(20.,512,seed=7112);p=a['particles'];q=a['q'];m1=20/3;m2=22/3
  np.testing.assert_allclose(p[:,:,:4].sum(axis=1),np.tile([0,0,0,m2],(len(p),1)),atol=2e-7)
  v=p[:,1:,:4].sum(axis=1)
  np.testing.assert_allclose(v[:,3]**2-(v[:,:3]**2).sum(axis=1),q*q,atol=1e-10)
  self.assertTrue(np.all(q<=m2-m1+1e-12));self.assertTrue(np.all(p[:,0,5]==4900101))
 def test_extended_current_range(self):
  ma=40.;delta=.1;m2=ma*(1+delta)/3
  result=d.generate(ma,128,delta=delta,seed=81931)
  np.testing.assert_allclose(result['particles'][:,:,:4].sum(axis=1),np.tile([0,0,0,m2],(128,1)),atol=2e-7)
  self.assertTrue(np.all(result['q']<=ma*delta/3+1e-12))
  self.assertGreater(result['width_per_eps2'],0.)
 def test_production_shell_and_energy(self):
  ma=20.;p=np.tile([0,0,50.,np.hypot(50,ma)],(3000,1));rng=np.random.default_rng(321)
  for fl in [0.,1/3,1.]:
   chi=d.produce(p,ma,fl,rng);other=p-chi
   np.testing.assert_allclose(chi[:,3]**2-(chi[:,:3]**2).sum(axis=1),(ma*1.1/3)**2,atol=1e-9)
   np.testing.assert_allclose(other[:,3]**2-(other[:,:3]**2).sum(axis=1),(ma/3)**2,atol=1e-9)
 def test_smaller_splitting_and_rest_production(self):
  ma=20.;delta=.05;m1=ma/3;m2=m1*(1+delta)
  result=d.generate(ma,128,delta=delta,seed=7149)
  np.testing.assert_allclose(result['particles'][:,:,:4].sum(axis=1),np.tile([0,0,0,m2],(128,1)),atol=2e-7)
  self.assertTrue(np.all(result['q']<=m2-m1+1e-12))
  chi=d.produce(np.tile([0,0,0,ma],(128,1)),ma,1/3,np.random.default_rng(617),delta=delta)
  self.assertTrue(np.all(np.isfinite(chi)))
  np.testing.assert_allclose(chi[:,3]**2-(chi[:,:3]**2).sum(axis=1),m2*m2,atol=1e-10)
if __name__=='__main__':unittest.main()
