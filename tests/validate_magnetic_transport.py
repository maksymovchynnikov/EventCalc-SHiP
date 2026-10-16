import sys,time,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eventcalc.sensitivity import Geometry,Selection,DecayBatch,prepare,dipole_displacement,charges_from_pdg

def run():
 g=Geometry();rng=np.random.default_rng(12015);errors=[]
 # Independent Lorentz-equation integration in path length, terminating at z_max.
 for charge in (-1.,1.):
  for p in ([.2,.3,1.],[-.4,.8,4.],[0.,0.,100.]):
   p=np.array(p);norm=np.linalg.norm(p);k=.299792458*charge*g.field_integral/(g.magnet_z_max-g.magnet_z_min)
   def rhs(t,a):
    return [a[3]/norm,a[4]/norm,a[5]/norm,-k*a[5]/norm,0.,k*a[3]/norm]
   def end(t,a):return a[2]-(g.magnet_z_max-g.magnet_z_min)
   end.terminal=True;end.direction=1
   r=solve_ivp(rhs,(0,100),np.r_[np.zeros(3),p],events=end,rtol=1e-11,atol=1e-12)
   assert len(r.y_events[0])==1
   a=r.y_events[0][0];drift=g.aperture_z-g.magnet_z_max
   expected=a[:2]+drift*a[3:5]/a[5]-(g.aperture_z-g.magnet_z_min)*p[:2]/p[2]
   x,y,ok=dipole_displacement(p,charge,g.magnet_z_min,g.magnet_z_max,g.field_integral,g.aperture_z)
   assert ok;errors.append(float(np.max(abs(expected-[x,y]))))
 assert max(errors)<1e-8,errors
 assert dipole_displacement(np.array([0.,0.,.01]),1.,86.,91.,.5,97.32)[2]==False
 assert dipole_displacement(np.array([.1,.2,1.]),0.,86.,91.,.5,97.32)==(0.,0.,True)
 assert np.array_equal(charges_from_pdg([11,-11,211,-211,22,2212,2112]),[-1,1,1,-1,0,1,0])
 # Isotropic two-body rest decays: independent dense decay-position quadrature.
 n=400;mass=1.;pz=rng.uniform(20,100,n);mother=np.column_stack([np.zeros((n,2)),pz,np.hypot(pz,mass)])
 u=rng.normal(size=(n,3));u/=np.linalg.norm(u,axis=1)[:,None]
 rest=np.zeros((n,2,4));rest[:,0,:3]=u*.5;rest[:,1,:3]=-u*.5;rest[:,:,3]=.5
 charges=np.tile([1.,-1.],(n,1));visible=np.ones((n,2),bool);weights=np.full(n,1/n)
 batch=DecayBatch(mother,rest,visible,weights,mass,frame='rest',charges=charges)
 selections=(Selection('two'),Selection('all',all_visible=True))
 prepared=prepare(batch,geometry=g,selections=selections);exact=prepared.evaluate([1.,10.,100.])
 arrays,_=batch.arrays();lab=arrays[1];zs=np.linspace(32,82,30001);dz=zs[1]-zs[0];quad=np.zeros(3)
 for i in range(n):
  accepted=np.ones(len(zs),bool)
  for j in range(2):
   d=lab[i,j];dx,dy,forward=dipole_displacement(d[:3],charges[i,j],86,91,.5,97.32)
   accepted &= forward & (d[3]>1) & (np.abs((97.32-zs)*d[0]/d[2]+dx)<g.aperture_x/2) & (np.abs((97.32-zs)*d[1]/d[2]+dy)<g.aperture_y/2)
  for k,ct in enumerate((1.,10.,100.)):
   ell=pz[i]*ct;quad[k]+=weights[i]*np.trapezoid(accepted*np.exp(-zs/ell)/ell,zs)
 assert np.allclose(exact[:,0],quad,rtol=2e-4,atol=2e-6),(exact,quad)
 assert np.array_equal(exact[:,0],exact[:,1])
 # Charge reversal is a mirror symmetry for an on-axis, symmetric aperture.
 reflected=rest.copy();reflected[:,:,0]*=-1
 refl=prepare(DecayBatch(mother,reflected,visible,weights,mass,frame='rest',charges=-charges),geometry=g,selections=selections).evaluate([1,10,100])
 assert np.allclose(exact,refl,atol=1e-13)
 # One missed daughter in a four-body decay must distinguish >=2 from ALL.
 four=np.tile([[[0,0,5,5],[0,0,5,5],[1,0,1,2**.5],[-1,0,1,2**.5]]],(1,1,1)).astype(float)
 mom=four.sum(1);m=np.sqrt(mom[0,3]**2-(mom[0,:3]**2).sum())
 mixed=prepare(DecayBatch(mom,four,np.ones((1,4),bool),np.ones(1),m,charges=np.zeros((1,4))),geometry=g,selections=selections).evaluate([100])
 assert mixed[0,0]>0 and mixed[0,1]==0,mixed
 # Benchmark warm kernels, 100k fully validated decays, both ECAL selections.
 n=100000;repeat=lambda a:np.resize(a,(n,)+a.shape[1:])
 big=DecayBatch(repeat(mother),repeat(rest),repeat(visible),np.full(n,1/n),mass,frame='rest',charges=repeat(charges))
 times=[]
 for _ in range(3):
  t=time.perf_counter();p=prepare(big,geometry=g,selections=selections);t1=time.perf_counter();r=p.response();value=r.evaluate(np.geomspace(.001,1e5,241));t2=time.perf_counter();times.append({'prepare_s':t1-t,'response_and_241_lifetimes_s':t2-t1,'total_s':t2-t})
 # Interpolated response accuracy against exact event integral over broad lifetimes.
 c=np.geomspace(.01,1e4,25);a=p.evaluate(c);b=r.evaluate(c)[0];mask=a>1e-12
 discrepancy=float(np.max(abs(a[mask]-b[mask])/a[mask]));assert discrepancy<.0001,discrepancy
 report={'passed':True,'lorentz_ode_max_error_m':max(errors),'quadrature_relative_error':float(np.max(abs(exact[:,0]-quad)/quad)),'response_max_relative_error':discrepancy,'events':n,'selections':2,'lifetimes':241,'warm_timings':times,'geometry':g.__dict__}
 Path('/tmp/eventcalc-magnetic-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':run()
