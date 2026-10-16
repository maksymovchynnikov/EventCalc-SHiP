"""One-time finite-width chi-prime mass/angle/energy tables; no event generation at launch."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse, json, hashlib
import numpy as np
from .model import ProductionModel
from .export_tables import THETA, ENERGY, deposit, basis_integrals
from .geometry import THETA_SHIP

MASS=np.unique(np.r_[np.geomspace(.04,20,105),np.arange(.65,1.31,.025),np.arange(20,40.01,1.),.1,.3,.775,.78266,1.01946,1.1,1.5,2.,3.,5.,7.,10.,15.,20.])

def initialize(root,delta,power):
    global MODEL, DELTA, POWER
    MODEL=ProductionModel(root); DELTA=delta; POWER=power

def calculate(mass):
    rows=[]; mother=mass*(1+DELTA)/3
    tw=basis_integrals(THETA,0,np.pi); ew=basis_integrals(ENERGY,mother,400.)
    for category in ('primary','cascade'):
        h=np.zeros((len(THETA),len(ENERGY))); total=cone=0.
        for b in MODEL.batches(mass,model='idm',delta=DELTA,power=POWER,categories=(category,),seed=62015001):
            p=b.momenta; angle=np.arctan2(np.linalg.norm(p[:,:2],axis=1),p[:,2])
            h+=deposit(angle,p[:,3],b.weights); total+=b.weights.sum(); cone+=b.weights[angle<THETA_SHIP].sum()
        if total<0 or h.min() < -max(total,1e-300)*1e-12:
            raise ValueError(f'Nonpositive density: {mass}, {category}')
        area=tw[:,None]*ew[None,:]
        d=np.divide(h,area*total,out=np.zeros_like(h),where=area*total>0)
        if total: assert np.isclose(np.sum(d*area),1.,rtol=1e-10)
        tabcone=total*np.sum(d*basis_integrals(THETA,0,THETA_SHIP)[:,None]*ew[None,:])
        rows.append((d.astype('<f4'),total,cone,tabcone))
    return float(mass), rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--delta',type=float,required=True);p.add_argument('--workers',type=int,default=2);p.add_argument('--power',type=int,default=17)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    cache=a.output.parent/('export-cache-idm-'+str(a.delta));cache.mkdir(exist_ok=True)
    todo=[m for m in MASS if not (cache/f'{m:.12g}.npz').exists()]
    with ProcessPoolExecutor(a.workers,initializer=initialize,initargs=(a.data_root,a.delta,a.power)) as pool:
        for m,r in pool.map(calculate,todo):
            np.savez_compressed(cache/f'{m:.12g}.npz',density=np.array([x[0] for x in r]),yields=[x[1] for x in r],cone=[x[2] for x in r],table_cone=[x[3] for x in r]);print(m,flush=True)
    data=[]
    for m in MASS:
        with np.load(cache/f'{m:.12g}.npz',allow_pickle=False) as d:data.append({k:d[k] for k in d.files})
    manifest=json.loads((a.data_root/'manifest.json').read_text())
    manifest.pop('sha256',None);manifest.pop('uncertainty_package',None)
    manifest.update(model='idm',delta=a.delta,ratio=3.,alpha_D=.1,masses=len(MASS),power=a.power,
        position_marginalized=True,origin_convention='point source at target front face',
        mass_coordinate='mediator pole mass; energy and angle describe chi-prime',
        normalization='per POT and epsilon^2; density normalized above chi-prime mass',sha256={})
    for i,cat in enumerate(('primary','cascade')):
        name=f'idm-delta{a.delta:g}-{cat}.npz'
        np.savez_compressed(a.output/name,mass=MASS,theta=THETA,energy=ENERGY,
            density=np.array([d['density'][i] for d in data]),yield_per_pot_epsilon2=np.array([d['yields'][i] for d in data]))
        manifest['sha256'][name]=hashlib.sha256((a.output/name).read_bytes()).hexdigest()
    errors=np.array([np.divide(d['table_cone']-d['cone'],d['cone'],out=np.zeros(2),where=d['cone']!=0) for d in data])
    manifest['cone_relative_error_max']=float(abs(errors).max())
    (a.output/f'idm-delta{a.delta:g}-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    np.savez_compressed(cache/'validation.npz',mass=MASS,cone_relative_error=errors)
    print('complete',a.delta,manifest['cone_relative_error_max'],flush=True)
if __name__=='__main__':main()
