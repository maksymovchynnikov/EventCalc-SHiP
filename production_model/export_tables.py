"""Export normalized compact tables from the calibrated weighted source.

This is a one-time compatibility export, not part of a lifetime scan. Direct
production/acceptance uses the source batches with their collision positions.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,hashlib
import numpy as np
from .model import ProductionModel
from .geometry import THETA_SHIP

MASS=np.unique(np.r_[np.geomspace(.01,20,91),np.arange(.755,.811,.001),np.arange(1.,1.041,.001),.03,.1,.1349766,.3,.547862,.6,.6476734,.85,.95778,1.1,1.5,2.,3.,5.,7.,10.,15.,20.])
THETA=np.unique(np.r_[0,np.geomspace(1e-6,np.pi,112),THETA_SHIP])
ENERGY=np.unique(np.r_[.005,np.linspace(.05,20,81),np.geomspace(20,400,81)])

def basis_integrals(knots,low,high):
    a=np.maximum(knots[:-1],low);b=np.minimum(knots[1:],high)
    width=np.maximum(b-a,0);fraction=((a+b)/2-knots[:-1])/np.diff(knots)
    result=np.zeros(len(knots));result[:-1]+=width*(1-fraction);result[1:]+=width*fraction
    return result

def deposit(theta,energy,weights):
    i=np.clip(np.searchsorted(THETA,theta,side='right')-1,0,len(THETA)-2)
    j=np.clip(np.searchsorted(ENERGY,energy,side='right')-1,0,len(ENERGY)-2)
    ft=(theta-THETA[i])/(THETA[i+1]-THETA[i]);fe=(energy-ENERGY[j])/(ENERGY[j+1]-ENERGY[j])
    if np.any((ft<0)|(ft>1)|(fe<0)|(fe>1)):raise ValueError('Production lies outside the table grid')
    h=np.zeros((len(THETA),len(ENERGY)))
    for di,wi in [(0,1-ft),(1,ft)]:
        for dj,wj in [(0,1-fe),(1,fe)]:np.add.at(h,(i+di,j+dj),weights*wi*wj)
    return h

def initialize(root,power):
    global MODEL,POWER
    MODEL=ProductionModel(root);POWER=power

def calculate(mass):
    result=[]
    for category in ('primary','cascade'):
        h=np.zeros((len(THETA),len(ENERGY)));total=cone=0.
        for batch in MODEL.batches(mass,model='dp',power=POWER,categories=(category,),seed=62015001):
            p=batch.momenta;t=np.arctan2(np.linalg.norm(p[:,:2],axis=1),p[:,2]);e=p[:,3]
            h+=deposit(t,e,batch.weights);total+=batch.weights.sum();cone+=batch.weights[t<THETA_SHIP].sum()
        if total<=0 or np.min(h)<-total*1e-12:raise ValueError(f'Nonpositive compatibility density at {mass}, {category}; use the weighted source')
        tw=basis_integrals(THETA,0,np.pi);ew=basis_integrals(ENERGY,mass,ENERGY[-1]);area=tw[:,None]*ew[None,:]
        density=np.divide(h,area*total,out=np.zeros_like(h),where=area>0)
        norm=np.sum(density*area);assert abs(norm-1)<1e-10,(mass,category,norm)
        sampled=total*np.sum(density*basis_integrals(THETA,0,THETA_SHIP)[:,None]*ew[None,:])
        result.append((density.astype('<f4'),total,cone,sampled))
    return mass,result

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=Path(__file__).parent/'data');p.add_argument('--workers',type=int,default=3);p.add_argument('--power',type=int,default=17);a=p.parse_args()
    out=a.data_root/'tables';out.mkdir(exist_ok=True);cache=out/'mass-points';cache.mkdir(exist_ok=True)
    todo=[m for m in MASS if not (cache/f'{m:.12g}.npz').exists()]
    with ProcessPoolExecutor(a.workers,initializer=initialize,initargs=(a.data_root,a.power)) as pool:
        for mass,result in pool.map(calculate,todo):
            np.savez_compressed(cache/f'{mass:.12g}.npz',density=np.array([r[0] for r in result]),yields=[r[1] for r in result],cone=[r[2] for r in result],table_cone=[r[3] for r in result]);print(mass,flush=True)
    all_data=[dict(np.load(cache/f'{m:.12g}.npz')) for m in MASS]
    for i,category in enumerate(('primary','cascade')):
        np.savez_compressed(out/(category+'.npz'),mass=MASS,theta=THETA,energy=ENERGY,density=np.array([d['density'][i] for d in all_data]),yield_per_pot_epsilon2=[d['yields'][i] for d in all_data])
    relative=np.array([np.divide(d['table_cone']-d['cone'],d['cone'],out=np.zeros(2),where=d['cone']!=0) for d in all_data])
    manifest=json.loads((a.data_root/'manifest.json').read_text());manifest.update({'masses':len(MASS),'power':a.power,'theta_nodes':len(THETA),'energy_nodes':len(ENERGY),'position_marginalized':True,'normalization':'per POT and epsilon^2; density normalized above the physical mass threshold','cone_relative_error_max':float(abs(relative).max()),'sha256':{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ('primary.npz','cascade.npz')}})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');np.savez_compressed(out/'validation.npz',mass=MASS,cone_relative_error=relative)

if __name__=='__main__':main()
