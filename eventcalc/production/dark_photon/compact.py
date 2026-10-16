"""Direct sampling of compact mass-angle-energy tables, independent of full EOS data."""
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import hashlib, json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .runtime import SourceSample, trajectory_intervals

ROOT=Path(__file__).resolve().parents[3]/'Distributions'

@lru_cache(maxsize=8)
def table(model,category,delta=.1):
    if model=='dp':manifest_name='manifest.json'; name=category+'.npz'
    elif model=='idm' and delta in (.05,.1):
        manifest_name=f'idm-delta{delta:g}-manifest.json'; name=f'idm-delta{delta:g}-{category}.npz'
    else:raise ValueError('Compact inelastic-DM tables cover Delta=0.05 and 0.1. Use --production-backend correlated for another splitting.')
    root=ROOT/('Dark-photons' if model=='dp' else 'Inelastic-DM')/'production'
    manifest=json.loads((root/manifest_name).read_text()); path=root/name
    with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=manifest['sha256'][name]:raise ValueError(f'Compact-table checksum mismatch: {name}')
    with np.load(path,allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
    return arrays,manifest

def at_mass(model,mass,category,delta=.1):
    a,manifest=table(model,category,delta);m=a['mass'];mass=float(mass)
    if not np.isfinite(mass) or not m[0]<=mass<=m[-1]:
        raise ValueError(f'Compact {model} table covers mediator masses {m[0]:g} to {m[-1]:g} GeV')
    hi=int(np.clip(np.searchsorted(m,mass,side='right'),1,len(m)-1));lo=hi-1
    f=(mass-m[lo])/(m[hi]-m[lo]);y=a['yield_per_pot_epsilon2']
    total=(1-f)*y[lo]+f*y[hi]
    d=((1-f)*y[lo]*a['density'][lo].astype(float)+f*y[hi]*a['density'][hi].astype(float))
    if total>0:d/=total
    return a['theta'],a['energy'],d,float(total),manifest

class Density:
    """Exact piecewise-bilinear sampling, with explicit mass and angle boundaries."""
    def __init__(self,theta,energy,density,mass,theta_max=np.pi):
        from eventcalc.runtime_generator import _sample_linear_density
        self._linear=_sample_linear_density
        low=max(float(mass),float(energy[0]));high=float(energy[-1]);tmax=min(theta_max,theta[-1])
        if low>=high or tmax<=theta[0]:raise ValueError('Empty compact-table phase space')
        self.theta=np.unique(np.r_[theta[0],theta[(theta>theta[0])&(theta<tmax)],tmax])
        self.energy=np.unique(np.r_[low,energy[(energy>low)&(energy<high)],high])
        t,e=np.meshgrid(self.theta,self.energy,indexing='ij')
        self.density=RegularGridInterpolator((theta,energy),density,bounds_error=True)(np.stack((t,e),axis=-1))
        if self.density.min() < -1e-14*self.density.max():raise ValueError('Negative compact-table density')
        self.density=np.maximum(self.density,0.)
        v=self.density
        weights=.25*(v[:-1,:-1]+v[1:,:-1]+v[:-1,1:]+v[1:,1:])*np.diff(self.theta)[:,None]*np.diff(self.energy)[None,:]
        self.shape=weights.shape; self.cdf=np.cumsum(weights.ravel());self.integral=float(self.cdf[-1]);self.cell_count=int(np.count_nonzero(weights))
        if not np.isfinite(self.integral) or self.integral<=0:raise ValueError('Empty compact production density')
    def sample(self,count,rng,*,uniforms=None):
        u=rng.random((count,3)) if uniforms is None else uniforms
        pick=np.minimum(np.searchsorted(self.cdf,u[:,0]*self.integral,side='right'),len(self.cdf)-1)
        i,j=np.unravel_index(pick,self.shape);d=self.density
        a,b,c,e=d[i,j],d[i+1,j],d[i,j+1],d[i+1,j+1]
        x=self._linear((a+c)/2,(b+e)/2,u[:,1]);y=self._linear(a+x*(b-a),c+x*(e-c),u[:,2])
        return self.theta[i]+x*np.diff(self.theta)[i],self.energy[j]+y*np.diff(self.energy)[j]

class CompactSource:
    """Same decay/acceptance interface as the full source; no finite production pool."""
    def __init__(self,config,mass,experiment):
        self.config=config;self.experiment=experiment;self.input_mass=float(mass)
        model='idm' if config.model=='Inelastic-DM' else 'dp'
        self.mass=mass*(1+config.idm_delta)/3 if model=='idm' else mass
        mode=config.dp_production_mode or 'combined'
        self.categories=('primary','cascade') if mode=='combined' else (mode,)
        self.parts=[];self.total_yield=0.
        for cat in self.categories:
            t,e,d,y,meta=at_mass(model,mass,cat,config.idm_delta)
            if y<=0:continue
            full=Density(t,e,d,self.mass)
            self.parts.append((cat,t,e,d/full.integral,y))
            self.total_yield+=y
        if self.total_yield<=0:raise ValueError('No production at this mass in compact tables')
        self.metadata=dict(meta)
        self.metadata.update(production_backend='compact',origin_convention='point source at target front face',
            position_marginalized=True,azimuth_uniform=True,model=model,mediator_mass_GeV=float(mass),
            delta=config.idm_delta if model=='idm' else None,categories=list(self.categories),
            interpolation='linear differential yield in mass, bilinear density in angle and energy; renormalized above mother mass')
        self._samplers={}
    @classmethod
    def build(cls,config,mass,experiment):return cls(config,mass,experiment)
    def sampler(self,c_tau,mode):return CompactSampler(self,c_tau,mode)

class CompactSampler:
    def __init__(self,source,c_tau,mode):
        self.source=source;self.c_tau=float(c_tau);self.parts=[];weights=[]
        theta_max=source.experiment.theta_max_rad if mode=='fiducial' else np.pi
        for cat,t,e,d,y in source.parts:
            try:cdf=Density(t,e,d,source.mass,theta_max)
            except ValueError as error:
                if str(error)=='Empty compact production density':continue
                raise
            self.parts.append((cat,cdf));weights.append(y*cdf.integral)
        self.weights=np.asarray(weights);self.absolute_yield=float(self.weights.sum())
        if self.absolute_yield<=0:raise ValueError('No compact production within the sampling angular range')
        self.cdf=np.cumsum(self.weights);self.integral=self.absolute_yield/source.total_yield
        self.cell_count=sum(c.cell_count for _,c in self.parts)
    def momenta(self,count,rng):
        # One fixed-size random row per particle keeps the stream unchanged
        # when callers choose a different batch size or primary/cascade mix.
        u=rng.random((count,5))
        ix=np.minimum(np.searchsorted(self.cdf,u[:,0]*self.absolute_yield,side='right'),len(self.parts)-1)
        theta=np.empty(count);energy=np.empty(count);categories=np.empty(count,dtype='<U7')
        for k,(cat,cdf) in enumerate(self.parts):
            mask=ix==k;theta[mask],energy[mask]=cdf.sample(int(mask.sum()),rng,uniforms=u[mask,1:4]);categories[mask]=cat
        phi=(2*u[:,4]-1)*np.pi;p=np.sqrt(np.maximum(energy*energy-self.source.mass**2,0.))
        momentum=np.column_stack((p*np.sin(theta)*np.cos(phi),p*np.sin(theta)*np.sin(phi),p*np.cos(theta),energy))
        return momentum,categories
    def sample(self,count,parent_rng,vertex_rng,pdg):
        source=self.source;momentum,categories=self.momenta(count,parent_rng);origins=np.zeros((count,3))
        direction,low,high,valid=trajectory_intervals(momentum,origins,source.experiment)
        p=np.linalg.norm(momentum[:,:3],axis=1);rate=np.divide(source.mass,self.c_tau*p,out=np.zeros(count),where=p>0)
        interval=-np.expm1(-rate*(high-low));probability=np.exp(-rate*low)*interval
        distance=low+np.divide(-np.log1p(-vertex_rng.random(count)*interval),rate,out=np.zeros(count),where=rate>0)
        vertex=distance[:,None]*direction
        mother=np.column_stack((momentum,np.full(count,source.mass),np.full(count,pdg),probability,vertex))
        return SourceSample(mother,momentum,valid,origins,categories,np.full(count,'combined'))
