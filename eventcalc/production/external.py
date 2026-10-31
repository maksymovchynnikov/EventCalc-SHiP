"""Portable production inputs. Weights are integral contributions per incident particle."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np


@dataclass(frozen=True)
class ParentBatch:
    momenta: np.ndarray
    origins: np.ndarray
    weights: np.ndarray
    mass: float

    def validate(self):
        p, o, w = (np.asarray(x, float) for x in (self.momenta, self.origins, self.weights))
        if p.ndim != 2 or p.shape[1] != 4 or o.shape != (len(p), 3) or w.shape != (len(p),):
            raise ValueError('Expected momenta[N,4], origins[N,3], weights[N]')
        if not np.isfinite(self.mass) or self.mass <= 0 or any(not np.all(np.isfinite(x)) for x in (p,o,w)):
            raise ValueError('Mass must be positive; production arrays must be finite')
        if np.any(p[:,3] <= 0) or not np.allclose(p[:,3]**2-np.sum(p[:,:3]**2,axis=1), self.mass**2, rtol=1e-6, atol=1e-7):
            raise ValueError('Parent four-vectors must be future on-shell')
        if np.any(np.linalg.norm(p[:,:3],axis=1) == 0):
            raise ValueError('Stationary parents require a different flight-distance parameterization')
        return ParentBatch(*(np.ascontiguousarray(x) for x in (p,o,w)), float(self.mass))

    def chunks(self, size):
        if isinstance(size,bool) or not isinstance(size,int) or size <= 0:
            raise ValueError('chunk_size must be a positive integer')
        for start in range(0,len(self.weights),size):
            sl=slice(start,start+size)
            yield ParentBatch(self.momenta[sl],self.origins[sl],self.weights[sl],self.mass)


def density_to_probabilities(density, theta_edges, energy_edges):
    """Convert density dP/(dtheta dE) by bin area; no implicit sin(theta)."""
    t,e=_edges(theta_edges,energy_edges)
    values=np.asarray(density,float)
    if values.shape != (len(t)-1,len(e)-1) or np.any(~np.isfinite(values)) or np.any(values<0):
        raise ValueError('Invalid density')
    probabilities=values*np.diff(t)[:,None]*np.diff(e)[None,:]
    if probabilities.sum() <= 0: raise ValueError('Density has no support')
    return probabilities/probabilities.sum()


def _edges(theta,energy):
    t,e=np.asarray(theta,float),np.asarray(energy,float)
    for x in (t,e):
        if x.ndim!=1 or len(x)<2 or np.any(~np.isfinite(x)) or np.any(np.diff(x)<=0):
            raise ValueError('Bin edges must be finite and strictly increasing')
    if t[0]<0 or t[-1]>np.pi: raise ValueError('theta edges must lie in [0,pi] radians')
    return t,e


class ExternalProduction:
    """One exact mass per manifest; interpolation and normalization are never inferred."""
    def __init__(self, manifest, *, mass):
        self.path=Path(manifest).resolve()
        self.metadata=json.loads(self.path.read_text(encoding='utf8'))
        m=self.metadata
        required={'schema_version':1,'frame':'lab','momentum_order':['px','py','pz','E'],
                  'energy_unit':'GeV','length_unit':'m','angle_unit':'rad',
                  'normalization':'produced_parents_per_incident'}
        if any(m.get(k)!=v for k,v in required.items()):
            raise ValueError('Unsupported production schema, units, frame or normalization')
        if not m.get('coordinate_frame'): raise ValueError('An explicit coordinate_frame description is required')
        if not np.isfinite(mass) or mass<=0 or float(m.get('mass_GeV',0)) != float(mass):
            raise ValueError('Requested mass is not supplied; mass extrapolation is forbidden')
        self.mass=float(mass)
        self.array_path=(self.path.parent/m['arrays']).resolve()
        with np.load(self.array_path,allow_pickle=False) as z:
            self.arrays={k:z[k] for k in z.files}
        self.hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.path,self.array_path)}
        if m['kind']=='parents':
            self.parents=ParentBatch(self.arrays['momenta'],self.arrays['origins'],self.arrays['weights'],self.mass).validate()
        elif m['kind']=='spectrum':
            self.theta,self.energy=_edges(self.arrays['theta_edges'],self.arrays['energy_edges'])
            self.probabilities=np.asarray(self.arrays['bin_probabilities'],float)
            if self.energy[0]<self.mass or self.probabilities.shape!=(len(self.theta)-1,len(self.energy)-1):
                raise ValueError('Invalid spectrum support or dimensions')
            if np.any(~np.isfinite(self.probabilities)) or np.any(self.probabilities<0) or not np.isclose(self.probabilities.sum(),1.,rtol=0,atol=1e-12):
                raise ValueError('Bin probabilities must be nonnegative and sum to one')
            self.probability=float(m['production_probability'])
            self.origin=np.asarray(m['origin_m'],float)
            if not np.isfinite(self.probability) or self.probability<0 or self.origin.shape!=(3,) or np.any(~np.isfinite(self.origin)):
                raise ValueError('Invalid production probability or explicit origin')
        else: raise ValueError('kind must be parents or spectrum')

    def batches(self, *, events=131072, seed=1, chunk_size=4096):
        if self.metadata['kind']=='parents':
            yield from self.parents.chunks(chunk_size)
            return
        if isinstance(events,bool) or not isinstance(events,int) or events<=0:
            raise ValueError('events must be a positive integer')
        if isinstance(chunk_size,bool) or not isinstance(chunk_size,int) or chunk_size<=0:
            raise ValueError('chunk_size must be a positive integer')
        # Independent streams make sample identities invariant under chunking.
        streams=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
        for start in range(0,events,chunk_size):
            n=min(chunk_size,events-start)
            bins=streams[0].choice(self.probabilities.size,n,p=self.probabilities.ravel())
            i,j=np.unravel_index(bins,self.probabilities.shape)
            theta=self.theta[i]+streams[1].random(n)*(self.theta[i+1]-self.theta[i])
            energy=self.energy[j]+streams[2].random(n)*(self.energy[j+1]-self.energy[j])
            phi=streams[3].uniform(0,2*np.pi,n);p=np.sqrt(energy**2-self.mass**2)
            momenta=np.column_stack((p*np.sin(theta)*np.cos(phi),p*np.sin(theta)*np.sin(phi),p*np.cos(theta),energy))
            yield ParentBatch(momenta,np.broadcast_to(self.origin,(n,3)).copy(),np.full(n,self.probability/events),self.mass).validate()


class BundledProduction:
    """Expose the existing calibrated production model through ParentBatch."""
    def __init__(self, model):
        self.model=model

    def batches(self, mass, **kwargs):
        for b in self.model.batches(mass,**kwargs):
            yield ParentBatch(b.momenta,b.origins,b.weights,b.mass).validate()
