"""CHARM off-axis volume and configurable straight-track lepton acceptance.

Default cuts follow the E>1 GeV phenomenological recast. Detector-plane masks
are nominal geometry proxies, not a validated historical response; see charm_cuts.md.
"""
from dataclasses import dataclass
import numpy as np
from eventcalc.sensitivity import PreparedAcceptance, Selection
from eventcalc.transport import ray_box_intervals

@dataclass(frozen=True)
class Plane:
    z: float
    lower: tuple
    upper: tuple

    def __post_init__(self):
        object.__setattr__(self,"lower",tuple(self.lower))
        object.__setattr__(self,"upper",tuple(self.upper))
        if (np.shape(self.lower) != (2,) or np.shape(self.upper) != (2,) or
                not np.all(np.isfinite([*self.lower, *self.upper, self.z])) or
                np.any(np.asarray(self.upper) <= np.asarray(self.lower))):
            raise ValueError('Invalid rectangular plane')


def daughter_intervals(box, planes, origins, parent_p, daughter_p, required):
    """Parent flight intervals for which every required track hits every plane.

    Daughter lab momenta are fixed per production/decay draw. Straight-line
    projection is affine in parent flight distance, so all cuts are intervals.
    Required flags must be chosen explicitly by the analysis. No magnet model.
    """
    o, p, d, req = (np.asarray(x) for x in (origins, parent_p, daughter_p, required))
    o, p, d = o.astype(float), p.astype(float), d.astype(float)
    req = req.astype(bool)
    if (o.ndim != 2 or o.shape[1] != 3 or p.shape != o.shape or
            d.ndim != 3 or d.shape[0] != len(o) or d.shape[2] != 3 or req.shape != d.shape[:2] or
            not np.all(np.isfinite(d))):
        raise ValueError('Invalid parent/daughter arrays')
    lo, hi = box.intervals(o, p)
    norm = np.linalg.norm(p, axis=1)
    u = np.divide(p, norm[:,None], out=np.zeros_like(p), where=norm[:,None] > 0)
    valid = (hi > lo) & np.any(req, axis=1)
    for plane in planes:
        for j in range(d.shape[1]):
            active = req[:,j]
            dz = d[:,j,2]
            # Forward reach: (plane.z - vertex.z)/daughter.pz > 0.
            moving = dz != 0
            valid &= ~active | moving
            for axis in range(2):
                slope = np.divide(d[:,j,axis], dz, out=np.zeros(len(o)), where=moving)
                intercept = o[:,axis] + (plane.z-o[:,2])*slope
                gradient = u[:,axis] - u[:,2]*slope
                low, high = _linear_interval(intercept, gradient, plane.lower[axis], plane.upper[axis])
                lo = np.maximum(lo, np.where(active,low,-np.inf))
                hi = np.minimum(hi, np.where(active,high,np.inf))
            intercept = (plane.z-o[:,2])*np.sign(dz)
            gradient = -u[:,2]*np.sign(dz)
            low, high = _linear_interval(intercept, gradient, 0., np.inf)
            lo = np.maximum(lo, np.where(active,low,-np.inf))
            hi = np.minimum(hi, np.where(active,high,np.inf))
    valid &= hi > lo
    return np.where(valid,lo,0.), np.where(valid,hi,0.)


def _linear_interval(a, b, lower, upper):
    moving = b != 0
    x = np.divide(lower-a,b,out=np.zeros_like(a),where=moving)
    y = np.divide(upper-a,b,out=np.zeros_like(a),where=moving)
    low = np.where(moving,np.minimum(x,y),-np.inf)
    high = np.where(moving,np.maximum(x,y),np.inf)
    outside = ~moving & ((a < lower) | (a > upper))
    return np.where(outside,np.inf,low), np.where(outside,-np.inf,high)


@dataclass(frozen=True)
class CHARMConfig:
    lower: tuple = (3.5,-1.5,480.)
    upper: tuple = (6.5,1.5,515.)
    planes: tuple = (Plane(521.4,(3.5,-1.5),(6.5,1.5)),)
    daughter_energy_min_GeV: float = 1.
    opposite_charge: bool = True
    efficiency: float = 1.

    def __post_init__(self):
        object.__setattr__(self,'lower',tuple(self.lower))
        object.__setattr__(self,'upper',tuple(self.upper))
        lo,hi=np.asarray(self.lower,float),np.asarray(self.upper,float)
        if lo.shape!=(3,) or hi.shape!=(3,) or not np.all(np.isfinite([lo,hi])) or np.any(hi<=lo):
            raise ValueError('CHARM volume bounds must be finite ordered three-vectors')
        planes=tuple(Plane(**p) if isinstance(p,dict) else p for p in self.planes)
        if any(not isinstance(p,Plane) for p in planes):raise ValueError('planes must contain rectangular Plane configurations')
        if any(p.z<hi[2] for p in planes):raise ValueError('CHARM detector planes must be at or beyond the volume exit')
        object.__setattr__(self,'planes',planes)
        if not np.isfinite(self.daughter_energy_min_GeV) or self.daughter_energy_min_GeV<0:
            raise ValueError('Daughter energy threshold must be finite and nonnegative')
        if not np.isfinite(self.efficiency) or not 0<=self.efficiency<=1:raise ValueError('Efficiency must lie in [0,1]')
        if not isinstance(self.opposite_charge,bool):raise ValueError('opposite_charge must be boolean')

    def intervals(self,origins,momenta):
        return ray_box_intervals(origins,momenta,self.lower,self.upper)


class CHARMResponse:
    def __init__(self,config=None):
        self.config=config or CHARMConfig()
        self.geometry=self.config
        self.selections=tuple(Selection(n,min_particles=0,energy_min=0.) for n in
            ('decay_in_volume','selected_pair','ee','mumu','emu','selected_leptons'))

    def prepare(self,batch):
        (p,d,v,w,o),mass=batch.arrays()
        ids=np.asarray(batch.pdg_ids);q=np.asarray(batch.charges,float)
        if ids.shape!=v.shape or not np.issubdtype(ids.dtype,np.integer) or q.shape!=v.shape or np.any(~np.isfinite(q)):
            raise ValueError('CHARM requires integer daughter pdg_ids and finite charges matching visibility')
        if np.any((v | (d[...,3]>0)) & (ids==0)):raise ValueError('Nonpadding daughters require PDG IDs')
        momentum=np.linalg.norm(p[:,:3],axis=1)
        if np.any((momentum==0)&(w!=0)):raise ValueError('Stationary parents are unsupported')
        c=self.config
        lo,hi=c.intervals(o,p[:,:3])
        accepted_lo,accepted_hi=daughter_intervals(c,c.planes,o,p[:,:3],d[:,:,:3],v)
        tracks=v & (q!=0)
        pair=(tracks.sum(axis=1)==2)&(v.sum(axis=1)==2)
        if c.opposite_charge:pair &= np.sum(np.where(tracks,q,0.),axis=1)==0
        pair &= np.all(~v | (d[...,3]>c.daughter_energy_min_GeV),axis=1)
        electrons=np.sum(tracks & (np.abs(ids)==11),axis=1)
        muons=np.sum(tracks & (np.abs(ids)==13),axis=1)
        masks=(np.ones(len(p),bool),pair,pair&(electrons==2),pair&(muons==2),
               pair&(electrons==1)&(muons==1),pair&((electrons+muons)==2))
        boost=np.divide(mass,momentum,out=np.zeros_like(momentum),where=momentum>0)
        lower=[];upper=[];weights=[];labels=[];indices=[]
        for i,mask in enumerate(masks):
            left,right=(lo,hi) if i==0 else (accepted_lo,accepted_hi)
            take=np.flatnonzero(mask & (right>left))
            lower.append(left[take]*boost[take]);upper.append(right[take]*boost[take])
            weights.append(w[take]*(1. if i==0 else c.efficiency))
            labels.append(np.full(len(take),i,np.int64));indices.append(take)
        return PreparedAcceptance(np.concatenate(lower),np.concatenate(upper),np.concatenate(weights),
            np.concatenate(labels),self.selections,self.geometry,event_index=np.concatenate(indices))
