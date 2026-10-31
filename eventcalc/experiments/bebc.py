"""BEBC chamber-volume acceptance and explicit two-track selections.

Defaults implement the E>1 GeV per-lepton phenomenological recast, not the
complete historical HNL analysis. See bebc_cuts.md for sources and omissions.
"""
from dataclasses import dataclass
import numpy as np
from eventcalc.sensitivity import PreparedAcceptance, Selection
from eventcalc.transport import ray_box_intervals

@dataclass(frozen=True)
class Box:
    lower: tuple = (-1.785, -1.26, 404.)
    upper: tuple = (1.785, 1.26, 405.85)

    def __post_init__(self):
        a, b = np.asarray(self.lower), np.asarray(self.upper)
        if a.shape != (3,) or b.shape != (3,) or not np.all(np.isfinite([a, b])) or np.any(b <= a):
            raise ValueError('Box bounds must be finite ordered three-vectors')

    def intervals(self, origins, momenta):
        """Forward ray distances [entry, exit]; zero interval for a miss/rest state."""
        return ray_box_intervals(origins,momenta,self.lower,self.upper)


@dataclass(frozen=True)
class PairCuts:
    """Exactly two visible charged daughters. Thresholds are independent options.

    No hidden efficiency, transverse-mass, PID or neutral-veto approximation.
    Charge and visible arrays must be supplied, so invisible daughters survive
    in the full event record but do not count as detected tracks.
    """
    track_p_min: float = 0.
    pair_energy_min: float = 0.
    opposite_charge: bool = True
    transverse_mass_max: float | None = None

    def __post_init__(self):
        if not np.all(np.isfinite([self.track_p_min, self.pair_energy_min])) or min(self.track_p_min, self.pair_energy_min) < 0:
            raise ValueError('Thresholds must be finite and nonnegative')
        if self.transverse_mass_max is not None and (not np.isfinite(self.transverse_mass_max) or self.transverse_mass_max <= 0):
            raise ValueError('Transverse mass bound must be positive')

    def select(self, p4, charge, visible, *, transverse_axis=None):
        # Last axis: px,py,pz,E. Padded daughters must have visible=False.
        p4, charge, visible = np.asarray(p4, float), np.asarray(charge, float), np.asarray(visible, bool)
        if p4.ndim != 3 or p4.shape[2] != 4 or charge.shape != p4.shape[:2] or visible.shape != charge.shape:
            raise ValueError('Expected p4 (N,D,4), charge/visible (N,D)')
        if not np.all(np.isfinite(p4)) or not np.all(np.isfinite(charge)) or np.any(p4[..., 3] < 0):
            raise ValueError('Invalid daughter record')
        tracks = visible & (charge != 0)
        pair = (tracks.sum(axis=1) == 2) & (visible.sum(axis=1) == 2)
        track_pass = np.all(~tracks | (np.linalg.norm(p4[..., :3], axis=2) > self.track_p_min), axis=1)
        energy_pass = np.sum(np.where(tracks, p4[..., 3], 0.), axis=1) > self.pair_energy_min
        opposite = (np.sum(np.where(tracks, charge, 0.), axis=1) == 0) if self.opposite_charge else True
        mt_pass = True
        if self.transverse_mass_max is not None:
            if transverse_axis is None:
                raise ValueError('Transverse-mass selection requires an explicit reference axis')
            summed = np.sum(np.where(tracks[..., None], p4, 0.), axis=1)
            mt_pass = transverse_mass(summed, transverse_axis) < self.transverse_mass_max
        return pair & track_pass & energy_pass & opposite & mt_pass


def transverse_mass(pair_p4, axis):
    """sqrt(M_pair^2 + pT_pair^2) + pT_pair, with caller-defined axis.

    The source's HNL cut must not silently become a validated DP/iDM cut.
    """
    a = np.asarray(pair_p4, float)
    n = np.broadcast_to(np.asarray(axis, float), a[..., :3].shape)
    if a.ndim != 2 or a.shape[1] != 4 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(n)):
        raise ValueError('Expected finite pair four-vectors and axes')
    length = np.linalg.norm(n, axis=1)
    if np.any(length == 0): raise ValueError('Reference axis must be nonzero')
    n = n / length[:, None]
    p = a[:, :3]
    m2 = a[:, 3]**2 - np.sum(p*p, axis=1)
    if np.any(a[:, 3] < 0) or np.any(m2 < -1e-10 * np.maximum(a[:, 3]**2, 1)):
        raise ValueError('Pair four-vector must be future timelike or null')
    pt = np.linalg.norm(p - np.sum(p*n, axis=1)[:, None]*n, axis=1)
    return np.sqrt(np.maximum(m2, 0) + pt**2) + pt


@dataclass(frozen=True)
class BEBCConfig:
    lower: tuple = (-1.785, -1.26, 404.)
    upper: tuple = (1.785, 1.26, 405.85)
    daughter_energy_min_GeV: float = 1.
    track_p_min_GeV: float = 0.
    pair_energy_min_GeV: float = 0.
    opposite_charge: bool = True
    transverse_mass_max_GeV: float | None = None
    transverse_axis: tuple | None = None
    efficiency: float = 1.

    def __post_init__(self):
        object.__setattr__(self,'lower',tuple(self.lower))
        object.__setattr__(self,'upper',tuple(self.upper))
        Box(self.lower,self.upper)
        PairCuts(self.track_p_min_GeV,self.pair_energy_min_GeV,self.opposite_charge,self.transverse_mass_max_GeV)
        if not np.isfinite(self.daughter_energy_min_GeV) or self.daughter_energy_min_GeV<0:
            raise ValueError('Daughter energy threshold must be finite and nonnegative')
        if not np.isfinite(self.efficiency) or not 0<=self.efficiency<=1:
            raise ValueError('Efficiency must lie in [0,1]')
        if not isinstance(self.opposite_charge,bool):raise ValueError('opposite_charge must be boolean')
        if self.transverse_axis is not None:
            axis=np.asarray(self.transverse_axis,float)
            if axis.shape!=(3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis)==0:
                raise ValueError('transverse_axis must be an explicit nonzero finite vector')
            object.__setattr__(self,'transverse_axis',tuple(axis))
        if self.transverse_mass_max_GeV is not None and self.transverse_axis is None:
            raise ValueError('A transverse-mass cut requires an explicit transverse_axis')


class BEBCResponse:
    def __init__(self, config=None):
        self.config=config or BEBCConfig()
        self.geometry=self.config
        self.selections=tuple(Selection(n,min_particles=0,energy_min=0.) for n in
            ('decay_in_volume','selected_pair','ee','mumu','emu','selected_leptons'))

    def prepare(self,batch):
        (p,d,v,w,o),mass=batch.arrays()
        ids=np.asarray(batch.pdg_ids);q=np.asarray(batch.charges,float)
        if ids.shape!=v.shape or not np.issubdtype(ids.dtype,np.integer) or q.shape!=v.shape or np.any(~np.isfinite(q)):
            raise ValueError('BEBC requires integer daughter pdg_ids and finite charges matching visibility')
        if np.any((v | (d[...,3]>0)) & (ids==0)):
            raise ValueError('Nonpadding daughters require PDG IDs')
        momentum=np.linalg.norm(p[:,:3],axis=1)
        if np.any((momentum==0)&(w!=0)):raise ValueError('Stationary parents are unsupported')
        c=self.config
        lo,hi=Box(c.lower,c.upper).intervals(o,p[:,:3])
        boost=np.divide(mass,momentum,out=np.zeros_like(momentum),where=momentum>0)
        selected=PairCuts(c.track_p_min_GeV,c.pair_energy_min_GeV,c.opposite_charge,
                          c.transverse_mass_max_GeV).select(d,q,v,transverse_axis=c.transverse_axis)
        selected &= np.all(~v | (d[...,3]>c.daughter_energy_min_GeV),axis=1)
        tracks=v & (q!=0)
        electrons=np.sum(tracks & (np.abs(ids)==11),axis=1)
        muons=np.sum(tracks & (np.abs(ids)==13),axis=1)
        masks=(np.ones(len(p),bool),selected,selected&(electrons==2),selected&(muons==2),
               selected&(electrons==1)&(muons==1),selected&((electrons+muons)==2))
        lower=[];upper=[];weights=[];labels=[];indices=[]
        for i,mask in enumerate(masks):
            take=np.flatnonzero(mask & (hi>lo))
            lower.append(lo[take]*boost[take]);upper.append(hi[take]*boost[take])
            weights.append(w[take]*(1. if i==0 else c.efficiency))
            labels.append(np.full(len(take),i,np.int64));indices.append(take)
        return PreparedAcceptance(np.concatenate(lower),np.concatenate(upper),np.concatenate(weights),
            np.concatenate(labels),self.selections,self.geometry,event_index=np.concatenate(indices))
