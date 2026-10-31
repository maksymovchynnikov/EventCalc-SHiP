"""NA62 geometry, cuts, transport, acceptance preparation and cutflow.

Frozen analytic ee/mumu response and ideal-response hadronic p>1 GeV projection.
No tracking/PID/trigger/veto efficiencies, photon merging or mass-window model.
The hadronic vertex discretization is a numerical approximation.
"""
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from numba import njit

from eventcalc.sensitivity import Geometry, PreparedAcceptance, Selection
from eventcalc.transport import dipole_displacement

# Configuration and geometry

@dataclass(frozen=True)
class NA62Config:
    vertex_cells: int = 128
    hadron_p_min_GeV: float = 1.
    lepton_p_min_GeV: float = 10.

    def __post_init__(self):
        if isinstance(self.vertex_cells, bool) or not isinstance(self.vertex_cells, int) or self.vertex_cells < 2:
            raise ValueError('At least two vertex cells required')
        if self.hadron_p_min_GeV != 1. or self.lepton_p_min_GeV != 10.:
            raise ValueError('This frozen response supports hadron p>1 and lepton p>10 GeV only')


LEPTON_SELECTIONS=('volume_ee','selected_ee','detector_ee','volume_mumu','selected_mumu','detector_mumu')


ORIGIN=np.array([0.,-.022,23.07])


ZMAG=(196.35,197.65)


FIELD=.6928*(ZMAG[1]-ZMAG[0])


PLANE=243.85


STRAW=183.


LKR=241.495  # SensCalc card; nominal front plane, not a verified 2021 alignment.


STAGES=('cylinder_volume','restricted_fv','two_opposite_hadron_tracks','track_p_threshold',
        'track_aperture_proxy','separation_straw','separation_lkr','photons_reconstructed','hadronic_pointing')

# Selection and transport primitives

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
class ElectronFV:
    """Piecewise linear upper-radius contour digitized from ee Fig.2.

    An explicitly supplied transverse centre must be transformed with vertices.
    shift_z/rho are digitization-envelope studies, not experimental nuisances.
    """
    vertices: tuple
    centre_xy: tuple
    shift_z: float = 0.
    shift_rho: float = 0.

    def __post_init__(self):
        a=np.asarray(self.vertices,float)
        if (a.shape != (5,2) or not np.all(np.isfinite(a)) or
                not (a[0,0] == a[1,0] < a[2,0] == a[3,0] < a[4,0]) or
                np.any(a[:,1]<0) or np.shape(self.centre_xy)!=(2,) or
                not np.all(np.isfinite([*self.centre_xy,self.shift_z,self.shift_rho]))):
            raise ValueError('Invalid step contour/centre')

    @classmethod
    def digitized(cls, *, centre_xy, shift_z=0., shift_rho=0.):
        path=Path(__file__).resolve().parent/'data/na62_ee_fv.json'
        data=json.loads(path.read_text(encoding='utf8'))
        vertices=np.asarray(data['vertices_z_rho_m'],float)
        radius_at_105=np.interp(105.,vertices[[1,2],0],vertices[[1,2],1])
        vertices[:2,0]=105.
        vertices[1,1]=radius_at_105
        return cls(tuple(map(tuple,vertices)),tuple(centre_xy),shift_z,shift_rho)

    def __call__(self,z,xy):
        z,xy=np.asarray(z,float)-self.shift_z,np.asarray(xy,float)
        a=np.asarray(self.vertices)
        if z.ndim!=1 or xy.shape!=(len(z),2) or not np.all(np.isfinite(z)) or not np.all(np.isfinite(xy)):
            raise ValueError('Invalid FV vertices')
        first=np.interp(z,a[[1,2],0],a[[1,2],1])
        second=np.interp(z,a[[3,4],0],a[[3,4],1])
        bound=np.where(z<a[2,0],first,second)+self.shift_rho
        return (z>a[0,0]) & (z<a[4,0]) & (np.linalg.norm(xy-self.centre_xy,axis=1)<bound)


def dilepton_selection(p4, pdg, visible, *, channel):
    """Two opposite-sign selected leptons with individual p>10 GeV/c.

    Only kinematic/species selection; no trigger, PID, veto or resolution.
    Invisible daughters remain in the input but do not enter the pair.
    """
    if channel not in ('ee','mumu'): raise ValueError('Choose ee or mumu')
    pdg,visible = np.asarray(pdg),np.asarray(visible,bool)
    if pdg.shape != visible.shape: raise ValueError('PDG and visibility shapes differ')
    species = 11 if channel == 'ee' else 13
    correct = np.all(~visible | (np.abs(pdg) == species),axis=1)
    charge = np.where(np.abs(pdg) == species,-np.sign(pdg),0)
    return correct & PairCuts(track_p_min=10.).select(p4,charge,visible)


def pointing(vertex, pair_momentum, *, beam_xy):
    """Closest points between the visible-pair line and a beam parallel to z.

    Returns z on the pair line, distance between lines, and validity. Exactly
    parallel lines have no unique z and are invalid, rather than assigned TAX z.
    Vertex momenta must be before the spectrometer kick, including for iDM.
    """
    v, p = np.asarray(vertex, float), np.asarray(pair_momentum, float)
    beam = np.asarray(beam_xy, float)
    if v.ndim != 2 or v.shape[1] != 3 or p.shape != v.shape or beam.shape != (2,) or not np.all(np.isfinite([v,p])) or not np.all(np.isfinite(beam)):
        raise ValueError('Expected finite (N,3) vertices/momenta and beam (2,)')
    delta = v[:,:2]-beam
    pt2 = np.sum(p[:,:2]**2,axis=1)
    valid = pt2 > 0
    t = np.divide(-np.sum(delta*p[:,:2],axis=1),pt2,out=np.zeros(len(v)),where=valid)
    z = v[:,2] + t*p[:,2]
    cda = np.linalg.norm(delta+t[:,None]*p[:,:2],axis=1)
    return np.where(valid,z,np.nan), np.where(valid,cda,np.nan), valid


def signal_region(z, cda, *, channel, nominal_target_z=0.):
    """Original-analysis pointing regions in translated nominal-target frame."""
    z,cda = np.broadcast_arrays(np.asarray(z,float)-nominal_target_z,np.asarray(cda,float))
    valid = np.isfinite(z) & np.isfinite(cda) & (cda >= 0)
    if channel == 'mumu': return valid & (z > 6) & (z < 40) & (cda < .020)
    if channel == 'ee': return valid & (((z-23)/12)**2+(cda/.03)**2 < 1)
    raise ValueError('Choose separate ee or mumu analysis')


def vertex_selection(vertices, *, channel, nominal_target_z=0., electron_fv=None):
    """Only established z cuts for muons; electron FV must be supplied explicitly.

    electron_fv receives (z relative to nominal target, absolute transverse xy)
    and must implement the verified Fig.2 boundary in that same transverse frame.
    """
    v = np.asarray(vertices,float)
    if v.ndim != 2 or v.shape[1] != 3 or not np.all(np.isfinite(v)):
        raise ValueError('Expected finite vertices (N,3)')
    z = v[:,2]-nominal_target_z
    if channel == 'mumu': return (z > 105) & (z < 180)
    if channel == 'ee':
        if electron_fv is None: raise ValueError('Verified electron restricted FV is required')
        result = np.asarray(electron_fv(z,v[:,:2]),bool)
        if result.shape != z.shape: raise ValueError('FV callback must return one flag per vertex')
        return result & (z > 105) & (z < 180)
    raise ValueError('Choose separate ee or mumu analysis')


@dataclass(frozen=True)
class Dipole:
    z_min: float
    z_max: float
    field_integral: float

    def __post_init__(self):
        if not np.all(np.isfinite([self.z_min,self.z_max,self.field_integral])) or self.z_max <= self.z_min:
            raise ValueError('Invalid uniform dipole')


def project_tracks(vertices, momenta, charges, plane_z, *, dipole=None):
    """Reuse EventCalc exact uniform-field transport, independently for each plane.

    Supports vertices upstream of the dipole and planes before or after it.
    Planes within the magnet and decays in/after it are rejected explicitly.
    No material effects or detector aperture is imposed here.
    """
    v,p = np.asarray(vertices,float),np.asarray(momenta,float)
    q = np.asarray(charges,float)
    if v.ndim != 2 or v.shape[1] != 3 or p.shape != v.shape or q.shape != (len(v),) or not np.all(np.isfinite([v,p])) or not np.all(np.isfinite(q)) or not np.isfinite(plane_z):
        raise ValueError('Invalid track arrays')
    forward = (p[:,2] > 0) & (plane_z > v[:,2])
    slope = np.divide(p[:,:2],p[:,2,None],out=np.zeros_like(p[:,:2]),where=p[:,2,None] != 0)
    xy = v[:,:2]+(plane_z-v[:,2])[:,None]*slope
    if dipole is not None:
        if np.any(v[:,2] >= dipole.z_min) or dipole.z_min < plane_z < dipole.z_max:
            raise ValueError('Only upstream vertices and planes outside magnet supported')
        if plane_z >= dipole.z_max:
            for i in range(len(v)):
                dx,dy,ok = dipole_displacement(p[i],q[i],dipole.z_min,dipole.z_max,dipole.field_integral,plane_z)
                xy[i] += [dx,dy]
                forward[i] &= ok
    return np.where(forward[:,None],xy,np.nan),forward

# Analytic lepton acceptance

def circle_polynomial(a,b,r0,r1=0.):
    """norm(a+b*z)^2 - (r0+r1*z)^2, ascending coefficients."""
    return np.stack([np.sum(a*a,axis=-1)-r0*r0,
                     2*(np.sum(a*b,axis=-1)-r0*r1),np.sum(b*b,axis=-1)-r1*r1],axis=-1)


def roots(c):
    """Two finite real roots or NaNs, including degenerate linear cases."""
    c=np.asarray(c,float);d,b,a=np.moveaxis(c,-1,0)
    out=np.full(c.shape[:-1]+(2,),np.nan)
    quad=a!=0
    disc=b*b-4*a*d
    valid=quad & (disc>=0)
    # Stable quadratic formula; repeated zero roots handled separately.
    q=-.5*(b+np.copysign(np.sqrt(np.maximum(disc,0)),b))
    np.divide(q,a,out=out[...,0],where=valid)
    np.divide(d,q,out=out[...,1],where=valid & (q!=0))
    repeat=valid & (q==0)
    out[...,1]=np.where(repeat,out[...,0],out[...,1])
    linear=~quad & (b!=0)
    np.divide(-d,b,out=out[...,0],where=linear)
    return out


def prepare_leptons(batch,pdg):
    (p,d,v,w,o),mass=batch.arrays()
    ids=np.asarray(pdg);q=np.asarray(batch.charges,float)
    if ids.shape!=v.shape or q.shape!=v.shape: raise ValueError('PDG/charge arrays required')
    forward=p[:,2]>0
    beta=np.divide(p[:,:2],p[:,2,None],out=np.zeros_like(p[:,:2]),where=forward[:,None])
    alpha=o[:,:2]-o[:,2,None]*beta
    fv=ElectronFV.digitized(centre_xy=(0,0));vertices=np.asarray(fv.vertices)
    lines=[]
    for left,right in [(1,2),(3,4)]:
        r1=(vertices[right,1]-vertices[left,1])/(vertices[right,0]-vertices[left,0])
        lines.append((vertices[left,1]-r1*vertices[left,0],r1))
    base=[circle_polynomial(alpha,beta,1.)]
    base += [circle_polynomial(alpha,beta,r0,r1) for r0,r1 in lines]
    # Restrict full inclusive events to the explicitly supported leptonic channels.
    # The complete inclusive branching fraction is already represented in samples.
    pair=np.sum(np.where(v[...,None],d,0.),axis=1)
    channel_pass={c:dilepton_selection(d,ids,v,channel=c) for c in ['ee','mumu']}
    selected_any=channel_pass['ee']|channel_pass['mumu']
    if np.any(selected_any & (np.sum(np.where(~v,d[...,3],0.),axis=1)>1e-7)):
        raise ValueError('Prototype interval preparation supports fully visible DP pairs only; iDM needs pointing-boundary roots')
    # Complete events have already been validated. Only the selected visible
    # pair needs detector transport; do not allocate K-daughter interval tensors
    # for hadronic events which fail both leptonic channels.
    order=np.argsort(~v,axis=1,kind='stable')[:,:2]
    if order.shape[1]!=2: raise ValueError('At least two daughter slots required')
    dd=np.take_along_axis(d,order[...,None],axis=1)
    qq=np.take_along_axis(q,order,axis=1)
    vv=np.broadcast_to(selected_any[:,None],order.shape)
    slopes=np.zeros_like(dd[...,:2]);shifts=np.zeros_like(slopes);ok=np.zeros(vv.shape,bool)
    for i,j in zip(*np.nonzero(vv)):
        if dd[i,j,2]<=0: continue
        sx,sy,good=dipole_displacement(dd[i,j,:3],qq[i,j],*ZMAG,FIELD,PLANE)
        if good:
            slopes[i,j]=dd[i,j,:2]/dd[i,j,2];shifts[i,j]=[sx,sy];ok[i,j]=True
    a=alpha[:,None,:]+PLANE*slopes+shifts
    b=beta[:,None,:]-slopes
    detector_polys=[]
    for j in range(2):
        detector_polys.extend([circle_polynomial(a[:,j],b[:,j],1.15),circle_polynomial(a[:,j],b[:,j],.08)])
    all_roots=np.concatenate([roots(c) for c in base+detector_polys],axis=1)
    all_roots=np.where(np.isfinite(all_roots),all_roots,180.)
    lefts=[];rights=[];weights=[];selections=[];indices=[]
    for ci,channel in enumerate(['ee','mumu']):
        low=105.
        breaks=np.column_stack([np.full(len(p),low),np.full(len(p),180.),np.full(len(p),vertices[2,0]),all_roots])
        breaks=np.sort(np.clip(breaks,low,180.),axis=1)
        lo,hi=breaks[:,:-1],breaks[:,1:];mid=(lo+hi)/2
        xyz=np.empty(mid.shape+(3,));xyz[...,:2]=alpha[:,None,:]+mid[...,None]*beta[:,None,:];xyz[...,2]=mid
        vol=(hi>lo)&forward[:,None]&(np.sum(xyz[...,:2]**2,axis=-1)<1.)&(lo>=o[:,2,None])
        z,cda,valid=pointing(xyz.reshape(-1,3),np.repeat(pair[:,:3],mid.shape[1],axis=0),beam_xy=ORIGIN[:2])
        sr=signal_region(z,cda,channel=channel).reshape(mid.shape)
        selected=vol & channel_pass[channel][:,None] & sr
        if channel=='ee': selected &= fv(mid.ravel(),xyz[...,:2].reshape(-1,2)).reshape(mid.shape)
        hit=a[:,None,:,:]+mid[:,:,None,None]*b[:,None,:,:]
        r2=np.sum(hit*hit,axis=-1)
        detector=np.all(~vv[:,None,:] | (ok[:,None,:] & (r2>.08**2)&(r2<1.15**2)),axis=-1)
        masks=[vol,selected,selected&detector]
        for si,mask in enumerate(masks):
            ii,jj=np.nonzero(mask)
            factor=mass/p[ii,2]
            lefts.append((lo[ii,jj]-o[ii,2])*factor);rights.append((hi[ii,jj]-o[ii,2])*factor)
            weights.append(w[ii]);selections.append(np.full(len(ii),3*ci+si,dtype=np.int64));indices.append(ii.astype(np.int64))
    geom=Geometry(z_min=105,z_max=180,x_in=2,x_out=2,y_in=2,y_out=2,aperture_z=PLANE,aperture_x=2.3,aperture_y=2.3,
                  magnet_z_min=ZMAG[0],magnet_z_max=ZMAG[1],field_integral=FIELD,target_z=ORIGIN[2])
    return PreparedAcceptance(np.concatenate(lefts),np.concatenate(rights),np.concatenate(weights),np.concatenate(selections),
        tuple(Selection(n,min_particles=0,energy_min=0) for n in LEPTON_SELECTIONS),geom,event_index=np.concatenate(indices))

# Hadronic cuts and cutflow

@njit(cache=True)
def fv_pass(z,x,y,vertices):
    if z<=vertices[0,0] or z>=vertices[4,0]:return False
    l,r=(1,2) if z<vertices[2,0] else (3,4)
    bound=vertices[l,1]+(z-vertices[l,0])*(vertices[r,1]-vertices[l,1])/(vertices[r,0]-vertices[l,0])
    return x*x+y*y<bound*bound


@njit(cache=True)
def hadron_pointing(x,y,z,momentum):
    pt2=momentum[0]**2+momentum[1]**2
    if pt2<=0:return False
    dx=x-ORIGIN[0];dy=y-ORIGIN[1]
    t=-(dx*momentum[0]+dy*momentum[1])/pt2
    ztax=z+t*momentum[2]
    cda2=(dx+t*momentum[0])**2+(dy+t*momentum[1])**2
    return ((ztax-23.)/23.)**2+cda2/.04**2<1.


@njit(cache=True)
def decisions(z,alpha,beta,d,ids,q,a,b,sa,sb,ga,gb,transport_ok,vertices,pmin):
    """One event at one vertex; a/b final track, s STRAW, g LKr."""
    flags=np.zeros(9,np.bool_);xy=alpha+z*beta
    if np.dot(xy,xy)>=1.:return flags,0,False
    flags[0]=True
    if not fv_pass(z,xy[0],xy[1],vertices):return flags,0,False
    flags[1]=True
    tracks=np.zeros(2,np.int64);nc=0;supported=True
    for j in range(len(ids)):
        code=abs(ids[j])
        if code==0 or code in (12,14,16,18,4900101,4900102):continue
        if code not in (211,321,22):supported=False
        if q[j]!=0:
            if nc<2:tracks[nc]=j
            nc+=1
    if not supported or nc!=2 or q[tracks[0]]*q[tracks[1]]>=0:return flags,0,False
    flags[2]=True
    for j in tracks:
        if np.sqrt(np.dot(d[j,:3],d[j,:3]))<=pmin:return flags,0,False
    flags[3]=True
    for j in tracks:
        hit=a[j]+z*b[j];r2=np.dot(hit,hit)
        if not transport_ok[j] or r2<=.08**2 or r2>=1.15**2:return flags,0,False
    flags[4]=True
    delta=sa[tracks[0]]-sa[tracks[1]]+z*(sb[tracks[0]]-sb[tracks[1]])
    if np.dot(delta,delta)<.02**2:return flags,0,False
    flags[5]=True
    delta=ga[tracks[0]]-ga[tracks[1]]+z*(gb[tracks[0]]-gb[tracks[1]])
    if np.dot(delta,delta)<.2**2:return flags,0,False
    flags[6]=True
    reconstructed=d[tracks[0],:3]+d[tracks[1],:3]
    ngamma=0;all_photons=True
    for j in range(len(ids)):
        if ids[j]!=22:continue
        hit=ga[j]+z*gb[j];r2=np.dot(hit,hit)
        detected=d[j,3]>5. and transport_ok[j] and r2>.08**2 and r2<1.15**2
        if detected:reconstructed+=d[j,:3];ngamma+=1
        else:all_photons=False
    # This is reconstruction, not a veto on undetected photons.
    flags[7]=True
    flags[8]=hadron_pointing(xy[0],xy[1],z,reconstructed)
    return flags,ngamma,all_photons


@njit(cache=True)
def transport(alpha,beta,d,q):
    k=len(d);a=np.zeros((k,2));b=np.zeros((k,2));sa=a.copy();sb=b.copy();ga=a.copy();gb=b.copy();ok=np.zeros(k,np.bool_)
    for j in range(k):
        if d[j,2]<=0:continue
        slope=d[j,:2]/d[j,2]
        dx,dy,good=dipole_displacement(d[j,:3],q[j],*ZMAG,FIELD,PLANE)
        if not good:continue
        a[j]=alpha+PLANE*slope+np.array([dx,dy]);b[j]=beta-slope
        sa[j]=alpha+STRAW*slope;sb[j]=beta-slope
        dx,dy,good=dipole_displacement(d[j,:3],q[j],*ZMAG,FIELD,LKR)
        if not good:continue
        ga[j]=alpha+LKR*slope+np.array([dx,dy]);gb[j]=beta-slope;ok[j]=True
    return a,b,sa,sb,ga,gb,ok


@njit(cache=True)
def _integrate_cutflow(p,d,ids,q,w,o,mass,ctaus,groups,vertices,pmins,nz):
    # [threshold scenario, lifetime, stage, terminal-state group]
    result=np.zeros((len(pmins),len(ctaus),9,6))
    photons=np.zeros((len(pmins),len(ctaus),6))
    full_photon_control=np.zeros((len(pmins),len(ctaus)))
    dz=75./nz
    for i in range(len(p)):
        if p[i,2]<=0:continue
        beta=p[i,:2]/p[i,2];alpha=o[i,:2]-o[i,2]*beta
        a,b,sa,sb,ga,gb,ok=transport(alpha,beta,d[i],q[i])
        for iz in range(nz):
            lo=105.+iz*dz;hi=lo+dz;z=(lo+hi)/2
            prob=np.zeros(len(ctaus))
            for c in range(len(ctaus)):
                rate=mass/p[i,2]/ctaus[c]
                prob[c]=w[i]*np.exp(-(lo-o[i,2])*rate)*(-np.expm1(-dz*rate))
            for s in range(len(pmins)):
                flags,ng,allg=decisions(z,alpha,beta,d[i],ids[i],q[i],a,b,sa,sb,ga,gb,ok,vertices,pmins[s])
                for c in range(len(ctaus)):
                    for stage in range(9):
                        if flags[stage]:result[s,c,stage,groups[i]]+=prob[c]
                    if flags[8]:
                        photons[s,c,min(ng,5)]+=prob[c]
                        if allg:full_photon_control[s,c]+=prob[c]
    return result,photons,full_photon_control


def evaluate_cutflow(batch,ids,groups,ctaus,pmins=(1.,10.),nz=64):
    arrays,mass=batch.arrays();p,d,v,w,o=arrays
    return _integrate_cutflow(p,d,np.asarray(ids,np.int64),np.asarray(batch.charges,float),w,o,mass,
        np.asarray(ctaus,float),np.asarray(groups,np.int64),np.asarray(ElectronFV.digitized(centre_xy=(0,0)).vertices),np.asarray(pmins,float),nz)

# Hadronic acceptance intervals

@njit(cache=True)
def accepted_cells(p,d,ids,q,w,o,mass,groups,vertices,nz):
    lower=[];upper=[];weights=[];selected=[];indices=[]
    dz=75./nz
    for i in range(len(p)):
        if p[i,2]<=0:continue
        tracks=0;charge=0.;supported=True
        for j in range(len(ids[i])):
            code=abs(ids[i,j])
            if code in (0,12,14,16,18,4900101,4900102):continue
            if code not in (211,321,22):supported=False
            if q[i,j]!=0:
                tracks+=1;charge+=q[i,j]
                if np.sqrt(np.dot(d[i,j,:3],d[i,j,:3]))<=1. or d[i,j,2]<=0:supported=False
        if not supported or tracks!=2 or charge!=0:continue
        beta=p[i,:2]/p[i,2];alpha=o[i,:2]-o[i,2]*beta
        a,b,sa,sb,ga,gb,ok=transport(alpha,beta,d[i],q[i])
        start=-1.
        for iz in range(nz):
            lo=105.+iz*dz;hi=lo+dz
            flags,ng,allg=decisions((lo+hi)/2,alpha,beta,d[i],ids[i],q[i],a,b,sa,sb,ga,gb,ok,vertices,1.)
            if flags[8] and start<0:start=lo
            if start>=0 and (not flags[8] or iz==nz-1):
                end=hi if flags[8] else lo
                start=max(start,o[i,2])
                if end<=start:
                    start=-1.
                    continue
                factor=mass/p[i,2]
                lower.append((start-o[i,2])*factor);upper.append((end-o[i,2])*factor)
                weights.append(w[i]);selected.append(groups[i]);indices.append(i);start=-1.
    return (np.asarray(lower),np.asarray(upper),np.asarray(weights),np.asarray(selected,dtype=np.int64),np.asarray(indices,dtype=np.int64))


def prepare_hadrons(batch,ids,groups,nz=128):
    if not isinstance(nz,int) or nz<2:raise ValueError('At least two vertex cells required')
    arrays,mass=batch.arrays();p,d,v,w,o=arrays
    values=accepted_cells(p,d,np.asarray(ids,np.int64),np.asarray(batch.charges,float),w,o,mass,
        np.asarray(groups,np.int64),np.asarray(ElectronFV.digitized(centre_xy=(0,0)).vertices),nz)
    geometry=Geometry(z_min=105,z_max=180,x_in=2,x_out=2,y_in=2,y_out=2,aperture_z=PLANE,aperture_x=2.3,aperture_y=2.3,
        magnet_z_min=ZMAG[0],magnet_z_max=ZMAG[1],field_integral=FIELD,target_z=ORIGIN[2])
    selections=tuple(Selection(f'hadronic_group_{g}',min_particles=0,energy_min=0) for g in range(6))
    return PreparedAcceptance(*values[:4],selections,geometry,event_index=values[4])

# Experiment response

def classify(ids):
    """Disjoint terminal-state diagnostics, not primary resonance labels."""
    result=[]
    for row in ids:
        s=set(np.abs(row));s.discard(0)
        if s and s<={211}:g=0
        elif 211 in s and s<={211,22}:g=1
        elif 321 in s:g=2
        elif s & {130,310}:g=3
        elif s & {2112,2212}:g=4
        else:g=5
        result.append(g)
    return np.asarray(result,dtype=np.int64)


class NA62Response:
    def __init__(self, config=None):
        self.config = config or NA62Config()
        self.geometry = None  # geometry belongs to this response, not the lifetime engine
        self.selections = tuple(Selection(n, min_particles=0, energy_min=0.) for n in
            (*LEPTON_SELECTIONS, *(f'hadronic_group_{i}' for i in range(6))))

    def prepare(self, batch):
        arrays, mass = batch.arrays()
        ids = np.asarray(batch.pdg_ids)
        charges = np.asarray(batch.charges, float)
        if ids.shape != arrays[2].shape or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError('NA62 requires integer daughter pdg_ids matching the visibility mask')
        if charges.shape != ids.shape or not np.all(np.isfinite(charges)):
            raise ValueError('NA62 requires finite daughter charges')
        a = prepare_leptons(batch, ids)
        b = prepare_hadrons(batch, ids, classify(ids), nz=self.config.vertex_cells)
        return PreparedAcceptance(np.r_[a.lower,b.lower], np.r_[a.upper,b.upper],
            np.r_[a.weights,b.weights], np.r_[a.selected,b.selected+len(a.selections)],
            self.selections, self.geometry, event_index=np.r_[a.event_index,b.event_index])

    def cutflow(self, batch, ctaus):
        self.prepare(batch)  # validate the same complete-event contract
        return evaluate_cutflow(batch,batch.pdg_ids,classify(batch.pdg_ids),ctaus,
                       pmins=(self.config.hadron_p_min_GeV,),nz=self.config.vertex_cells)
