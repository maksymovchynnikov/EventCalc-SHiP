"""NA62 truth-level selection primitives, not a reconstructed analysis response."""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from ._pair_cuts import PairCuts


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
        return cls(tuple(map(tuple,data['vertices_z_rho_m'])),tuple(centre_xy),shift_z,shift_rho)

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
        return result
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
            from eventcalc.sensitivity import dipole_displacement
            for i in range(len(v)):
                dx,dy,ok = dipole_displacement(p[i],q[i],dipole.z_min,dipole.z_max,dipole.field_integral,plane_z)
                xy[i] += [dx,dy]
                forward[i] &= ok
    return np.where(forward[:,None],xy,np.nan),forward
