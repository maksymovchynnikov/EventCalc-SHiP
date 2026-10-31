"""SHiP geometry, selections and acceptance intervals; legacy defaults preserved."""
from dataclasses import dataclass, asdict
import numpy as np
from numba import njit

from eventcalc.transport import dipole_displacement

@dataclass(frozen=True)
class Geometry:
    z_min: float = 32.
    z_max: float = 82.
    x_in: float = 1.
    x_out: float = 4.
    y_in: float = 2.7
    y_out: float = 6.2
    aperture_z: float = 97.32
    aperture_x: float = 4.32
    aperture_y: float = 6.48
    target_z: float = 0.
    magnet_z_min: float = 86.
    magnet_z_max: float = 91.
    field_integral: float = 0.77  # central integral B_y dz in tesla metres; SHiP-ECN3

    @classmethod
    def ship(cls):
        from funcs import ship_setup as s
        return cls(z_min=s.z_min, z_max=s.z_max, x_in=s.Delta_x_in,
                   x_out=s.Delta_x_out, y_in=s.Delta_y_in, y_out=s.Delta_y_out)

    def array(self):
        a = np.array(list(asdict(self).values()), dtype=float)
        if not np.all(np.isfinite(a)) or self.z_max <= self.z_min or self.aperture_z < self.z_max or np.any(a[2:6] <= 0) or np.any(a[7:9] <= 0):
            raise ValueError('Invalid decay volume or downstream aperture')
        if self.field_integral != 0 and not (self.z_max <= self.magnet_z_min < self.magnet_z_max <= self.aperture_z):
            raise ValueError('The dipole must lie after the decay volume and before the aperture')
        return a


@dataclass(frozen=True)
class Selection:
    name: str
    min_particles: int = 2
    all_visible: bool = False
    energy_min: float = 1.
    ip_max: float | None = None


def ship_selections():
    return (Selection('decay_in_volume', 0, energy_min=0.),
            Selection('at_least_two_energy'),
            Selection('at_least_two_energy_ip', ip_max=2.5),
            Selection('all_visible_energy', all_visible=True),
            Selection('all_visible_energy_ip', all_visible=True, ip_max=2.5))


def _selection_array(selections):
    if not selections or len({s.name for s in selections}) != len(selections):
        raise ValueError('Selections need unique names')
    rows = []
    for s in selections:
        if not s.name or isinstance(s.min_particles, bool) or int(s.min_particles) != s.min_particles or s.min_particles < 0:
            raise ValueError('Invalid particle count')
        if not np.isfinite(s.energy_min) or s.energy_min < 0 or (s.ip_max is not None and (not np.isfinite(s.ip_max) or s.ip_max < 0)):
            raise ValueError('Invalid energy or impact-parameter cut')
        if s.min_particles == 0 and (s.all_visible or s.ip_max is not None):
            raise ValueError('A zero-particle selection is decay-volume only')
        rows.append([s.min_particles, s.all_visible, s.energy_min, np.inf if s.ip_max is None else s.ip_max])
    return np.asarray(rows, dtype=float)


@njit(cache=True)
def _volume(origin, direction, g):
    lo, hi = 0., np.inf
    if abs(direction[2]) < 1e-15:
        if origin[2] < g[0] or origin[2] > g[1]: return 0., 0.
    else:
        z1, z2 = (g[0]-origin[2])/direction[2], (g[1]-origin[2])/direction[2]
        lo, hi = max(lo, min(z1, z2)), min(hi, max(z1, z2))
    for dim, win, wout in ((0, g[2], g[3]), (1, g[4], g[5])):
        slope = (wout-win)/(2*(g[1]-g[0]))
        offset = win/2-slope*g[0]
        for sign in (-1., 1.):
            a = sign*direction[dim]-slope*direction[2]
            b = offset+slope*origin[2]-sign*origin[dim]
            if abs(a) < 1e-15:
                if b < 0: return 0., 0.
            elif a > 0: hi = min(hi, b/a)
            else: lo = max(lo, b/a)
    return (lo, hi) if hi > lo else (0., 0.)


@njit(cache=True)
def _ip(lo, hi, origin, direction, momentum, target_z, radius):
    if np.isinf(radius): return lo, hi
    if momentum[2] <= 0: return lo, lo
    a = origin[:2] - (origin[2]-target_z)*momentum[:2]/momentum[2]
    b = direction[:2] - direction[2]*momentum[:2]/momentum[2]
    aa, ab, cc = np.dot(b, b), np.dot(a, b), np.dot(a, a)-radius*radius
    if aa < 1e-30: return (lo, hi) if cc <= 0 else (lo, lo)
    disc = ab*ab-aa*cc
    if disc < 0: return lo, lo
    root = np.sqrt(disc)
    return max(lo, (-ab-root)/aa), min(hi, (-ab+root)/aa)


@njit(cache=True)
def _prepare(p, d, visible, weights, origins, mass, g, cuts, charges):
    n, nk, ns = len(p), d.shape[1], len(cuts)
    maximum = n*ns*max(1, 2*nk)
    lower = np.empty(maximum); upper = np.empty(maximum)
    weight = np.empty(maximum); selected = np.empty(maximum, np.int64); event_index = np.empty(maximum, np.int64)
    size = 0
    for i in range(n):
        momentum = np.sqrt(np.sum(p[i, :3]**2))
        if weights[i] == 0: continue
        if momentum <= 0:
            raise ValueError('Stationary mothers require a time-dependent source calculation')
        u = p[i, :3]/momentum
        lo, hi = _volume(origins[i], u, g)
        if hi <= lo: continue
        inverse_boost = mass/momentum
        for s in range(ns):
            count, all_visible, energy, radius = cuts[s]
            if count == 0:
                lower[size], upper[size], weight[size], selected[size] = lo*inverse_boost, hi*inverse_boost, weights[i], s
                event_index[size] = i
                size += 1
                continue
            left = np.empty(nk); right = np.empty(nk); mom = np.empty((nk, 3))
            nh, nv = 0, 0
            for k in range(nk):
                if not visible[i, k]: continue
                nv += 1
                if d[i, k, 3] <= energy or d[i, k, 2] <= 0: continue
                shift_x, shift_y, forward = dipole_displacement(d[i, k, :3], charges[i, k], g[10], g[11], g[12], g[6])
                if not forward: continue
                a, b = lo, hi
                for dim, width in ((0, g[7]), (1, g[8])):
                    slope = u[dim]-u[2]*d[i, k, dim]/d[i, k, 2]
                    offset = origins[i, dim]+(g[6]-origins[i, 2])*d[i, k, dim]/d[i, k, 2] + (shift_x if dim == 0 else shift_y)
                    if abs(slope) < 1e-15:
                        if abs(offset) >= width/2: b = a
                    else:
                        x = (-width/2-offset)/slope; y = (width/2-offset)/slope
                        a, b = max(a, min(x, y)), min(b, max(x, y))
                if b > a:
                    left[nh], right[nh], mom[nh] = a, b, d[i, k, :3]
                    nh += 1
            if nh < count or (all_visible and nh < nv): continue
            endpoints = np.sort(np.concatenate((left[:nh], right[:nh])))
            for k in range(len(endpoints)-1):
                a, b = endpoints[k], endpoints[k+1]
                if b <= a: continue
                mid = (a+b)/2; hits = 0; psum = np.zeros(3)
                for j in range(nh):
                    if left[j] < mid < right[j]:
                        hits += 1; psum += mom[j]
                if hits < count or (all_visible and hits != nv): continue
                a, b = _ip(a, b, origins[i], u, psum, g[9], radius)
                if b <= a: continue
                lower[size], upper[size], weight[size], selected[size] = a*inverse_boost, b*inverse_boost, weights[i], s
                event_index[size] = i
                size += 1
    return lower[:size].copy(), upper[:size].copy(), weight[:size].copy(), selected[:size].copy(), event_index[:size].copy()


def _prepare_ship(batch, *, geometry=None, selections=None):
    """Prepare one bounded batch; weights already include production and BR.

Weights are integral contributions per incident particle, NOT a normalized
histogram and NOT decay probabilities. They may be signed. This function
does not infer which daughters are detectable: supply a visibility mask.
"""
    from eventcalc.sensitivity import PreparedAcceptance
    geometry = geometry or Geometry.ship()
    selections = tuple(ship_selections() if selections is None else selections)
    arrays, mass = batch.arrays()
    g = geometry.array()
    if batch.charges is None:
        if geometry.field_integral != 0:
            raise ValueError('Magnetic transport requires explicit daughter charges; set field_integral=0 for straight trajectories')
        charges = np.zeros(arrays[1].shape[:2])
    else:
        charges = np.asarray(batch.charges, dtype=float)
        if charges.shape != arrays[1].shape[:2] or not np.all(np.isfinite(charges)):
            raise ValueError('charges must be finite with shape [event, daughter]')
    values = _prepare(*arrays, mass, g, _selection_array(selections), np.ascontiguousarray(charges))
    return PreparedAcceptance(*values[:4], selections, geometry, event_index=values[4])



class ShipResponse:
    def __init__(self, geometry=None, selections=None):
        self.geometry = geometry or Geometry.ship()
        self.selections = tuple(ship_selections() if selections is None else selections)
        self.geometry.array()
        _selection_array(self.selections)

    def prepare(self, batch):
        return _prepare_ship(batch, geometry=self.geometry, selections=self.selections)
