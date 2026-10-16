"""Decay acceptances with uniform-dipole magnetic transport, reused for an entire lifetime scan.

Model-independent inputs: complete decay four-vectors, a visibility mask,
production positions, and production weights per incident particle. All
four-vectors are (px, py, pz, E) in GeV and all distances are metres.
No vertex sampling, detector efficiencies or implicit coupling law is used.
"""
from dataclasses import dataclass, asdict
import numpy as np
from numba import njit


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
    field_integral: float = 0.5  # integral B_y dz in tesla metres; SensCalc ECN3

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


@dataclass
class DecayBatch:
    mother: np.ndarray
    daughters: np.ndarray
    visible: np.ndarray
    weights: np.ndarray
    mass: float
    origins: np.ndarray | None = None
    frame: str = 'lab'
    charges: np.ndarray | None = None  # daughter charges in units of positron charge

    def arrays(self):
        p, d = np.asarray(self.mother, float), np.asarray(self.daughters, float)
        v, w = np.asarray(self.visible), np.asarray(self.weights, float)
        m = float(self.mass)
        if p.ndim != 2 or p.shape[1] != 4 or d.ndim != 3 or d.shape[0] != len(p) or d.shape[2] != 4:
            raise ValueError('Expected mother[N,4] and complete daughters[N,K,4]')
        if v.dtype != np.bool_ or v.shape != d.shape[:2] or w.shape != (len(p),):
            raise ValueError('Explicit boolean visibility and one weight per event are required')
        o = np.zeros((len(p), 3)) if self.origins is None else np.asarray(self.origins, float)
        if o.shape != (len(p), 3) or not np.isfinite(m) or m <= 0 or self.frame not in ('lab', 'rest'):
            raise ValueError('Invalid origins, mass or daughter frame')
        if any(not np.all(np.isfinite(a)) for a in (p, d, o, w)):
            raise ValueError('Nonfinite kinematics or weights')
        if np.any(p[:, 3] <= 0) or np.any(d[:, :, 3] < 0) or not np.allclose(p[:, 3]**2 - np.sum(p[:, :3]**2, axis=1), m*m, rtol=1e-6, atol=1e-7):
            raise ValueError('Unphysical mother or daughter energies')
        if np.any(d[:, :, 3]**2 - np.sum(d[:, :, :3]**2, axis=2) < -1e-6):
            raise ValueError('Spacelike daughter')
        target = p if self.frame == 'lab' else np.broadcast_to([0., 0., 0., m], p.shape)
        if not np.allclose(d.sum(axis=1), target, rtol=1e-6, atol=2e-7):
            raise ValueError('Complete decay must conserve four-momentum; include invisible daughters')
        if self.frame == 'rest':
            beta = p[:, :3] / p[:, 3, None]
            gamma = p[:, 3] / m
            bp = np.einsum('ni,nki->nk', beta, d[:, :, :3])
            lab = d.copy()
            lab[:, :, :3] += ((gamma**2/(gamma+1))[:, None]*bp + gamma[:, None]*d[:, :, 3])[:, :, None]*beta[:, None, :]
            lab[:, :, 3] = gamma[:, None]*(d[:, :, 3]+bp)
            d = lab
        return tuple(np.ascontiguousarray(a) for a in (p, d, v, w, o)), m


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
def dipole_displacement(momentum, charge, z_min, z_max, field_integral, aperture_z):
    """ECAL shift relative to a straight trajectory, plus forward-crossing flag.

    A uniform B_y changes px by -0.299792458*q*integral(B_y dz).
    Integrating the circular orbit through the magnet avoids the small-angle
    kick failure for soft tracks. Material and fringe fields are omitted.
    Input momenta are at the decay vertex; the returned shift is independent
    of that vertex when it is upstream of the magnet. Thus the lifetime
    integral remains analytic. The neutral/zero-field limit is exact.
    """
    px, py, pz = momentum
    if pz <= 0: return 0., 0., False
    if charge == 0 or field_integral == 0: return 0., 0., True
    length = z_max-z_min
    curvature = 0.299792458*charge*field_integral/length
    kick = curvature*length
    px_out = px-kick
    transverse2 = px*px+pz*pz
    if px_out*px_out >= transverse2: return 0., 0., False
    pz_out = np.sqrt(transverse2-px_out*px_out)
    # Rationalized difference avoids cancellation as the field tends to zero.
    dx = length*(px+px_out)/(pz+pz_out)
    # Difference of angles via atan2 is stable for small kicks.
    angle = np.arctan2(px*pz_out-px_out*pz, px*px_out+pz*pz_out)
    dy = py*angle/curvature
    drift = aperture_z-z_max
    return (dx-length*px/pz + drift*(px_out/pz_out-px/pz),
            dy-length*py/pz + drift*(py/pz_out-py/pz), True)


def charges_from_pdg(pdg, *, visible=None):
    """Charges for terminal SM particles; unknown visible IDs must be explicit.

    This helper is for EventCalc's adapter. The acceptance engine accepts any
    model's explicit charge array and never guesses visibility from PDG IDs.
    """
    ids = np.asarray(pdg)
    if not np.all(np.isfinite(ids)) or np.any(ids != np.rint(ids)):
        raise ValueError('PDG identifiers must be finite integers')
    ids = ids.astype(np.int64)
    active = np.ones(ids.shape, bool) if visible is None else np.asarray(visible, bool)
    if active.shape != ids.shape: raise ValueError('Visibility/PDG shape mismatch')
    positive_id = {11:-1., 13:-1., 15:-1., 24:1., 211:1., 321:1.,
                   2212:1., 3112:-1., 3222:1., 3312:-1., 3334:-1.,
                   411:1., 431:1., 521:1., 541:1., 4122:1.}
    neutral = {0,12,14,16,18,21,22,23,111,130,310,311,2112,3122,
               3212,3322,421,511,531,999,4900101,4900102}
    result = np.zeros(ids.shape, float)
    for code in np.unique(np.abs(ids[active])):
        take = np.abs(ids) == code
        if code in positive_id: result[take] = np.sign(ids[take])*positive_id[code]
        elif code in neutral: pass
        elif code >= 1000000000: result[take] = np.sign(ids[take])*((code//10000)%1000)
        else: raise ValueError(f'Unknown visible PDG ID {code}; provide explicit charges')
    return result


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


@njit(cache=True)
def _evaluate(lower, upper, weights, selected, ctaus, nselection):
    result = np.zeros((len(ctaus), nselection))
    for i in range(len(lower)):
        for j in range(len(ctaus)):
            probability = np.exp(-lower[i]/ctaus[j])*(-np.expm1(-(upper[i]-lower[i])/ctaus[j]))
            result[j, selected[i]] += weights[i]*probability
    return result


@dataclass
class PreparedAcceptance:
    lower: np.ndarray
    upper: np.ndarray
    weights: np.ndarray
    selected: np.ndarray
    selections: tuple
    geometry: Geometry
    event_index: np.ndarray | None = None

    def response(self, event_reweights=None, *, nodes=8192):
        """Cache a log-distance response for many lifetime/weight variations.

        event_reweights has shape [input event, variation]. This is a
        controlled interpolation; use evaluate() to check the exact integral.
        """
        return lifetime_response(self, event_reweights, nodes=nodes)

    def evaluate(self, ctaus):
        ctaus = np.atleast_1d(np.asarray(ctaus, float))
        if ctaus.ndim != 1 or not len(ctaus) or np.any(~np.isfinite(ctaus)) or np.any(ctaus <= 0):
            raise ValueError('Proper decay lengths must be finite, positive metres')
        return _evaluate(self.lower, self.upper, self.weights, self.selected, ctaus, len(self.selections))


def prepare(batch, *, geometry=None, selections=None):
    """Prepare one bounded batch; weights already include production and BR.

Weights are integral contributions per incident particle, NOT a normalized
histogram and NOT decay probabilities. They may be signed. This function
does not infer which daughters are detectable: supply a visibility mask.
"""
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


def integrate(batches, ctaus, *, geometry=None, selections=None, exposure=1., production_scales=None, response_nodes=None):
    """Consume batches once and return event yields [lifetime, selection].

Normalized momentum and decay distributions must stay fixed across this
scan. Supply arbitrary lifetime and production scaling arrays; no epsilon,
mass ratio, width or branching-fraction law is assumed. Regenerate batches
when the scanned parameter changes their distributions.
"""
    ctaus = np.atleast_1d(np.asarray(ctaus, float))
    selections = tuple(ship_selections() if selections is None else selections)
    scales = np.ones_like(ctaus) if production_scales is None else np.asarray(production_scales, float)
    if scales.shape != ctaus.shape or not np.all(np.isfinite(scales)) or np.any(scales < 0) or not np.isfinite(exposure) or exposure < 0:
        raise ValueError('Invalid production scales or exposure')
    if ctaus.ndim != 1 or not len(ctaus) or np.any(~np.isfinite(ctaus)) or np.any(ctaus <= 0):
        raise ValueError('Invalid proper decay lengths')
    geometry=geometry or Geometry.ship()
    result = np.zeros((len(ctaus), len(selections)))
    response=None if response_nodes is None else ResponseAccumulator(selections,geometry,nodes=response_nodes)
    for batch in batches:
        accepted=prepare(batch, geometry=geometry, selections=selections)
        if response is None:result+=accepted.evaluate(ctaus)
        else:response.add(accepted)
    if response is not None:result=response.evaluate(ctaus)[0]
    return result * (exposure*scales[:, None])


def scan_replicas(factory, ctaus, *, replicas=2, **kwargs):
    """factory(i) returns an independently sampled, fully normalized batch stream.

The error covers independent integration replicas, not model uncertainty
or the statistics of a reused source pool. Reusing the same events with
different block labels does not create independent replicas.
"""
    if isinstance(replicas, bool) or int(replicas) != replicas or replicas < 2:
        raise ValueError('At least two independent replicas are required')
    replicas = int(replicas)
    samples = np.asarray([integrate(factory(i), ctaus, **kwargs) for i in range(replicas)])
    return samples.mean(axis=0), samples.std(axis=0, ddof=1)/np.sqrt(replicas)


def threshold_intervals(parameters, yields, threshold=2.3):
    """All connected intervals above threshold, with log-coordinate crossings.

None at an endpoint means the reach is open at that grid boundary. Never
join separate excluded regions by sorting their contour vertices.
"""
    x, y = np.asarray(parameters, float), np.asarray(yields, float)
    if x.ndim != 1 or len(x) < 2 or y.shape != x.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(x <= 0) or np.any(np.diff(x) <= 0) or not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Expected ordered positive parameter nodes and finite event yields')
    inside = y >= threshold; intervals = []; start = None
    def crossing(i):
        a, b = y[i], y[i+1]
        fraction = ((np.log(threshold)-np.log(a))/(np.log(b)-np.log(a))
                    if a > 0 and b > 0 else (threshold-a)/(b-a))
        return float(np.exp(np.log(x[i])+fraction*np.log(x[i+1]/x[i])))
    for i in range(len(x)-1):
        if not inside[i] and inside[i+1]: start = crossing(i)
        elif inside[i] and not inside[i+1]:
            intervals.append((start, crossing(i))); start = None
    if inside[-1]: intervals.append((start, None))
    return intervals


@dataclass
class LifetimeResponse:
    """Reusable production-weight responses, independent of the particle model."""
    distance: np.ndarray
    weights: np.ndarray
    zero_lower: np.ndarray

    def evaluate(self, ctaus):
        ctaus = np.atleast_1d(np.asarray(ctaus, float))
        if ctaus.ndim != 1 or not len(ctaus) or np.any(~np.isfinite(ctaus)) or np.any(ctaus <= 0):
            raise ValueError('Proper decay lengths must be finite, positive metres')
        if not len(self.distance):
            return np.zeros((self.weights.shape[0], len(ctaus), self.weights.shape[1]))
        # Empty bins may extend far below the physical decay-distance support.
        # Switching on the grid minimum causes cancellation of saturated -1
        # terms at short lifetimes and can create a false high-coupling reach.
        active = np.any(self.weights != 0., axis=(0, 1))
        first = self.distance[np.flatnonzero(active)[0]] if active.any() else self.distance[0]
        short = first/ctaus > .1
        z = self.distance[:, None]/ctaus[None, :]
        basis = np.where(short[None, :], np.exp(-z), np.expm1(-z))
        result = self.weights @ basis
        result += self.zero_lower[:, :, None]*short[None, None, :]
        return result.transpose(0, 2, 1)


@njit(cache=True)
def _response_histogram(lower, upper, weights, selected, identity, ratios, low, step, nodes, ns):
    h = np.zeros((ratios.shape[1], ns, nodes))
    zero = np.zeros((ratios.shape[1], ns))
    for i in range(len(lower)):
        for endpoint, sign in [(lower[i], 1.), (upper[i], -1.)]:
            if endpoint == 0.:
                for k in range(ratios.shape[1]):
                    zero[k, selected[i]] += weights[i]*ratios[identity[i], k]*sign
                continue
            x = (np.log(endpoint)-low)/step
            j = min(max(int(x), 0), nodes-2)
            t = x-j
            for k in range(ratios.shape[1]):
                w = weights[i]*ratios[identity[i], k]*sign
                h[k, selected[i], j] += w*(1-t)
                h[k, selected[i], j+1] += w*t
    return h, zero


def lifetime_response(prepared, event_reweights=None, *, nodes=8192):
    if not isinstance(nodes, int) or nodes < 2:
        raise ValueError('The response needs at least two distance nodes')
    if prepared.event_index is None:
        raise ValueError('Rebuild acceptance with prepare() to retain event identities')
    n = int(prepared.event_index.max(initial=-1))+1
    ratios = np.ones((n, 1)) if event_reweights is None else np.asarray(event_reweights, float)
    if ratios.ndim != 2 or ratios.shape[0] < n or ratios.shape[1] == 0 or not np.all(np.isfinite(ratios)):
        raise ValueError('Reweights must be finite [input event, variation] values')
    if not len(prepared.lower):
        return LifetimeResponse(np.empty(0), np.zeros((ratios.shape[1], len(prepared.selections), 0)), np.zeros((ratios.shape[1], len(prepared.selections))))
    positive = np.r_[prepared.lower[prepared.lower > 0], prepared.upper]
    low = np.log(positive.min())-1e-10
    high = np.log(positive.max())+1e-10
    step = (high-low)/(nodes-1)
    h, zero = _response_histogram(prepared.lower, prepared.upper, prepared.weights, prepared.selected, prepared.event_index, np.ascontiguousarray(ratios), low, step, nodes, len(prepared.selections))
    return LifetimeResponse(np.exp(np.linspace(low, high, nodes)), h, zero)


class ResponseAccumulator:
    """Bounded-memory lifetime response for an on-the-fly event stream.

    All batches use one log-distance grid. `nodes` controls interpolation;
    compare with exact PreparedAcceptance.evaluate() when selecting precision.
    Zero lower endpoints are handled separately, without taking log(0).
    """
    def __init__(self,selections,geometry,*,nodes=8192,distance_range=(1e-12,1e12),variations=1):
        self.selections=tuple(selections);self.geometry=geometry
        _selection_array(self.selections);geometry.array()
        if not isinstance(nodes,int) or nodes<2 or not isinstance(variations,int) or variations<1:
            raise ValueError('Invalid response grid size')
        low,high=distance_range
        if not np.isfinite(low+high) or not 0<low<high:raise ValueError('Invalid proper-distance range')
        self.low=np.log(low);self.step=(np.log(high)-self.low)/(nodes-1)
        self.response=LifetimeResponse(np.geomspace(low,high,nodes),np.zeros((variations,len(selections),nodes)),np.zeros((variations,len(selections))))

    def add(self,prepared,event_reweights=None):
        if prepared.selections!=self.selections or prepared.geometry!=self.geometry:
            raise ValueError('All response batches must use the same geometry and selections')
        if prepared.event_index is None:raise ValueError('Event identities are required')
        if not len(prepared.lower):return
        positive=np.r_[prepared.lower[prepared.lower>0],prepared.upper]
        if positive.min()<self.response.distance[0] or positive.max()>self.response.distance[-1]:
            raise ValueError('Acceptance lies outside response distance_range; enlarge it or integrate exactly')
        n=int(prepared.event_index.max(initial=-1))+1
        ratios=np.ones((n,1)) if event_reweights is None else np.asarray(event_reweights,float)
        if ratios.ndim!=2 or ratios.shape[0]<n or ratios.shape[1]!=self.response.weights.shape[0] or not np.all(np.isfinite(ratios)):
            raise ValueError('Reweights must be finite [input event, variation] values')
        h,z=_response_histogram(prepared.lower,prepared.upper,prepared.weights,prepared.selected,prepared.event_index,np.ascontiguousarray(ratios),self.low,self.step,len(self.response.distance),len(self.selections))
        self.response.weights+=h;self.response.zero_lower+=z

    def evaluate(self,ctaus):
        return self.response.evaluate(ctaus)
