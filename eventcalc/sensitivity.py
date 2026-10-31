"""Decay acceptances with uniform-dipole magnetic transport, reused for an entire lifetime scan.

Model-independent inputs: complete decay four-vectors, a visibility mask,
production positions, and production weights per incident particle. All
four-vectors are (px, py, pz, E) in GeV and all distances are metres.
No vertex sampling, detector efficiencies or implicit coupling law is used.
"""
from dataclasses import dataclass, asdict
import numpy as np
from numba import njit


from eventcalc.experiments.ship import Geometry, Selection, ship_selections
from eventcalc.experiments._ship_kernels import _volume, _ip, _prepare
from eventcalc.transport import dipole_displacement


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
    pdg_ids: np.ndarray | None = None

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
    geometry: object = None
    event_index: np.ndarray | None = None

    def __post_init__(self):
        self.lower=np.ascontiguousarray(self.lower,dtype=float)
        self.upper=np.ascontiguousarray(self.upper,dtype=float)
        self.weights=np.ascontiguousarray(self.weights,dtype=float)
        self.selected=np.asarray(self.selected)
        n=len(self.lower)
        if any(a.shape!=(n,) for a in (self.lower,self.upper,self.weights,self.selected)):
            raise ValueError('Acceptance arrays must be aligned one-dimensional intervals')
        if (not np.all(np.isfinite(self.lower)) or not np.all(np.isfinite(self.weights)) or
                np.any(np.isnan(self.upper)) or np.any(self.lower<0) or np.any(self.upper<self.lower)):
            raise ValueError('Invalid proper-distance intervals or weights')
        if not np.issubdtype(self.selected.dtype,np.integer) or np.any(self.selected<0) or np.any(self.selected>=len(self.selections)):
            raise ValueError('Acceptance selection indices are invalid')
        self.selected=np.ascontiguousarray(self.selected,dtype=np.int64)
        if self.event_index is not None:
            indices=np.asarray(self.event_index)
            if indices.shape!=(n,) or not np.issubdtype(indices.dtype,np.integer) or np.any(indices<0):
                raise ValueError('Acceptance event indices are invalid')
            self.event_index=np.ascontiguousarray(indices,dtype=np.int64)

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
    """Compatibility entry point for the SHiP response."""
    from eventcalc.experiments.ship import ShipResponse
    return ShipResponse(geometry, selections).prepare(batch)


def integrate(batches, ctaus, *, geometry=None, selections=None, exposure=1., production_scales=None, response_nodes=None, experiment=None):
    """Consume batches once and return event yields [lifetime, selection].

Normalized momentum and decay distributions must stay fixed across this
scan. Supply arbitrary lifetime and production scaling arrays; no epsilon,
mass ratio, width or branching-fraction law is assumed. Regenerate batches
when the scanned parameter changes their distributions.
"""
    ctaus = np.atleast_1d(np.asarray(ctaus, float))
    if experiment is not None and (geometry is not None or selections is not None):
        raise ValueError("Use experiment or legacy geometry/selections, not both")
    selections = tuple(experiment.selections if experiment is not None else (ship_selections() if selections is None else selections))
    scales = np.ones_like(ctaus) if production_scales is None else np.asarray(production_scales, float)
    if scales.shape != ctaus.shape or not np.all(np.isfinite(scales)) or np.any(scales < 0) or not np.isfinite(exposure) or exposure < 0:
        raise ValueError('Invalid production scales or exposure')
    if ctaus.ndim != 1 or not len(ctaus) or np.any(~np.isfinite(ctaus)) or np.any(ctaus <= 0):
        raise ValueError('Invalid proper decay lengths')
    geometry=experiment.geometry if experiment is not None else (geometry or Geometry.ship())
    result = np.zeros((len(ctaus), len(selections)))
    response=None if response_nodes is None else ResponseAccumulator(selections,geometry,nodes=response_nodes)
    for batch in batches:
        accepted=experiment.prepare(batch) if experiment is not None else prepare(batch, geometry=geometry, selections=selections)
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
        # Sum the weight carried between adjacent endpoints. Stable exponential
        # differences avoid cancellation on both lifetime boundaries, including
        # signed NLO contributions and empty leading/trailing grid bins.
        cumulative=self.zero_lower[...,None]+np.cumsum(self.weights,axis=-1)
        active=self.weights!=0
        last=np.where(active,np.arange(len(self.distance)),-1).max(axis=-1)
        cumulative=np.where(np.arange(len(self.distance))<last[...,None],cumulative,0.)
        basis=np.exp(-self.distance[:-1,None]/ctaus[None,:])*(-np.expm1(-np.diff(self.distance)[:,None]/ctaus[None,:]))
        result=cumulative[...,:-1] @ basis
        result+=self.zero_lower[...,None]*(-np.expm1(-self.distance[0]/ctaus))[None,None,:]
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
        if not self.selections or len({s.name for s in self.selections}) != len(self.selections):
            raise ValueError("Selections need unique names")
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
