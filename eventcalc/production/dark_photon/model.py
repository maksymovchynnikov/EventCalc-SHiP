"""Calibrated SHiP production, with the target front face at z=0.

Weights are signed integral contributions per incident proton and epsilon^2.
No decay-volume or polar-angle cut is imposed. A fixed mass reuses the stored
single-collision distributions; it never launches Geant4 or Pythia8.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from .fold_idm_production import Inputs,evaluate
from .idm_production_spin import daughter,decay_weight

@dataclass(frozen=True)
class ProductionBatch:
    momenta: np.ndarray
    origins: np.ndarray
    weights: np.ndarray
    mass: float
    category: str
    channel: str

class ProductionModel:
    """Load the calibrated baseline once, then reuse it at arbitrary masses.

    form_factors selects the tabulated proton form-factor pair at the *same*
    fitted Foroughi-Abari virtuality scale. It never selects the old broad
    Central/Lower/Upper flux prescription. Model and PDF uncertainty scans
    require correlated weights; one must not vary each event independently.
    """
    def __init__(self, data_root=None, *, form_factors='central'):
        from .datasets import resolve_full_data
        root=resolve_full_data(data_root)
        metadata=root/'manifest.json'
        if not metadata.exists():
            raise FileNotFoundError(f'Calibrated production data are missing: {metadata}')
        self.metadata=json.loads(metadata.read_text())
        target=self.metadata['target']
        if target.get('z_origin')!='front face':
            raise ValueError('Production coordinates must use the target front face')
        if any(not np.isfinite(target.get(k,np.nan)) or target[k]<=0 for k in ('length_m','diameter_m')):
            raise ValueError('Production data need finite, positive target dimensions')
        source=json.loads((root/'results/idm_production/sources/provenance.json').read_text())
        origins=json.loads((root/'results/propagation/origins/provenance.json').read_text())
        if source['shower_sha256']!=self.metadata['shower_sha256'] or origins['source_provenance']!=source:
            raise ValueError('Production pools and collision origins belong to different showers')
        self.inputs=Inputs(with_origins=True,data_root=root)
        self.inputs.set_formfactors(form_factors)

    def batches(self, mediator_mass, *, model='dp', delta=.1, power=15, seed=1,
                categories=('primary','cascade'), channels=None, chunk_size=4096):
        """Generate weighted mother batches, independently of lifetime.

        iDM uses m_chi=m_A'/3 and alpha_D=0.1, including the running mediator
        width and polarization. The acceptance engine itself has no model
        restriction. `power` gives 2**power common integration points/channel.
        """
        mass=float(mediator_mass)
        if model not in ('dp','idm') or not np.isfinite(mass) or mass<=0:
            raise ValueError('Expected dp or idm and a positive mediator mass')
        if not 4<=int(power)<=24 or int(power)!=power or chunk_size<=0:
            raise ValueError('Invalid integration power or chunk size')
        if not np.isfinite(delta) or delta<=0:raise ValueError('delta must be positive')
        for ic,cat in enumerate(categories):
            if cat not in ('primary','cascade'):raise ValueError('Unknown collision category')
            pending=[]
            def consume(acc,q4,q,m1,m2,n,weight,projection,uncertain=False):
                if model=='idm':
                    mother=daughter(q4,q,m1,m2,n);w=weight*decay_weight(q,m1,m2,projection);m=m2
                else:mother=q4;w=weight;m=mass
                keep=np.flatnonzero(w!=0);norm=len(w)
                for left in range(0,len(keep),chunk_size):
                    ix=keep[left:left+chunk_size]
                    pending.append(ProductionBatch(mother[ix],self.inputs.current_origin[ix].copy(),
                                                   w[ix]/norm,m,cat,acc['channel']))
            evaluate(self.inputs,mass,delta,cat,power=int(power),seed=seed+ic*1721,
                     finite=model=='idm',channels=channels,consumer=consume)
            yield from pending

    def decay_batches(self, mediator_mass, decay_sampler, **kwargs):
        """Compose production with any complete rest-frame decay sampler.

        decay_sampler(mass, count, rng) returns (four_vectors, visibility,
        charges, branching_weights). The last entry is one weight per decay,
        including any branching fraction or importance-sampling correction.
        Invisible daughters must remain in the four-vectors for conservation.
        This stream can be passed straight to eventcalc.sensitivity.integrate/scan_replicas.
        """
        from eventcalc.sensitivity import DecayBatch
        rng=np.random.default_rng(kwargs.get('seed',1)+73619)
        for b in self.batches(mediator_mass,**kwargs):
            d,v,q,br=decay_sampler(b.mass,len(b.weights),rng)
            br=np.asarray(br,float)
            if br.shape!=(len(b.weights),) or not np.all(np.isfinite(br)):
                raise ValueError('The decay sampler must supply one finite branching weight per event')
            yield DecayBatch(b.momenta,d,v,b.weights*br,b.mass,b.origins,frame='rest',charges=q)
