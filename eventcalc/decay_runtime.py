"""Existing conditional decay generators applied to independently supplied parents."""
import hashlib
import numpy as np
from eventcalc.sensitivity import DecayBatch, charges_from_pdg


class RuntimeDecayAdapter:
    def __init__(self, config, *, mass, seed=1, invisible_pdgs=(12,14,16,18,4900101)):
        from simulate import _load_runtime, _make_llp
        from eventcalc.runtime_generator import RuntimeEventGenerator
        from funcs.LLP_selection import validate_masses_in_domain
        self.generator=RuntimeEventGenerator(config,mass=mass,c_tau=1.,mode='attempted')
        g=self.generator
        g.resolved_seed=int(seed)
        g.runtime=_load_runtime(headless=True)
        g.llp=_make_llp(g.runtime,config,decay_only=True)
        self.input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in getattr(g.llp,'decay_input_paths',[]) if p.is_file()}
        if g.llp.LLP_name=='ALP-mixed':
            if not np.isfinite(mass) or mass<=0: raise ValueError('Positive finite mass required')
        else:
            mass=validate_masses_in_domain([mass],g.llp.m_min_tabulated,g.llp.m_max_tabulated)[0]
        g.llp.set_mass(mass)
        g.llp.compute_mass_dependent_properties()
        g.llp.set_c_tau(1.)
        g.mass=float(g.llp.mass)
        g.selected_decay_indices=config.selected_decay_indices(tuple(str(x) for x in g.llp.decayChannels))
        g.visible_branching_ratio=float(sum(g.llp.BrRatios_distr[i] for i in g.selected_decay_indices))
        if not np.isfinite(g.visible_branching_ratio) or not 0 < g.visible_branching_ratio <= 1.+1e-10:
            raise ValueError('Selected decay branching fraction must lie in (0,1]')
        self.invisible_pdgs=np.abs(invisible_pdgs)

    def decay(self, parents):
        parents=parents.validate()
        g=self.generator
        if not np.isclose(parents.mass,g.mass,rtol=1e-12,atol=0):
            raise ValueError('External parent mass differs from the propagated decay mother mass')
        n=len(parents.weights)
        products,_=g._decay(parents.momenta,parents.momenta)
        products=np.asarray(products).reshape(n,-1,6)
        present=products[:,:,5]!=-999
        ids=np.where(present,products[:,:,5],0).astype(np.int64)
        visible=present & ~np.isin(np.abs(ids),self.invisible_pdgs)
        daughters=products[:,:,:4].copy();daughters[~present]=0.
        batch=DecayBatch(parents.momenta,daughters,visible,
            parents.weights*g.visible_branching_ratio,parents.mass,parents.origins,
            charges=charges_from_pdg(ids,visible=visible),pdg_ids=ids)
        batch.arrays()  # verify complete daughters and four-momentum conservation
        return batch

    def batches(self, parents):
        for batch in parents:
            if len(batch.weights): yield self.decay(batch)
