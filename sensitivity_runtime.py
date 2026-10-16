"""Lifetime-independent production/decay batches from installed EventCalc models.

Use sensitivity.DecayBatch directly for external polarized production, target
cascades or weighted hadronization. This adapter uses RuntimeModelScan's normal
decay generator, with correlated target positions for the calibrated dark-photon source. Other installed table sources retain their point-source convention.
"""
import copy
import numpy as np
from sensitivity import DecayBatch, charges_from_pdg


def runtime_batches(scan, mass, *, events=131072, seed=1, chunk_size=4096,
                    invisible_pdgs=(12, 14, 16, 4900101), yield_per_incident=None):
    """Yield batches with no decay-position or lifetime-dependent energy cut.

The default production normalization is llp.Yield, before coupling scaling.
Override yield_per_incident for another explicitly chosen normalization.
All selected decay branching fractions are included exactly once. Invisible
new particles must be named in invisible_pdgs in addition to neutrinos.
"""
    from runtime_generator import _ProductionInverseCDF
    for n in (events, chunk_size):
        if isinstance(n, bool) or int(n) != n or n <= 0:
            raise ValueError('events and chunk_size must be positive integers')
    # c_tau is only required to initialize the existing interface. Neither its
    # coupling normalization nor its survival-biased CDF is used below.
    generator = scan.generator(mass=mass, c_tau=1., mode='attempted', seed=seed)
    generator.init()
    if getattr(generator.llp,'production_profile',None)=='calibrated':
        if yield_per_incident is not None:
            raise ValueError('The calibrated weighted source supplies its absolute normalization; use production_scales to rescale it')
        from production_model import ProductionModel
        mode=generator.llp.dp_production_mode
        categories=('primary','cascade') if mode=='combined' else (mode,)
        model=ProductionModel()
        # Keep signed production weights and the actual correlated collision
        # positions. No unweighting or lifetime-dependent CDF is used here.
        for batch in model.batches(mass,model="idm" if scan.config.model=="Inelastic-DM" else "dp",
                                   delta=scan.config.idm_delta,power=max(4,int(np.ceil(np.log2(events)))),
                                   seed=seed,categories=categories,chunk_size=chunk_size):
            mother=batch.momenta;n=len(mother)
            products,_=generator._decay(mother,mother)
            products=products.reshape(n,-1,6);present=products[:,:,5]!=-999
            visible=present & ~np.isin(np.abs(products[:,:,5]),np.abs(invisible_pdgs))
            daughters=products[:,:,:4].copy();daughters[~present]=0.
            yield DecayBatch(mother,daughters,visible,batch.weights*generator.visible_branching_ratio,
                             batch.mass,batch.origins,charges=charges_from_pdg(products[:,:,5],visible=visible))
        return
    grid = copy.copy(generator._grid)
    grid.survival_energy_floor = 0.
    grid.theta_max = min(float(grid.Distr[1].max()), float(grid.grid_a[-1]))
    cdf = _ProductionInverseCDF(grid)
    production = generator.llp.Yield if yield_per_incident is None else yield_per_incident
    if not np.isfinite(production) or production < 0:
        raise ValueError('Invalid production normalization')
    weight = float(production)*generator.visible_branching_ratio*cdf.integral/events
    if weight == 0: return
    rng = np.random.default_rng(np.random.SeedSequence([seed, 173]))
    for offset in range(0, events, chunk_size):
        n = min(chunk_size, events-offset)
        theta, energy = cdf.sample(n, rng)
        phi = rng.uniform(-np.pi, np.pi, n)
        p = np.sqrt(np.maximum(energy**2-mass**2, 0.))
        mother = np.column_stack((p*np.sin(theta)*np.cos(phi),
                                  p*np.sin(theta)*np.sin(phi), p*np.cos(theta), energy))
        products, _ = generator._decay(mother, mother)
        products = products.reshape(n, -1, 6)
        present = products[:, :, 5] != -999
        visible = present & ~np.isin(np.abs(products[:, :, 5]), np.abs(invisible_pdgs))
        daughters = products[:, :, :4].copy()
        daughters[~present] = 0.
        yield DecayBatch(mother, daughters, visible, np.full(n, weight), mass,
                         charges=charges_from_pdg(products[:,:,5],visible=visible))
