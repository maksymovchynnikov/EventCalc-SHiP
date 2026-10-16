"""Integrate production variations before extracting both mixing boundaries."""
from dataclasses import replace
import numpy as np
from eventcalc.sensitivity import DecayBatch,Geometry,ResponseAccumulator,prepare,threshold_intervals
from eventcalc.production.dark_photon.uncertainties import FORM_FACTORS,pdf_envelope


def integrate_variations(batches,sampler,ctaus,selections,*,seed,nodes=8192):
    """Share decays and accepted paths across variations; verify nominal response."""
    geometry=Geometry();rng=np.random.default_rng(seed+73619)
    responses={};exact=np.zeros((len(ctaus),len(selections)))
    for b in batches:
        daughters,visible,charges,br=sampler(b.mass,len(b.weights),rng)
        br=np.asarray(br,float)
        if br.shape!=b.weights.shape or not np.all(np.isfinite(br)):
            raise ValueError('Expected one finite branching weight per decay')
        accepted=prepare(DecayBatch(b.momenta,daughters,visible,b.weights*br,
                         b.mass,b.origins,frame='rest',charges=charges),
                         geometry=geometry,selections=selections)
        if b.group not in responses:
            responses[b.group]=ResponseAccumulator(selections,geometry,nodes=nodes,
                                                   variations=b.reweights.shape[1])
        responses[b.group].add(accepted,b.reweights)
        nominal=replace(accepted,weights=accepted.weights*b.reweights[accepted.event_index,0])
        exact+=nominal.evaluate(ctaus)
    return exact,{group:response.evaluate(ctaus) for group,response in responses.items()}


def combine_envelopes(central,fixed,brem,dy,labels):
    """The envelope uses accepted yields, never inclusive-flux ratios."""
    lows=[];highs=[]
    for profile in dy:
        low,high=pdf_envelope(profile,labels);lows.append(low);highs.append(high)
    lower=np.maximum(0,fixed+brem.min(axis=0)+np.min(lows,axis=0))
    upper=fixed+brem.max(axis=0)+np.max(highs,axis=0)
    return np.minimum(central,lower),np.maximum(central,upper)


def boundary_bands(parameters,central,lower,upper,threshold=2.3):
    """Keep all components and explicitly identify unresolved/open boundaries.

For a single closed component, the lower-mixing range is [upper-yield lower
crossing, lower-yield lower crossing], and the upper-mixing range is the reverse.
If the lower-yield curve loses reach or components split, report the interval
sets without inventing a correspondence. None means outside the scanned grid.
"""
    central=np.asarray(central);lower=np.asarray(lower);upper=np.asarray(upper)
    if np.any(lower>central) or np.any(upper<central):
        raise ValueError('Production envelopes must contain the nominal yield')
    sets={name:threshold_intervals(parameters,y,threshold) for name,y in
          [('central',central),('lower_yield',lower),('upper_yield',upper)]}
    x=np.asarray(parameters)
    def limits(interval):return (x[0] if interval[0] is None else interval[0],
                                x[-1] if interval[1] is None else interval[1])
    components=[]
    for outer in sets['upper_yield']:
        lo,hi=limits(outer)
        overlap=lambda interval:limits(interval)[1]>=lo and limits(interval)[0]<=hi
        nominal=[v for v in sets['central'] if overlap(v)]
        inner=[v for v in sets['lower_yield'] if overlap(v)]
        single=len(nominal)==len(inner)==1
        components.append({'upper_yield_interval':outer,'central_intervals':nominal,
            'lower_yield_intervals':inner,
            'lower_epsilon2_boundary_range':[outer[0],inner[0][0]] if single else None,
            'upper_epsilon2_boundary_range':[inner[0][1],outer[1]] if single else None,
            'topology':'matched_component' if single else 'reach_absent_or_components_split'})
    return {**{name+'_intervals':intervals for name,intervals in sets.items()},
            'components':components}


def coherent_intervals(parameters,fixed,brem,dy,labels,endpoints,selections,threshold):
    """Threshold both branches for every fixed form-factor/PDF/scale choice."""
    result=[]
    for ib,ff in enumerate(FORM_FACTORS):
        for it,end in enumerate(endpoints):
            for ip,label in enumerate(labels):
                yields=fixed+brem[ib]+dy[it,ip]
                result.append({'form_factors':ff,'transition_endpoint_GeV':end,
                    'pdf':label,'intervals':{s.name:threshold_intervals(parameters,yields[:,i],threshold)
                                            for i,s in enumerate(selections)}})
    return result
