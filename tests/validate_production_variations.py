"""Check production columns against separately evaluated nominal/form-factor folds.

Run from the repository root. No exHad generation or shower regeneration.
"""
import json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eventcalc.production.dark_photon import ProductionModel
from eventcalc.production.dark_photon.uncertainties import ProductionVariations,FORM_FACTORS


def run():
    start=time.perf_counter();production=ProductionModel();variations=ProductionVariations(production)
    report=[]
    for model,mass in [('dp',1.1),('dp',2.),('idm',1.1),('idm',2.8),('idm',20.)]:
        nominal=0.;brem=np.zeros(3);dy=np.zeros(len(variations.endpoints(model,mass,.1)))
        for batch in variations.batches(mass,model=model,power=10,seed=453):
            nominal+=np.dot(batch.weights,batch.reweights[:,0])
            if batch.group=='brem':brem+=batch.weights@batch.reweights
            elif batch.group=='dy':dy+=batch.weights@batch.reweights[:,::len(variations.pdf.labels)]
        production.inputs.set_formfactors('central')
        direct=sum(b.weights.sum() for b in production.batches(mass,model=model,power=10,seed=453))
        np.testing.assert_allclose(nominal,direct,rtol=3e-13,atol=1e-100)
        for i,ff in enumerate(FORM_FACTORS):
            production.inputs.set_formfactors(ff)
            expected=sum(b.weights.sum() for b in production.batches(mass,model=model,power=10,seed=453,channels=['brem']))
            np.testing.assert_allclose(brem[i],expected,rtol=3e-13,atol=1e-100)
        if len(dy)==3:
            # Earlier onset cannot reduce a physical DP Born yield (iDM uses
            # signed NLO event weights, so do not impose this as a general law).
            if model=='dp':assert dy[1]>=dy[0]>=dy[2]
        report.append({'model':model,'mass_GeV':mass,'nominal_relative_error':abs(nominal-direct)/max(abs(direct),1e-100),
                       'brem_form_factor_checks':'passed','dy_transition_yields':dy.tolist()})
    print(json.dumps({'passed':True,'cases':report,'seconds':time.perf_counter()-start},indent=2))


if __name__=='__main__':run()
