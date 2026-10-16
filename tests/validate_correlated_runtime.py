"""Installed-data integration check; writes its report only when asked."""
from dataclasses import replace
from pathlib import Path
import argparse
import json
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from funcs.simulation_config import config_from_mapping
from eventcalc.runtime_generator import RuntimeModelScan


def validate():
    reports=[]
    for model,mass,delta in [("Dark-photons",1.8,.1),("Inelastic-DM",20.,.1),("Inelastic-DM",20.,.05)]:
        config=config_from_mapping(dict(model=model,masses=[mass],c_taus=[1.,100.],events=64,
            idm_delta=delta,production_power=11,seed=12016,
            decay_channels=["e-pair"] if model=="Dark-photons" else ["all"],
            exhad_mode="off" if model=="Dark-photons" else "auto"))
        scan=RuntimeModelScan(config);start=time.perf_counter()
        first=scan.generator(mass=mass,c_tau=1.,seed=1);first.init()
        init_seconds=time.perf_counter()-start
        source=first._collision_source
        assert first._grid is None
        assert np.all((source.origins[:,2]>=0)&(source.origins[:,2]<=1.46))
        assert np.any(source.origins[:,:2]!=0)
        for ctau in (1.,100.):
            gen=scan.generator(mass=mass,c_tau=ctau,seed=2);gen.init()
            assert gen._collision_source is source
            p=np.linalg.norm(source.momenta[:,:3],axis=1)
            rate=source.mass/(ctau*p)
            prob=np.exp(-rate*source.low)*(-np.expm1(-rate*(source.high-source.low)))
            exact=config.n_pot*gen._coupling_squared*gen.visible_branching_ratio*np.dot(source.weights,prob)
            start=time.perf_counter();sample=gen._parents_for_output(100000);seconds=time.perf_counter()-start
            estimate=gen.yields()
            tolerance=6*estimate.expected_events_error+1e-12*max(exact,1.)
            assert abs(estimate.expected_events-exact)<tolerance,(model,ctau,estimate,exact)
            delta_vertex=sample.mother[:,7:10]-sample.origins
            cross=np.cross(delta_vertex,sample.momentum[:,:3])
            assert np.max(abs(cross))<1e-9
            assert gen.experiment.contains(*sample.mother[:,7:10].T).all()
            assert np.max(abs(sample.mother[:,3]**2-(sample.mother[:,:3]**2).sum(axis=1)-source.mass**2))<1e-5
            reports.append(dict(model=model,mass=mass,delta=delta,ctau=ctau,
                source_rows=len(source.weights),init_seconds=init_seconds,
                parents_100k_seconds=seconds,exact_yield=float(exact),
                estimated_yield=estimate.expected_events,mc_error=estimate.expected_events_error))
        # Generate and boost real decay products; missing chi would fail conservation.
        gen=scan.generator(mass=mass,c_tau=10.,seed=8)
        batch=next(gen.generate(64,batch_size=64));products=batch.records[:,10:].reshape(64,-1,6)
        present=products[:,:,5]!=-999
        summed=np.where(present[:,:,None],products[:,:,:4],0.).sum(axis=1)
        np.testing.assert_allclose(summed,batch.records[:,:4],rtol=2e-8,atol=2e-7)
        assert batch.production_origins.shape==(64,3)
        assert batch.event(0).production_origin.shape==(3,)
        if model=="Inelastic-DM":
            assert np.all(batch.records[:,5]==4900102)
            assert np.all(np.any(products[:,:,5]==4900101,axis=1))
        # Changing batch partition must not change parent or collision samples.
        gen=scan.generator(mass=mass,c_tau=10.,seed=8);gen.init()
        a=gen._parents_for_output(31);b=gen._parents_for_output(33)
        np.testing.assert_array_equal(np.concatenate([a.mother,b.mother]),batch.records[:,:10])
        np.testing.assert_array_equal(np.concatenate([a.origins,b.origins]),batch.production_origins)
    return reports


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args()
    report=validate();payload=json.dumps(report,indent=2,allow_nan=False)
    if a.output:a.output.write_text(payload+'\n')
    print(payload)
