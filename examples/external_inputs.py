"""Generate and run eight small external-input examples, without iteration scripts.

python -m examples.external_inputs --output example-results
"""
import argparse
import json
from pathlib import Path
import numpy as np
from eventcalc.production.external import ExternalProduction
from eventcalc.scan_sensitivity import run


def create(directory, *, events=512):
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=False)
    decay=dict(model='Dark-photons',masses=[.3],c_taus=[1.],events=events,
               decay_channels=['e-pair'],exhad='off',plots=False,export_events=False)
    (directory/'decay.json').write_text(json.dumps(decay,indent=2)+'\n')
    cards=[]
    frames={'ship':'target front at z=0','na62':'nominal NA62 target at z=0',
            'bebc':'BEBC dump end at z=0','charm':'CHARM target reference at z=0'}
    for experiment,origin in [('ship',[0,0,0]),('na62',[0,-.022,23.07]),
                              ('bebc',[0,0,0]),('charm',[0,0,0])]:
        prefix=experiment+'_'
        meta=dict(schema_version=1,kind='spectrum',mass_GeV=.3,arrays=prefix+'spectrum.npz',
            frame='lab',momentum_order=['px','py','pz','E'],energy_unit='GeV',length_unit='m',angle_unit='rad',
            coordinate_frame='beam +z; '+frames[experiment],
            normalization='produced_parents_per_incident',origin_m=origin,production_probability=1e-7)
        manifest=directory/(prefix+'spectrum.json')
        manifest.write_text(json.dumps(meta,indent=2)+'\n')
        np.savez(directory/meta['arrays'],theta_edges=([.008,.010,.012] if experiment=='charm' else [.001,.002,.003]),energy_edges=[60.,90.,120.],
                 bin_probabilities=[[.2,.3],[.3,.2]])
        spectrum=ExternalProduction(manifest,mass=.3)
        replicas=[]
        for replica in range(2):
            batch=list(spectrum.batches(events=events,seed=17+replica,chunk_size=events))[0]
            name=prefix+f'parents_{replica}'
            np.savez(directory/(name+'.npz'),momenta=batch.momenta,origins=batch.origins,weights=batch.weights)
            parent_meta={k:v for k,v in meta.items() if k not in ('production_probability','origin_m')}
            parent_meta.update(kind='parents',arrays=name+'.npz')
            (directory/(name+'.json')).write_text(json.dumps(parent_meta,indent=2)+'\n')
            replicas.append(name+'.json')
        for kind in ('spectrum','parents'):
            source=dict(kind='external_'+kind,decay_card='decay.json',events=events,seed=17,chunk_size=events)
            source.update({'manifest':manifest.name} if kind=='spectrum' else {'replicas':replicas})
            card=dict(mass_GeV=.3,experiment={'name':experiment},parameters=[1.,10.,100.],ctau_m=[1.,10.,100.],
                production_scales=[1.,1.,1.],exposure=1e9,replicas=2,event_threshold=2.302585092994046,
                source=source,response_nodes=None)
            path=directory/(prefix+kind+'_scan.json')
            path.write_text(json.dumps(card,indent=2)+'\n');cards.append(path)
    return cards


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--events',type=int,default=512)
    parser.add_argument('--inputs-only',action='store_true')
    args=parser.parse_args()
    cards=create(args.output,events=args.events)
    if not args.inputs_only:
        for path in cards:
            arrays,metadata=run(json.loads(path.read_text()),base=path.parent)
            output=path.with_suffix('');output.mkdir()
            np.savez_compressed(output/'events.npz',**arrays)
            (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
            print(path.stem, 'maximum expected events:',float(arrays['events'].max()),flush=True)


if __name__=='__main__':main()
