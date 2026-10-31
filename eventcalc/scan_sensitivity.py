"""One mass per job; scan arbitrary lifetime/production laws without event files.

Run `python -m eventcalc.scan_sensitivity --help` and see SENSITIVITY.md for the card format.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import time
import numpy as np
from eventcalc.sensitivity import (DecayBatch, Geometry, Selection, ship_selections,
                         scan_replicas, threshold_intervals)


def run(card, *, base=Path('.')):
    parameters = np.asarray(card['parameters'], float)
    ctaus = np.asarray(card['ctau_m'], float)
    scales = np.asarray(card['production_scales'], float)
    if parameters.shape != ctaus.shape or parameters.ndim != 1 or len(parameters) < 2 or np.any(~np.isfinite(parameters)) or np.any(parameters <= 0) or np.any(np.diff(parameters) <= 0):
        raise ValueError('parameters must be a strictly increasing positive grid matching ctau_m')
    from eventcalc.experiments.ship import ShipResponse
    experiment_card=card.get('experiment', {'name':'ship'})
    experiment_name=experiment_card['name'].lower()
    if experiment_name=='ship':
        geometry = Geometry(**experiment_card.get('geometry',card.get('geometry',{}))) if ('geometry' in card or 'geometry' in experiment_card) else Geometry.ship()
        selected=experiment_card.get('selections',card.get('selections'))
        selections=tuple(Selection(**s) for s in selected) if selected is not None else ship_selections()
        experiment=ShipResponse(geometry,selections)
    elif experiment_name=='na62':
        if 'geometry' in card or 'selections' in card:
            raise ValueError('NA62 uses its named frozen response configuration')
        from eventcalc.experiments.na62 import NA62Response, NA62Config
        experiment=NA62Response(NA62Config(**experiment_card.get('config',{})))
        geometry=experiment.geometry;selections=experiment.selections
    elif experiment_name in ('bebc','charm'):
        if 'geometry' in card or 'selections' in card:
            raise ValueError('Configure BEBC/CHARM through experiment.config')
        if experiment_name=='bebc':
            from eventcalc.experiments.bebc import BEBCResponse,BEBCConfig
            experiment=BEBCResponse(BEBCConfig(**experiment_card.get('config',{})))
        else:
            from eventcalc.experiments.charm import CHARMResponse,CHARMConfig
            experiment=CHARMResponse(CHARMConfig(**experiment_card.get('config',{})))
        geometry=experiment.geometry;selections=experiment.selections
    else: raise ValueError('experiment.name must be ship, na62, bebc or charm')
    input_hashes={}
    source_records=[]
    replica_seeds=[]
    source = card['source']
    if source['kind'] in ('external_spectrum','external_parents'):
        from eventcalc.production.external import ExternalProduction
        from eventcalc.decay_runtime import RuntimeDecayAdapter
        from funcs.simulation_config import config_from_mapping,load_card
        paths=source.get('replicas')
        count=card.get('replicas',2)
        if not isinstance(count,int) or isinstance(count,bool) or count<2:
            raise ValueError('At least two independent replicas required')
        if paths is None:
            if source['kind']=='external_parents':
                raise ValueError('Sampled parents need one independently sampled manifest per replica')
            paths=[source['manifest']]*count
        if len(paths)!=count:
            raise ValueError('Provide one manifest per replica')
        sources=[ExternalProduction(base/p,mass=float(card['mass_GeV'])) for p in paths]
        expected='spectrum' if source['kind']=='external_spectrum' else 'parents'
        if any(s.metadata['kind']!=expected for s in sources):
            raise ValueError('External manifest kind does not match scan source kind')
        if expected=='parents' and len({s.hashes[str(s.array_path)] for s in sources})!=count:
            raise ValueError('Parent replicas must contain distinct independently sampled arrays')
        decay_path=(base/source['decay_card']).resolve()
        input_hashes[str(decay_path)]=hashlib.sha256(decay_path.read_bytes()).hexdigest()
        config=config_from_mapping(load_card(decay_path),project_root=Path(__file__).resolve().parents[1],check_files=False)
        for s in sources:
            input_hashes.update(s.hashes);source_records.append(s.metadata)
        def factory(i):
            seed=int(source.get('seed',1))+i
            replica_seeds.append(seed)
            adapter=RuntimeDecayAdapter(config,mass=float(source.get('decay_mass_GeV',card['mass_GeV'])),seed=seed,
                invisible_pdgs=source.get('invisible_pdgs',[12,14,16,18,4900101]))
            input_hashes.update(adapter.input_hashes)
            return adapter.batches(sources[i].batches(events=source.get('events',131072),seed=seed,
                chunk_size=source.get('chunk_size',4096)))
    elif source['kind'] == 'provider':
        # A provider is ordinary user-supplied Python, just like a simulation
        # script. It must return independently normalized DecayBatch streams.
        module, name = source['callable'].split(':')
        provider = getattr(importlib.import_module(module), name)
        factory = lambda i: provider(replica=i, **source.get('kwargs', {}))
    elif source['kind'] == 'runtime':
        if not Path(__file__).with_name('runtime_generator.py').exists():
            raise ValueError('This historical EventCalc copy has no RuntimeModelScan; use the provider/npz source or model_sensitivity.py.')
        from funcs.simulation_config import config_from_mapping, load_card
        from eventcalc.runtime_generator import RuntimeModelScan
        from eventcalc.sensitivity_runtime import runtime_batches
        path = (base/source['card']).resolve()
        input_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        config = config_from_mapping(load_card(path), project_root=Path(__file__).resolve().parents[1])
        scan = RuntimeModelScan(config)
        factory = lambda i: runtime_batches(scan, float(card['mass_GeV']),
            events=int(source.get('events', 131072)), seed=int(source.get('seed', 1))+i,
            chunk_size=int(source.get('chunk_size', 4096)),
            invisible_pdgs=source.get('invisible_pdgs', [12, 14, 16]),
            yield_per_incident=source.get('yield_per_incident'))
    elif source['kind'] == 'npz':
        files = [(base/f).resolve() for f in source['replicas']]
        if len(files) != int(card.get('replicas', 2)) or len(set(files)) != len(files):
            raise ValueError('Provide one distinct, independently sampled NPZ file per replica')
        input_hashes.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        def factory(i):
            with np.load(files[i], allow_pickle=False) as archive:
                # Decompress each array once, not once per processing chunk.
                z = {k: archive[k] for k in archive.files}
                n = len(z['mother']); chunk = int(source.get('chunk_size', 4096))
                if chunk <= 0: raise ValueError('chunk_size must be positive')
                for start in range(0, n, chunk):
                    sl = slice(start, start+chunk)
                    yield DecayBatch(z['mother'][sl], z['daughters'][sl], z['visible'][sl],
                        z['weights'][sl], float(z['mass']),
                        z['origins'][sl] if 'origins' in z else None,
                        str(z['frame']) if 'frame' in z else 'lab',
                        charges=z['charges'][sl] if 'charges' in z else None,
                        pdg_ids=z['pdg_ids'][sl] if 'pdg_ids' in z else None)
    else:
        raise ValueError('Unknown source.kind; use runtime, provider, npz, external_spectrum or external_parents')
    start = time.perf_counter()
    events, error = scan_replicas(factory, ctaus, replicas=card.get('replicas', 2),
        experiment=experiment, exposure=float(card['exposure']),
        production_scales=scales, response_nodes=card.get('response_nodes',8192))
    threshold = float(card.get('event_threshold', 2.3))
    intervals = {s.name: threshold_intervals(parameters, events[:, j], threshold)
                 for j, s in enumerate(selections)}
    metadata = dict(schema_version=3, response_nodes=card.get('response_nodes',8192), card=card, seconds=time.perf_counter()-start,
        experiment=experiment_card, external_sources=source_records, input_sha256=input_hashes,
        geometry=asdict(geometry) if geometry is not None else None, selections=[asdict(s) for s in selections],
        threshold_intervals=intervals, reconstruction_efficiency=1.,
        error_scope='Standard error across independent integration replicas; excludes model and finite source-pool uncertainties.',
        units={'mass':'GeV', 'ctau':'m', 'energy':'GeV', 'events':'expected count'},
        code_sha256={p: hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
                     for p in ('sensitivity.py', 'sensitivity_runtime.py', 'scan_sensitivity.py')})
    metadata['replica_seeds']=replica_seeds
    if experiment_name=='na62':
        metadata['resolved_experiment_config']=asdict(experiment.config)
        metadata['response_scope']='Frozen truth-level lepton response and ideal-response hadronic projection; not full detector validation or an observed exclusion.'
    if experiment_name in ('bebc','charm'):
        metadata['resolved_experiment_config']=asdict(experiment.config)
        metadata['response_scope']='Phenomenological two-track lepton acceptance with explicit geometry and efficiency assumptions; not a full historical detector response or observed-limit construction.'
    package=Path(__file__).resolve().parent
    metadata['code_sha256'].update({str(p.relative_to(package.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*package.joinpath('experiments').rglob('*.py'),*package.joinpath('experiments/data').glob('*.json'),
                  package/'production/external.py',package/'decay_runtime.py',package/'transport.py',
                  *(package.parent/'funcs'/p for p in ('decay_tables.py','initLLP.py','HNLmerging.py','inelastic_llp.py')),
                  package.parent/'simulate.py'] if p.is_file()})
    return dict(parameters=parameters, ctau_m=ctaus, production_scales=scales,
                events=events, integration_error_events=error), metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('card', type=Path)
    parser.add_argument('--output', required=True, type=Path,
                        help='New directory for compact grids and metadata; never overwritten')
    args = parser.parse_args()
    card = json.loads(args.card.read_text())
    if args.output.exists(): parser.error('output directory already exists')
    arrays, metadata = run(card, base=args.card.parent)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/'events.npz', **arrays)
    metadata['events_sha256'] = hashlib.sha256((args.output/'events.npz').read_bytes()).hexdigest()
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({'output':str(args.output), 'seconds':metadata['seconds'],
                      'max_events':float(arrays['events'].max())}))


if __name__ == '__main__':
    main()
