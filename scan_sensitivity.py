"""One mass per job; scan arbitrary lifetime/production laws without event files.

Run `python scan_sensitivity.py --help` and see SENSITIVITY.md for the card format.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path
import time
import numpy as np
from sensitivity import (DecayBatch, Geometry, Selection, ship_selections,
                         scan_replicas, threshold_intervals)


def run(card, *, base=Path('.')):
    parameters = np.asarray(card['parameters'], float)
    ctaus = np.asarray(card['ctau_m'], float)
    scales = np.asarray(card['production_scales'], float)
    if parameters.shape != ctaus.shape or parameters.ndim != 1 or len(parameters) < 2 or np.any(~np.isfinite(parameters)) or np.any(parameters <= 0) or np.any(np.diff(parameters) <= 0):
        raise ValueError('parameters must be a strictly increasing positive grid matching ctau_m')
    geometry = Geometry(**card['geometry']) if 'geometry' in card else Geometry.ship()
    selections = tuple(Selection(**s) for s in card['selections']) if 'selections' in card else ship_selections()
    source = card['source']
    if source['kind'] == 'provider':
        # A provider is ordinary user-supplied Python, just like a simulation
        # script. It must return independently normalized DecayBatch streams.
        module, name = source['callable'].split(':')
        provider = getattr(importlib.import_module(module), name)
        factory = lambda i: provider(replica=i, **source.get('kwargs', {}))
    elif source['kind'] == 'runtime':
        if not Path(__file__).with_name('runtime_generator.py').exists():
            raise ValueError('This historical EventCalc copy has no RuntimeModelScan; use the provider/npz source or model_sensitivity.py.')
        from funcs.simulation_config import config_from_mapping, load_card
        from runtime_generator import RuntimeModelScan
        from sensitivity_runtime import runtime_batches
        path = (base/source['card']).resolve()
        config = config_from_mapping(load_card(path), project_root=Path(__file__).resolve().parent)
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
                        charges=z['charges'][sl] if 'charges' in z else None)
    else:
        raise ValueError('source.kind must be runtime, provider or npz')
    start = time.perf_counter()
    events, error = scan_replicas(factory, ctaus, replicas=card.get('replicas', 2),
        geometry=geometry, selections=selections, exposure=float(card['exposure']),
        production_scales=scales, response_nodes=card.get('response_nodes',8192))
    threshold = float(card.get('event_threshold', 2.3))
    intervals = {s.name: threshold_intervals(parameters, events[:, j], threshold)
                 for j, s in enumerate(selections)}
    metadata = dict(schema_version=2, response_nodes=card.get('response_nodes',8192), card=card, seconds=time.perf_counter()-start,
        geometry=asdict(geometry), selections=[asdict(s) for s in selections],
        threshold_intervals=intervals, reconstruction_efficiency=1.,
        error_scope='Standard error across independent integration replicas; excludes model and finite source-pool uncertainties.',
        units={'mass':'GeV', 'ctau':'m', 'energy':'GeV', 'events':'expected count'},
        code_sha256={p: hashlib.sha256(Path(__file__).with_name(p).read_bytes()).hexdigest()
                     for p in ('sensitivity.py', 'sensitivity_runtime.py', 'scan_sensitivity.py')})
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
