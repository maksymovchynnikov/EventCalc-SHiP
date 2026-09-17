#!/usr/bin/env python3
"""End-to-end rest-frame generation, boosts, labels, and yield smoke test."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from runtime_generator import RuntimeEventGenerator
from funcs import decayProducts
from funcs.simulation_config import config_from_mapping
from funcs.exhadDecays import close_generators


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exhad-root', required=True)
    parser.add_argument('--models', nargs='+', default=['ALP-fermion', 'Scalar-mixing', 'Dark-photons'])
    parser.add_argument('--events', type=int, default=40)
    parser.add_argument('--mass', type=float, default=2.)
    parser.add_argument('--mixing', type=float, nargs=3, default=[1.,0.,0.])
    parser.add_argument('--channels', nargs='+', default=['all'])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    results = {}
    for model in args.models:
        values = dict(model=model, events=args.events, masses=[args.mass], c_taus=[10.],
                      decay_channels=args.channels, seed=23456,
                      exhad_root=args.exhad_root)
        if model == 'Dark-photons':
            values['uncertainty'] = 'central'
        if model == 'HNL':
            values['mixing_pattern'] = args.mixing
        config = config_from_mapping(values, project_root=ROOT)
        generator = RuntimeEventGenerator(config, mass=args.mass, c_tau=10.)
        start = time.perf_counter()
        generator.init()
        initialized = time.perf_counter()
        batch = next(generator.generate(args.events, batch_size=args.events))
        elapsed = time.perf_counter()-initialized
        assert len(batch) == args.events
        residuals = []
        for i in range(len(batch)):
            event = batch.event(i)
            residual = np.max(np.abs(event.daughters[:, :4].sum(axis=0)-event.mother[:4]))/event.mother[3]
            assert residual < 2e-6, residual
            assert np.all(np.isfinite(event.daughters))
            residuals.append(float(residual))
        result = dict(mass=args.mass, mixing=args.mixing if model == 'HNL' else None,
                      init_seconds=initialized-start, batch_seconds=elapsed,
                      events=len(batch), max_relative_lab_p4_residual=max(residuals),
                      channels=sorted(set(batch.channels.tolist())), yields=generator.yields().as_dict())
        if model != 'HNL':
            assert any(name.startswith(decayProducts.MATCHED_PROCESS_LABEL)
                       for name in result['channels'])
        results[model] = result
        print(json.dumps({model: result}), flush=True)
    close_generators()
    if args.output:
        with args.output.open('x') as stream:
            json.dump(results, stream, indent=2)
            stream.write('\n')


if __name__ == '__main__':
    main()
