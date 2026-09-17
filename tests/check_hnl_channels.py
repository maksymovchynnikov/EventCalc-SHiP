#!/usr/bin/env python3
"""Exercise all active HNL currents with the installed matrix elements."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from funcs.initLLP import LLP
from funcs import decayProducts, exhadDecays
from funcs import exhad_integration as integration
from funcs.simulation_config import config_from_mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exhad-root', required=True)
    parser.add_argument('--events', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = []
    for flavor, mixing in [('e',[1.,0.,0.]), ('mu',[0.,1.,0.]), ('tau',[0.,0.,1.])]:
        mass = 5.2
        llp = LLP(mass, dict(LLP_name='HNL', particle_path=str(ROOT/'Distributions/HNL')),
                  mixing_pattern=mixing)
        config = config_from_mapping(dict(model='HNL', masses=[mass], c_taus=[10.],
            mixing_pattern=mixing, decay_channels=['all'], events=args.events,
            exhad_root=args.exhad_root), project_root=ROOT)
        integration.configure_llp(llp, config)
        branching = llp.get_Br(mass)
        matrix = llp.get_MatrixElements(mass)
        original_branching = branching.copy()
        lifetime = llp.get_ctau(mass)
        labels = [f'Jets-{pair}{flavor}{sign}' for pair in ['ud','us','cd','cs']
                  for sign in ['', 'bar']] + ['Jets-uuv','Jets-ddv','Jets-ssv']
        for label in labels:
            index = list(llp.decayChannels).index(label)
            assert branching[index] > 0, label
            start = time.perf_counter()
            rows, sizes, _labels = integration.simulate_decays(llp, decayProducts, mass,
                llp.PDGs, branching, args.events, matrix, [index],
                float(branching[index]), seed=20260912)
            assert list(sizes) == [args.events]
            assert len(rows) == args.events
            for row in rows:
                parts = np.asarray(row).reshape(-1,6)
                parts = parts[parts[:,5] != -999]
                np.testing.assert_allclose(parts[:,:4].sum(axis=0), [0.,0.,0.,mass], atol=2e-7)
                assert np.all(np.isfinite(parts))
                assert not any(1 <= abs(int(pid)) <= 6 for pid in parts[:,5])
            np.testing.assert_array_equal(branching, original_branching)
            assert llp.get_ctau(mass) == lifetime
            result = dict(mixing=flavor, mass_gev=mass, channel=label, events=len(rows),
                          seconds=time.perf_counter()-start, branching=float(branching[index]))
            results.append(result)
            print(json.dumps(result), flush=True)
    exhadDecays.close_generators()
    with args.output.open('x') as stream:
        json.dump(results, stream, indent=2)


if __name__ == '__main__':
    main()
