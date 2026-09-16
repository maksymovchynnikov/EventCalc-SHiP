"""Explicit exHad release binding; no physics aliasing or raw fallback on failure."""
from __future__ import annotations
import atexit
import bisect
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

_GENERATORS = {}
MODELS = {"Dark-photons": ("dark-photon", 1.7),
          "HNL": ("hnl", .02),
          "ALP-fermion": ("alp-fermion", 1.911),
          "Scalar-mixing": ("scalar", 2.0), "Scalar-quartic": ("scalar", 2.0)}


def close_generators():
    for generator in _GENERATORS.values():
        generator.close()
    _GENERATORS.clear()


atexit.register(close_generators)


def release(root):
    """The public exhad package of one configured release, imported once per root."""
    name = '_eventcalc_exhad_' + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    module = sys.modules.get(name)
    if module is None:
        module_path = Path(root) / 'exhad/__init__.py'
        spec = importlib.util.spec_from_file_location(name, module_path,
            submodule_search_locations=[str(module_path.parent)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return module


def model_info(root, model):
    """Matched support, EventCalc rate tables, row ownership and version of a release model."""
    module = release(root)
    if not hasattr(module, 'model_info'):
        raise ValueError('This exHad release has no model_info host interface; install a newer release')
    return module.model_info(model)


def _generator(binding):
    cpu_count = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
    workers = int(os.environ.get('EXHAD_WORKERS', str(min(8, cpu_count))))
    chunk_size = int(os.environ.get('EXHAD_CHUNK_SIZE', '512'))
    if not 1 <= workers <= 64 or chunk_size < 1:
        raise ValueError('EXHAD_WORKERS must be 1..64; EXHAD_CHUNK_SIZE must be positive')
    key = (binding['root'], binding['python'], binding['model'], workers, chunk_size)
    if key not in _GENERATORS:
        _GENERATORS[key] = release(binding['root']).ParallelGenerator(binding['model'],
            workers=workers, chunk_size=chunk_size, root=binding['root'], python=binding['python'])
    generator = _GENERATORS[key]
    binding['execution'] = dict(workers=generator.workers, chunk_size=generator.chunk_size,
                                seed_stream='chunked-v1')
    return generator


def configure_llp(llp, config):
    llp._exhad_binding = None
    llp._exhad_pooled_indices = ()
    if config.hadronization == 'raw':
        return
    if llp.LLP_name not in MODELS:
        raise ValueError(f'No exHad release model for {llp.LLP_name}')
    raw_root = config.exhad_root or os.environ.get('EXHAD_ROOT')
    if not raw_root:
        raise ValueError('Set exhad_root in the card or EXHAD_ROOT to the configured release')
    root = Path(raw_root).expanduser().resolve()
    model, start = MODELS[llp.LLP_name]
    info = model_info(root, model)
    support, tables = info['support_gev'], info['tables']
    if (len(support) != 2 or not all(math.isfinite(float(x)) for x in support)
            or float(support[0]) != start or float(support[1]) < start):
        raise ValueError('exHad deployment has incompatible mass support')
    binding = {'root': str(root), 'python': config.exhad_python or sys.executable,
               'model': model, 'version': info['version'], 'start': start, 'end': float(support[1])}
    if model == 'dark-photon':
        _install_rows(llp, json.loads(Path(tables['decay']).read_text()))
        llp.get_Br = llp.setup_br_interpolators(llp.BrRatios)
        lifetime = np.loadtxt(tables['ctau'])
        interpolator = RegularGridInterpolator((lifetime[:, 0],), lifetime[:, 1])
        llp.get_ctau = lambda m: interpolator([m])[0]
    if model == 'scalar':
        # Adopt the published Blackstone-central rates and their documented
        # continuation together; do not apply this generator to an old table.
        from funcs.scalar_hadronic_continuation import load_continuation
        _install_rows(llp, json.loads(Path(tables['decay']).read_text()))
        scalar_authority = load_continuation(tables['decay'], tables['ctau'])
        llp.get_Br = lambda m: np.array([scalar_authority.at(m)[label] for label in llp.decayChannels])
        llp.get_ctau = lambda m: 1.973269804e-16 / scalar_authority.total_width(m)
    if model == 'hnl':
        installed = Path(llp.particle_path)
        if (json.loads(Path(tables['decay']).read_text()) != json.loads((installed / 'HNL-decay.json').read_text())
                or not np.array_equal(np.loadtxt(tables['widths']), np.loadtxt(installed / 'HNLdecayWidth.dat'))):
            raise ValueError('EventCalc HNL decay and width inputs must match the exHad release')
    elif model != 'dark-photon':
        binding['signatures'] = {tuple(row['pdg_signature']): row['authority_row_id']
                                 for row in info['eventcalc_rows']}
        binding['owner_probabilities'] = info['owner_probabilities']
    llp._exhad_binding = binding


def metadata(llp):
    binding = getattr(llp, '_exhad_binding', None)
    if binding is None:
        return {'backend': 'raw'}
    result = {'backend': 'exhad', 'model': binding['model'],
            'release_root': binding['root'], 'release_version': binding['version'],
            'matched_support_gev': [binding['start'], binding['end']],
            'event_count_scope': 'complete selected hadronic row pool'}
    if 'execution' in binding:
        result['execution'] = dict(binding['execution'])
    if binding['model'] == 'hnl':
        result.update(event_count_scope='each signed Jets row separately',
                      composition_coordinate='per-event hadronic invariant mass W',
                      currents=['CC_ud', 'CC_us', 'CC_cd', 'CC_cs', 'NC_ud', 'NC_s'])
    return result


def legacy_output(llp, mothers, daughters, sizes, selected):
    """Group matched events under their truthful pooled label in text exports."""
    pooled = set(getattr(llp, '_exhad_pooled_indices', ()))
    if not pooled:
        return mothers, daughters, llp.decayChannels, sizes, selected
    offset, native, matched, labels, counts = 0, [], [], [], []
    for index, count in zip(selected, sizes):
        count = int(count)
        block = list(range(offset, offset + count))
        offset += count
        if index in pooled:
            matched.extend(block)
        else:
            native.extend(block)
            labels.append(str(llp.decayChannels[index]))
            counts.append(count)
    if matched:
        labels.append('Hadronic-exHad')
        counts.append(len(matched))
    order = native + matched
    return (mothers[order], daughters[order], labels, np.asarray(counts), list(range(len(labels))))


def _install_rows(llp, rows):
    llp.decayChannels = np.array([row[0] for row in rows], dtype=object)
    llp.PDGs = np.empty(len(rows), dtype=object)
    llp.PDGs[:] = [np.array(row[1]) for row in rows]
    llp.BrRatios = np.empty(len(rows), dtype=object)
    llp.BrRatios[:] = [row[2] for row in rows]
    llp.Matrix_elements_raw = [str(row[3]) if len(row) > 3 else '1.' for row in rows]
    llp.Matrix_elements = llp.compile_matrix_elements(llp.Matrix_elements_raw)


def import_fermion_alp(llp):
    from funcs.alp_fermion import build_br_interpolator, prepare_decay_rows
    from funcs.coupling_conventions import load_coupling_metadata
    root = Path(llp.particle_path)
    llp.coupling_metadata = load_coupling_metadata(root)
    llp.Distr = pd.read_csv(root / 'DoubleDistr-ALP-fermion.txt', header=None, sep='\t')
    llp.Energy_distr = pd.read_csv(root / 'Emax-ALP-fermion.txt', header=None, sep='\t')
    llp.Yield_data = pd.read_csv(root / 'Total-yield-ALP-fermion.txt', header=None, sep='\t')
    llp.ctau_data = pd.read_csv(root / 'ctau-ALP-fermion.txt', header=None, sep='\t')
    rows = json.loads((root / 'ALP-fermion-decay.json').read_text())
    prepare_decay_rows(rows)
    _install_rows(llp, rows)
    interpolator = build_br_interpolator(llp.decayChannels, llp.PDGs, llp.BrRatios,
        1.911, 3.741, llp.ctau_data.itertuples(index=False, name=None))
    lifetime = RegularGridInterpolator((llp.ctau_data.iloc[:, 0].to_numpy(),), llp.ctau_data.iloc[:, 1].to_numpy())
    production = RegularGridInterpolator((llp.Yield_data.iloc[:, 0].to_numpy(),), llp.Yield_data.iloc[:, 1].to_numpy())
    llp.get_Br = lambda m: np.asarray(interpolator(m), dtype=float)
    llp.get_ctau = lambda m: lifetime([m])[0]
    llp.get_total_yield = lambda m: production([m])[0]
    llp.get_distribution = lambda m: llp.Distr
    llp.get_MatrixElements = lambda m: llp.Matrix_elements
    llp.define_tabulated_range_nonHNL(llp.Yield_data.iloc[:, 0].to_numpy(),
        llp.Yield_data.iloc[:, 1].to_numpy(), llp.Distr, llp.ctau_data)


def simulate_decays(llp, decay_module, mass, pdgs, branching, size, matrix_elements,
                    selected, visible, *, seed=1, weighted=False, weight_floor_fraction=0.):
    binding = getattr(llp, '_exhad_binding', None)
    if (isinstance(weight_floor_fraction, bool) or not isinstance(weight_floor_fraction,(int,float))
            or not math.isfinite(weight_floor_fraction) or not 0. <= weight_floor_fraction <= 1.):
        raise ValueError('weight_floor_fraction must lie in [0, 1]')
    if not weighted and weight_floor_fraction != 0.:
        raise ValueError('weight_floor_fraction requires weighted sampling')
    if weighted and (binding is None or binding['model'] != 'alp-fermion'):
        raise ValueError('Weighted sampling requires the explicit ALP exHad binding')
    llp._exhad_pooled_indices = ()
    if getattr(llp, 'LLP_name', None) == 'HNL':
        if binding is None:
            return decay_module.simulateDecays_rest_frame(mass, pdgs, branching, size,
                matrix_elements, selected, visible, hnl=True)
        return _simulate_hnl(binding, decay_module, mass, pdgs, branching, size,
                             matrix_elements, selected, visible, seed)
    if binding is None or mass < binding['start']:
        result = decay_module.simulateDecays_rest_frame(mass, pdgs, branching, size,
                                                       matrix_elements, selected, visible)
        if weighted:
            return (*result, {'raw_weights': np.ones(size),
                             'normalization_groups': np.full(size, 'external')})
        return result
    if mass > binding['end']:
        raise ValueError(f"The exHad {binding['model']} deployment is supported only through "
                         f"{binding['end']:g} GeV; select raw explicitly")
    pooled = set()
    for i, row in enumerate(pdgs):
        signature = tuple(sorted(int(p) for p in row if int(p) != -999))
        if binding['model'] == 'dark-photon':
            eligible = len(signature) == 2 and signature[0] == -signature[1] and 1 <= signature[1] <= 4
        else:
            eligible = signature in binding['signatures']
        if eligible:
            pooled.add(i)
    missing = {i for i in pooled - set(selected) if branching[i] > 0}
    if missing:
        raise ValueError('exHad requires the complete hadronic row pool; select all hadronic rows')
    if binding['model'] != 'dark-photon':
        masses = binding['owner_probabilities']['masses']
        columns = binding['owner_probabilities']['probabilities']
        right = bisect.bisect_left(masses, mass)
        if right == 0 or masses[right] == mass:
            expected = {key: column[right] for key, column in columns.items()}
        else:
            fraction = (mass-masses[right-1])/(masses[right]-masses[right-1])
            expected = {key: (1-fraction)*column[right-1] + fraction*column[right]
                        for key, column in columns.items()}
        total = math.fsum(expected.values())
        expected = {key: value/total for key, value in expected.items()}
        observed = {binding['signatures'][tuple(sorted(int(p) for p in pdgs[i] if int(p) != -999))]: float(branching[i])
                    for i in pooled}
        total = math.fsum(observed.values())
        if total > 0 and (set(observed) != set(expected) or any(
                not math.isclose(observed[key]/total, expected[key], rel_tol=2e-8, abs_tol=5e-10)
                for key in expected)):
            raise ValueError('EventCalc hadronic branching ratios disagree with the sealed exHad input')
    sizes = decay_module.distribute_events(size, [branching[i]/visible for i in selected])
    needed = sum(int(n) for i, n in zip(selected, sizes) if i in pooled)
    if needed:
        if weighted:
            weighted_pool = _generator(binding).generate_weighted(float(mass), needed, seed=int(seed),
                weight_floor_fraction=weight_floor_fraction)
            events = weighted_pool['events']
        else:
            events = _generator(binding).generate(float(mass), needed, seed=int(seed))
        pool = [[value for particle in event for value in particle] for event in events]
    else:
        pool = []
    result, cursor = [], 0
    raw_weights, groups = [], []
    for i, n in zip(selected, sizes):
        n = int(n)
        if i in pooled:
            result.extend(pool[cursor:cursor+n])
            if weighted:
                raw_weights.extend(weighted_pool['raw_weights'][cursor:cursor+n] if n else [])
                groups.extend(weighted_pool['normalization_groups'][cursor:cursor+n] if n else [])
            cursor += n
        elif n:
            if weighted:
                raw_weights.extend([1.]*n)
                groups.extend(['external']*n)
            rows, _ = decay_module.simulateDecays_rest_frame(mass, pdgs, branching, n,
                matrix_elements, [i], float(branching[i]))
            for row in rows:
                result.append([value for j in range(0, len(row), 6) if int(row[j+5]) != -999
                               for value in row[j:j+6]])
    if len(result) != size or cursor != needed:
        raise RuntimeError('exHad row pooling lost or duplicated events')
    llp._exhad_pooled_indices = tuple(sorted(pooled))
    if weighted:
        if len(raw_weights) != size or len(groups) != size:
            raise RuntimeError('Importance weights do not align with pooled/native events')
        return decay_module.pad_processed_events(result), sizes, {
            'raw_weights': np.asarray(raw_weights), 'normalization_groups': np.asarray(groups)}
    return decay_module.pad_processed_events(result), sizes


def _simulate_hnl(binding, decay_module, mass, pdgs, branching, size,
                  matrix_elements, selected, visible, seed):
    if not binding['start'] <= mass <= binding['end']:
        raise ValueError('HNL exHad support is 0.02 <= mass <= 5.27 GeV')
    sizes = decay_module.distribute_events(size, [branching[i]/visible for i in selected])
    output = []
    for index, count in zip(selected, sizes):
        count = int(count)
        if not count:
            continue
        ids = [int(p) for p in pdgs[index] if int(p) != -999]
        # Signed currents must remain separate. Explicit meson poles never
        # enter the replacement and their channel weights are not pooled.
        partonic = any(1 <= abs(p) <= 6 or p == 21 for p in ids)
        channel_seed = int(np.random.SeedSequence([int(seed), int(index)]).generate_state(1)[0])
        np.random.seed(channel_seed)
        decay_module.ThreeBodyDecay.seed_random(channel_seed)
        rows, _ = decay_module.simulateDecays_rest_frame(mass, pdgs, branching, count,
            matrix_elements, [index], float(branching[index]), hnl=True, primary_only=partonic)
        if partonic:
            events = _generator(binding).hadronize_hnl(
                float(mass), ids, np.asarray(rows).tolist(), seed=channel_seed)
            output.extend([[value for particle in event for value in particle] for event in events])
        else:
            output.extend([[value for j in range(0, len(row), 6) if int(row[j+5]) != -999
                            for value in row[j:j+6]] for row in rows])
    if len(output) != size:
        raise RuntimeError('HNL current routing lost or duplicated events')
    return decay_module.pad_processed_events(output), sizes
