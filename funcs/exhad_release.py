"""The single binding between EventCalc and a configured exHad release.

This module is the *shared* half of the exHad hook: it is byte-identical in
both EventCalc copies and it owns everything that is physics or release
interface.  The two host drivers keep their own idioms on top of it -- the
card-driven ``simulate.py``/``funcs.simulation_config`` and the
``run_batch.py``/``Distributions/<LLP>/exhad.json`` resolution -- but neither
of them talks to the release directly.

Division of ownership (a user decision, not a default):

* EventCalc owns branching ratios, lifetimes and channel selection;
* exHad supplies final states only;
* both sides use the same default mother PDG codes, overridable per run with
  ``--llp-pdg``;
* EventCalc's *internal* Pythia hadronization mother code 25 is unrelated to
  the LLP PDG code and is left alone.

Release interface
-----------------
``Generator`` generates in parallel by default (one worker per physical core,
at most 8); ``workers=1`` runs in a single process.  Its methods are
``generate``, ``generate_all``, ``generate_rows``, ``generate_weighted``
(alp-fermion only) and ``hadronize_hnl``.

Model names are ``dark-photon``, ``alp-fermion``, ``scalar-central``,
``scalar-lower``, ``scalar-upper``, ``scalar-1809``, ``b-l`` and ``hnl``.

Mass windows
------------
``model_info`` describes three intervals, and each answers a different
question.

``generation_gev`` holds the intervals of *mother mass* -- the mass of the
decaying particle -- in which the release generates decays: one key for the
decays that contain hadrons, and ``"all"`` for the whole decay table.
``"all"`` is the interval this module enforces, because it is the range of
mother mass the release serves.

``exclusive_gev`` is the low-mass part of that range in which each hadronic
decay is one named table row with an exact final state, realized at exact
counts by ``generate_rows``.  The boson portals have such a window; the heavy
neutral lepton has none.

``support_gev`` is the interval of the matched description.  Its lower edge is
a mother mass: at and above it the inclusive matched generator runs.  Its
upper edge bounds *W*, the invariant mass of the hadronic system exHad
hadronizes.  A boson portal decays to hadrons alone, so its W equals its own
mass and the two readings of that edge give the same number.  A heavy neutral
lepton shares its mass with a charged lepton or a neutrino, so its W stays
below its own mass, and above that edge each decay reaches a growing part of
its W range through the generic route while generation continues to the end of
``generation_gev["all"]``.
"""
from __future__ import annotations

import atexit
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

#: EventCalc LLP name -> exHad model.  Scalars resolve to a prescription
#: variant; the host picks it and the default is the published central one.
MODEL_BY_LLP = {
    "Dark-photons": "dark-photon",
    "HNL": "hnl",
    "ALP-fermion": "alp-fermion",
    "Scalar-mixing": "scalar-central",
    "Scalar-quartic": "scalar-central",
    "B-L": "b-l",
    # EventCalc's "ALP-photon" is the photon-coupled ALP, a different
    # benchmark from the fermion-universal exHad ALP; it is not mapped.
}

#: Host prescription name -> exHad scalar model.  ``scalar`` alone names no
#: model; a card that says it is resolved to the central prescription and the
#: resolution is reported, never silently aliased.
SCALAR_VARIANTS = {
    "central": "scalar-central",
    "lower": "scalar-lower",
    "upper": "scalar-upper",
    "1809": "scalar-1809",
}

SCALAR_MODELS = frozenset(SCALAR_VARIANTS.values())

#: Research-copy benchmark key -> exHad model.
MODEL_BY_BENCH = {
    "dv": "dark-photon",
    "alp": "alp-fermion",
    "hls": "scalar-central",
    "hls-central": "scalar-central",
    "hls-lower": "scalar-lower",
    "hls-upper": "scalar-upper",
    "hls-1809": "scalar-1809",
    "bl": "b-l",
    "hnl": "hnl",
}

#: Retired model names -> what to use instead.  Kept only to give a precise
#: error; nothing resolves through it automatically.
RETIRED_MODELS = {
    "scalar": "scalar-central (or -lower/-upper/-1809)",
    "b-l-light": "b-l",
}

KNOWN_MODELS = ("dark-photon", "alp-fermion", "scalar-central", "scalar-lower",
                "scalar-upper", "scalar-1809", "b-l", "hnl")

#: The decaying particle of each model, so that a message quoting a mass names
#: the particle whose mass it is.
MOTHER_NAME = {
    "dark-photon": "dark-photon",
    "alp-fermion": "ALP",
    "scalar-central": "scalar",
    "scalar-lower": "scalar",
    "scalar-upper": "scalar",
    "scalar-1809": "scalar",
    "b-l": "B-L boson",
    "hnl": "HNL",
}

_GENERATORS = {}


def close_generators():
    for generator in _GENERATORS.values():
        generator.close()
    _GENERATORS.clear()


atexit.register(close_generators)


# ---------------------------------------------------------------------------
# Release import
# ---------------------------------------------------------------------------

def release(root):
    """The public exhad package of one configured release, imported once per root."""
    name = '_eventcalc_exhad_' + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    module = sys.modules.get(name)
    if module is None:
        module_path = Path(root) / 'exhad/__init__.py'
        spec = importlib.util.spec_from_file_location(
            name, module_path, submodule_search_locations=[str(module_path.parent)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(name, None)
            raise
    return module


def resolve_root(explicit=None):
    """The configured release root, from the card or EXHAD_ROOT."""
    raw = explicit or os.environ.get('EXHAD_ROOT')
    if not raw:
        raise ValueError('Set exhad_root in the card or EXHAD_ROOT to the configured release')
    return Path(raw).expanduser().resolve()


def resolve_model(name, *, variant=None):
    """Map a host name (LLP name, benchmark key or model name) to an exHad model.

    ``variant`` selects the scalar prescription.  A retired name raises, with
    the replacement spelled out in the message.
    """
    name = str(name)
    if name in RETIRED_MODELS:
        raise ValueError(
            "exHad has no model %r; use %s" % (name, RETIRED_MODELS[name]))
    model = None
    if name in KNOWN_MODELS:
        model = name
    elif name in MODEL_BY_LLP:
        model = MODEL_BY_LLP[name]
    elif name in MODEL_BY_BENCH:
        model = MODEL_BY_BENCH[name]
    else:
        raise ValueError(
            "No exHad model for %r; known models: %s" % (name, ', '.join(KNOWN_MODELS)))
    if variant is not None and model in SCALAR_MODELS:
        key = str(variant)
        if key not in SCALAR_VARIANTS:
            raise ValueError(
                "Unknown scalar prescription %r; choose from: %s"
                % (variant, ', '.join(sorted(SCALAR_VARIANTS))))
        model = SCALAR_VARIANTS[key]
    elif variant is not None:
        raise ValueError(
            "Model %r has no prescription variants; %r was requested" % (model, variant))
    return model


def model_info(root, model):
    """Matched support, EventCalc rate tables, row ownership and version of a release model."""
    module = release(root)
    if not hasattr(module, 'model_info'):
        raise ValueError('This exHad release has no model_info host interface')
    if hasattr(module, 'ParallelGenerator'):
        raise ValueError(
            'This exHad release exposes ParallelGenerator; EventCalc requires a release '
            'whose single Generator parallelizes by default')
    return module.model_info(model)


def release_version(root):
    """A provenance string for the configured release.

    exHad carries no release numbering: ``model_info`` returns no ``version``
    and the package defines no ``__version__``.  Report whatever the release
    offers, and otherwise say plainly that it is unversioned.
    """
    module = release(root)
    version = getattr(module, '__version__', None)
    if version:
        return str(version)
    return 'unversioned'


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generator(binding):
    """The pooled ``Generator`` for one binding.

    One ``Generator`` generates in parallel by default.  ``workers=1`` keeps a
    single process, which is what the deterministic-seed tests use.
    """
    cpu_count = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
    workers = int(os.environ.get('EXHAD_WORKERS', str(min(8, cpu_count))))
    chunk_size = int(os.environ.get('EXHAD_CHUNK_SIZE', '512'))
    if not 1 <= workers <= 64 or chunk_size < 1:
        raise ValueError('EXHAD_WORKERS must be 1..64; EXHAD_CHUNK_SIZE must be positive')
    key = (binding['root'], binding['python'], binding['model'], workers, chunk_size)
    if key not in _GENERATORS:
        _GENERATORS[key] = release(binding['root']).Generator(
            binding['model'], workers=workers, chunk_size=chunk_size,
            root=binding['root'], python=binding['python'])
    instance = _GENERATORS[key]
    binding['execution'] = dict(workers=instance.workers, chunk_size=instance.chunk_size,
                                seed_stream='chunked-v1')
    return instance


# ---------------------------------------------------------------------------
# Binding
# ---------------------------------------------------------------------------

def mother_name(model):
    """The decaying particle of a model, for messages that quote its mass."""
    return MOTHER_NAME.get(str(model), 'mother')


def bind(root, model, *, python=None, llp_pdg=None):
    """Everything about one release model this host needs, validated.

    The interval this host enforces on a requested mass is
    ``generation_gev["all"]``, the range of mother mass in which the release
    generates decays; it is kept as ``generation_start`` and
    ``generation_end``.  The upper edge of ``support_gev`` is kept separately
    as ``matched_w_end`` because it bounds the hadronic invariant mass W, and
    a mother that shares its mass with a lepton has a W below its own mass.
    """
    root = Path(root).expanduser().resolve()
    info = model_info(root, model)
    support = [float(x) for x in info['support_gev']]
    if len(support) != 2 or not all(math.isfinite(x) for x in support) or support[1] <= support[0]:
        raise ValueError('exHad deployment has incompatible mass support')
    generation = {key: [float(x) for x in value]
                  for key, value in dict(info.get('generation_gev') or {}).items()}
    window = list(generation.get('all') or support)
    if (len(window) != 2 or not all(math.isfinite(x) for x in window) or window[1] <= window[0]
            or not window[0] <= support[0] <= window[1]):
        raise ValueError(
            'exHad deployment generates outside the mass support of its matched description')
    exclusive = info.get('exclusive_gev')
    exclusive = [float(x) for x in exclusive] if exclusive else None
    rows = info.get('exclusive_rows') or []
    default_pdg = int(info['mother_pdg'])
    binding = {
        'root': str(root),
        'python': python or sys.executable,
        'model': model,
        'version': info.get('version') or release_version(root),
        'generation_start': window[0],
        'generation_end': window[1],
        'generation_gev': generation,
        'matched_start': support[0],
        'matched_w_end': support[1],
        'support_gev': support,
        'exclusive_gev': exclusive,
        'tables': dict(info.get('tables') or {}),
        'mother_pdg': default_pdg if llp_pdg is None else int(llp_pdg),
        'default_mother_pdg': default_pdg,
        'exclusive_rows': {row['label']: tuple(sorted(int(p) for p in row['pdg_signature']))
                           for row in rows},
    }
    binding['exclusive_by_signature'] = {}
    for label, signature in binding['exclusive_rows'].items():
        binding['exclusive_by_signature'].setdefault(signature, []).append(label)
    if 'eventcalc_rows' in info:
        binding['signatures'] = {tuple(row['pdg_signature']): row['authority_row_id']
                                 for row in info['eventcalc_rows']}
    if 'owner_probabilities' in info:
        binding['owner_probabilities'] = info['owner_probabilities']
    return binding


def check_mass(binding, mass, *, allow_exclusive=True):
    """Classify a mother mass against the intervals the release generates in.

    Returns 'matched' when the inclusive matched generator serves the mass
    (``generate``), 'exclusive' when the mass is below the start of that
    generator and inside the exclusive window (``generate_rows``), and raises
    otherwise.

    The ceiling is ``generation_end``, the end of generation, which is a
    mother mass.  ``matched_w_end`` is a bound on the hadronic invariant mass
    W; an HNL leaves part of its own mass to the accompanying lepton, so
    enforcing that bound as a mass ceiling would refuse HNL masses the release
    generates.
    """
    mass = float(mass)
    mother = mother_name(binding['model'])
    interval = "%s mass %g to %g GeV" % (mother, binding['generation_start'],
                                         binding['generation_end'])
    if mass > binding['generation_end']:
        raise ValueError(
            "exHad %s generates decays for %s; %g GeV was requested. Select raw "
            "hadronization explicitly above that mass."
            % (binding['model'], interval, mass))
    if mass >= binding['matched_start']:
        return 'matched'
    window = binding['exclusive_gev']
    if window and window[0] <= mass < window[1]:
        if allow_exclusive:
            return 'exclusive'
        raise ValueError(
            "%s mass %g GeV is served by the exclusive rows of exHad %s; its inclusive "
            "matched generator starts at %g GeV"
            % (mother, mass, binding['model'], binding['matched_start']))
    if mass < binding['generation_start']:
        raise ValueError(
            "exHad %s generates decays for %s; %g GeV was requested"
            % (binding['model'], interval, mass))
    raise ValueError(
        "%s mass %g GeV is below the %g GeV start of the exHad %s matched generator and "
        "outside its exclusive window %s"
        % (mother, mass, binding['matched_start'], binding['model'], window))


# ---------------------------------------------------------------------------
# Exclusive routing
# ---------------------------------------------------------------------------

def exclusive_label(binding, pdg_row, *, row_label=None):
    """The exclusive row a host decay row maps onto, or None.

    Matched by PDG signature; the row label breaks the tie where several
    exclusive rows share one signature (the K_S/K_L partners of the
    kaon-pion towers do).
    """
    signature = tuple(sorted(int(p) for p in pdg_row if int(p) != -999))
    candidates = binding['exclusive_by_signature'].get(signature)
    if not candidates:
        return None
    if row_label is not None and str(row_label) in candidates:
        return str(row_label)
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        "Decay row with signature %s matches several exHad exclusive rows (%s); "
        "the host row label must name one of them" % (list(signature), ', '.join(candidates)))


def generate_exclusive(binding, mass, counts, *, seed=1, terminal='pythia', mixing=None):
    """Exact final states for named exclusive rows, at exact counts.

    ``counts`` maps exclusive row labels to event counts.  EventCalc keeps the
    rates: it decides the counts from its own branching ratios, and this only
    realizes them.  A label's events depend on the seed, the label and the
    count alone, so an unrelated channel's count does not move them.
    """
    wanted = {label: int(count) for label, count in counts.items() if int(count) > 0}
    if not wanted:
        return {}
    unknown = sorted(set(wanted) - set(binding['exclusive_rows']))
    if unknown:
        raise ValueError(
            "Not exclusive rows of exHad %s: %s" % (binding['model'], ', '.join(unknown)))
    kwargs = {'seed': int(seed), 'terminal': terminal}
    if mixing is not None:
        kwargs['mixing'] = mixing
    produced = generator(binding).generate_rows(float(mass), wanted, **kwargs)
    for label, count in wanted.items():
        got = len(produced.get(label, ()))
        if got != count:
            raise RuntimeError(
                "exHad returned %d events for exclusive row %r, expected %d" % (got, label, count))
    return produced


# ---------------------------------------------------------------------------
# Mother PDG code
# ---------------------------------------------------------------------------

def default_mother_pdg(root, model):
    """The release's default mother PDG code for a model."""
    return int(model_info(root, model)['mother_pdg'])


def resolve_mother_pdg(root, model, override=None):
    """The mother PDG code for a run: the release default unless overridden.

    Both EventCalc copies use the same defaults as exHad (dark-photon
    4900022, b-l 32, the scalar variants 35, alp-fermion 36, hnl 9900012).
    ``--llp-pdg`` overrides it for one run.  This is the code that labels the
    LLP in the event record; EventCalc's internal hadronization mother code 25
    is a separate, unrelated convention and is not touched here.
    """
    if override is None:
        return default_mother_pdg(root, model)
    code = int(override)
    if code == 0:
        raise ValueError('--llp-pdg must be a non-zero PDG code')
    return code


def payload_metadata(binding):
    """The provenance block both hosts write into their JSON payloads.

    ``mother_mass_gev`` is the interval of mother mass the run may request.
    ``matched_w_end_gev`` is the largest hadronic invariant mass W the matched
    description covers; it is reported rather than enforced, because for a
    mother that shares its mass with a lepton it is not a bound on that mass.
    """
    result = {
        'backend': 'exhad',
        'model': binding['model'],
        'release_root': binding['root'],
        'release_version': binding['version'],
        'mother_mass_gev': [binding['generation_start'], binding['generation_end']],
        'generation_gev': binding['generation_gev'],
        'matched_start_gev': binding['matched_start'],
        'matched_w_end_gev': binding['matched_w_end'],
        'exclusive_gev': binding['exclusive_gev'],
        'mother_pdg': binding['mother_pdg'],
        'default_mother_pdg': binding['default_mother_pdg'],
        'event_count_scope': 'complete selected hadronic row pool',
    }
    if 'execution' in binding:
        result['execution'] = dict(binding['execution'])
    if binding['model'] == 'hnl':
        result.update(event_count_scope='each signed Jets row separately',
                      composition_coordinate='per-event hadronic invariant mass W',
                      currents=['CC_ud', 'CC_us', 'CC_cd', 'CC_cs', 'NC_ud', 'NC_s'])
    return result


def check_hnl_inputs(binding, particle_path):
    """The release check: EventCalc's HNL inputs must be the release's own.

    The release takes lepton masses from the generator, carries
    threshold-aware rates, and has no radiative N -> nu gamma row; EventCalc
    must not add one either, or this comparison fails.
    """
    import numpy as np
    tables = binding['tables']
    installed = Path(particle_path)
    released_rows = json.loads(Path(tables['decay']).read_text())
    installed_rows = json.loads((installed / 'HNL-decay.json').read_text())
    if released_rows != installed_rows or not np.array_equal(
            np.loadtxt(tables['widths']), np.loadtxt(installed / 'HNLdecayWidth.dat')):
        raise ValueError('EventCalc HNL decay and width inputs must match the exHad release')
    return True
