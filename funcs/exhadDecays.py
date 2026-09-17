"""
Hook between EventCalc and a configured public exHad release.

When enabled, the hadronic decay rows owned by a matched exHad model are
replaced by complete rest-frame decays from the release's public interface
(``exhad.model_info`` and ``exhad.Generator``).  Events come in the
downstream 6-field format ([px,py,pz,E,m,pdg] per particle, padded), with
pi0/K_S decayed.  The record retains every stable final particle, including
neutrinos.  The hook changes only the conditional composition and kinematics,
never a partial width, total width, branching ratio, or lifetime.

Release: ``export EXHAD_ROOT=/path/to/configured/exhad``.  EXHAD_PYTHON selects
the worker interpreter (default: this one); EXHAD_WORKERS (default: up to 8
CPUs) and EXHAD_CHUNK_SIZE (default 512) set the persistent parallel workers.
A launch card may name the same two paths through ``exhad_root`` and
``exhad_python``; ``configure_llp`` applies them to this module for the run.

Enabling (in precedence order):
  0. the host's raw-Pythia choice, set_force(False);
  1. env override  export EXHAD_BENCH=hls  (force a benchmark for any LLP)
     or export EXHAD_BENCH=raw (force baseline Pythia);
  2. per-LLP card  Distributions/<LLP>/exhad.json  {"bench": "dv", ...}, with
     "bench_variants" mapping the scalar decay-table prescriptions -- the
     default: activates AUTOMATICALLY from the LLP chosen in the EventCalc
     menu (the drivers call set_selection() with the LLP's name + path);
  3. built-in map  BENCH_BY_LLP  (legacy callers that provide an LLP name
     but no EventCalc particle path);
  4. legacy env  export EXHAD_LLP="Dark-photons".

Benchmarks select release models (EXHAD_MODELS).  ``dv``: the diagonal u, d, s
and c rows form one total-electromagnetic pool.  ``alp`` and ``hls*``: the
Model-1 deployments own every row listed by model_info, and the selected
EventCalc branching ratios must reproduce the release's outer-owner
probabilities.  ``hnl``: each signed Jets row is replaced at the exact
hadronic invariant mass W of every event, keeping the lepton or neutrino.

This module is the one exHad adapter in the tree.  ``funcs.exhad_integration``
re-exports part of it under the names its historical callers use; the routing
itself lives in ``funcs.decayProducts.simulateDecays_rest_frame``, the single
decay entry point both drivers call.
"""

import atexit
import json
import math
import os
import sys
from bisect import bisect_left
from functools import lru_cache

# LLP name (as used by EventCalc/SensCalc) -> exhad benchmark key.
BENCH_BY_LLP = {
    "Dark-photons": "dv",
    "Scalar-mixing": "hls",
    "Scalar-quartic": "hls",
    "HNL": "hnl",
    "ALP-fermion": "alp",
    # NOTE: EventCalc's "ALP-photon" is the PHOTON-coupled ALP, a different
    # benchmark from the fermion-universal exHad ALP; it is not mapped.
}

# Benchmark key -> public exHad model.  The authority is the shared
# funcs.exhad_release module, which both EventCalc copies import: the scalar is
# named by its prescription (central/lower/upper/1809), the B-L boson is
# "b-l", and one Generator parallelizes by default.
from funcs import exhad_release
from funcs.exhad_release import (MODEL_BY_BENCH as EXHAD_MODELS, MODEL_BY_LLP,
                                 SCALAR_MODELS, check_mass, exclusive_label,
                                 generate_exclusive, generator as _generator,
                                 resolve_mother_pdg)

#: Retained for callers that introspect the host model map.  The authority is
#: funcs.exhad_release.
MODELS = dict(MODEL_BY_LLP)

_SEL_LLP = None      # LLP name (as in Distributions/ and LLP_selection.py)
_SEL_PATH = None     # that LLP's Distributions/<LLP>/ folder (the card home)
_SEL_VARIANT = None  # optional decay-table prescription (scalar variants)
_GENERATORS = {}

# Runtime override from a host prompt or a --exhad flag. None: follow the
# card/env resolution; True: request exhad; False: force raw Pythia this run.
_FORCE = None

# Release location named by a launch card instead of the environment.  Set by
# configure_llp, read by _release_root/_release_python, so that one run resolves
# the release in exactly one way whichever driver started it.
_ROOT_OVERRIDE = None
_PYTHON_OVERRIDE = None

#: Mass-window fields a card may declare, and the release coordinate each one
#: must equal.  The code is the authority: a card that disagrees with the
#: installed release is refused rather than followed.
_CARD_WINDOW_FIELDS = {
    "m_start": ("support_gev", 0),
    "m_edge": ("support_gev", 0),
    "m_inclusive": ("support_gev", 0),
    "m_high": ("support_gev", 1),
    "m_free": ("support_gev", 1),
}

_CARD_WINDOW_TOLERANCE = 1.0e-9


def set_selection(llp_name=None, particle_path=None, variant=None):
    """Register the LLP EventCalc is about to decay (None, None clears it).

    ``variant`` is the host decay-table prescription.  Scalar cards use it to
    keep the EventCalc branching-ratio table and the exhad model synchronized.
    """
    global _SEL_LLP, _SEL_PATH, _SEL_VARIANT
    _SEL_LLP, _SEL_PATH, _SEL_VARIANT = llp_name, particle_path, variant


def set_force(use_exhad):
    """Host choice: True requests exhad, False forces raw Pythia, None follows env/card."""
    global _FORCE
    _FORCE = use_exhad


@lru_cache(maxsize=32)
def _card_config(particle_path):
    """Read and cache a per-LLP exhad card; ``None`` only when none exists.

    An existing but unreadable, malformed, or non-object card fails closed.
    """
    if not particle_path:
        return None
    card_path = os.path.join(particle_path, "exhad.json")
    try:
        with open(card_path) as f:
            cfg = json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "malformed exhad card %s: %s" % (card_path, exc)) from exc
    except OSError as exc:
        raise RuntimeError(
            "cannot read exhad card %s: %s" % (card_path, exc)) from exc
    if not isinstance(cfg, dict):
        raise RuntimeError(
            "exhad card %s must contain a JSON object" % card_path)
    return cfg


def _card_bench(particle_path):
    """The card's benchmark, resolved for the selected decay-table prescription."""
    cfg = _card_config(particle_path)
    if cfg is None:
        return None
    bench = cfg.get("bench")
    if not isinstance(bench, str) or not bench.strip():
        raise RuntimeError(
            "exhad card must define a non-empty string field 'bench'")
    if _SEL_VARIANT is None:
        return bench.strip()
    variants = cfg.get("bench_variants")
    if not isinstance(variants, dict) or _SEL_VARIANT not in variants:
        raise RuntimeError(
            "exhad card has no benchmark for decay-table prescription %r" %
            (_SEL_VARIANT,))
    resolved = variants[_SEL_VARIANT]
    if not isinstance(resolved, str) or not resolved.strip():
        raise RuntimeError(
            "invalid exhad benchmark alias for prescription %r" %
            (_SEL_VARIANT,))
    return resolved.strip()


def get_bench():
    """Benchmark key or None (None -> the raw-Pythia path); see the module docstring."""
    if _FORCE is False:
        return None
    b = os.environ.get("EXHAD_BENCH")
    if b and b.strip().lower() in {"0", "off", "none", "raw", "pythia"}:
        return None
    if b:
        return b.strip()
    c = _card_bench(_SEL_PATH)
    if c:
        return c
    if _SEL_PATH and _SEL_LLP in BENCH_BY_LLP:
        raise RuntimeError(
            "missing exhad card %s for configured LLP %s; refusing to "
            "fall back to built-in benchmark %s" %
            (os.path.join(_SEL_PATH, "exhad.json"), _SEL_LLP,
             BENCH_BY_LLP[_SEL_LLP]))
    if _SEL_PATH is None and _SEL_LLP and BENCH_BY_LLP.get(_SEL_LLP):
        return BENCH_BY_LLP[_SEL_LLP]
    llp = os.environ.get("EXHAD_LLP")
    if llp:
        return BENCH_BY_LLP.get(llp)
    return None


def _release_root():
    """The configured release directory, or None when none is configured."""
    if _ROOT_OVERRIDE:
        return _ROOT_OVERRIDE
    root = os.environ.get("EXHAD_ROOT", "").strip()
    return os.path.realpath(os.path.expanduser(root)) if root else None


def _release_python():
    """The interpreter the release's workers run under."""
    return _PYTHON_OVERRIDE or os.environ.get("EXHAD_PYTHON") or sys.executable


@lru_cache(maxsize=4)
def _release(root):
    """Import the public exhad package of one release once, under a private name.

    Delegates to the shared funcs.exhad_release binding so that both EventCalc
    copies import the release in exactly one way.
    """
    module = exhad_release.release(root)
    if not hasattr(module, "model_info"):
        raise RuntimeError(
            "the exHad release at %s has no model_info host interface" % root)
    if hasattr(module, "ParallelGenerator"):
        raise RuntimeError(
            "the exHad release at %s still exposes ParallelGenerator; EventCalc "
            "requires the release with a single Generator that parallelizes by "
            "default" % root)
    return module


@lru_cache(maxsize=16)
def _release_model_info(root, model, particle_path):
    info = _release(root).model_info(model)
    if model == "hnl":
        # EventCalc keeps the HNL widths; they must be the release's matched tables.
        import numpy as np
        tables = info["tables"]
        with open(tables["decay"]) as released, \
                open(os.path.join(particle_path, "HNL-decay.json")) as installed:
            same = json.load(released) == json.load(installed)
        if not same or not np.array_equal(
                np.loadtxt(tables["widths"]),
                np.loadtxt(os.path.join(particle_path, "HNLdecayWidth.dat"))):
            raise RuntimeError(
                "EventCalc HNL decay and width inputs must match the exHad release")
    return info


def model_info(bench=None):
    """The release's model_info for the selected (or given) benchmark."""
    selected = get_bench() if bench is None else bench
    if selected not in EXHAD_MODELS:
        raise RuntimeError(
            "exhad benchmark %r has no public exHad model" % (selected,))
    root = _release_root()
    if root is None:
        raise RuntimeError("exhadDecays: set EXHAD_ROOT to the configured exHad release")
    path = _SEL_PATH or os.path.join("Distributions", "HNL")
    return _release_model_info(root, EXHAD_MODELS[selected],
                               os.path.realpath(path) if selected == "hnl" else None)


def _model_generator(model):
    """The pooled Generator of one release model, keyed by model name."""
    root = _release_root()
    workers = int(os.environ.get(
        "EXHAD_WORKERS", str(min(8, os.cpu_count() or 1))))
    chunk_size = int(os.environ.get("EXHAD_CHUNK_SIZE", "512"))
    if not 1 <= workers <= 64 or chunk_size < 1:
        raise ValueError(
            "EXHAD_WORKERS must be 1..64; EXHAD_CHUNK_SIZE must be positive")
    python = _release_python()
    key = (root, python, model, workers, chunk_size)
    if key not in _GENERATORS:
        # exHad has one Generator, which parallelizes by default.
        _release(root)
        _GENERATORS[key] = exhad_release.release(root).Generator(
            model, workers=workers, chunk_size=chunk_size, root=root, python=python)
    return _GENERATORS[key]


def close_generators():
    for generator in _GENERATORS.values():
        generator.close()
    _GENERATORS.clear()
    exhad_release.close_generators()


atexit.register(close_generators)


def can_use_exhad():
    """True if exhad could handle the currently-selected LLP's hadronic decays:
    a benchmark resolves for it AND an exHad release is configured.  A resolved
    benchmark without a public model or with an invalid release fails closed."""
    bench = get_bench()
    if bench is None or _release_root() is None:
        return False
    model_info(bench)
    return True


def enabled(mass=None):
    """True if exhad handles the fixed-mass hadronic rows at ``mass``: inside
    the closed matched support of the selected release model.  HNL uses the
    separate per-event-W path and is not gated here."""
    bench = get_bench()
    if bench is None or bench == "hnl" or _release_root() is None:
        return False
    if mass is not None:
        low, high = model_info(bench)["support_gev"]
        return low <= float(mass) <= high
    return True


def _owner_probabilities_at(owners, mass):
    """Outer-owner row probabilities: piecewise linear, normalized at ``mass``."""
    masses, columns = owners["masses"], owners["probabilities"]
    mass = float(mass)
    right = bisect_left(masses, mass)
    if not masses[0] <= mass <= masses[-1]:
        raise RuntimeError("mass lies outside the outer-owner probability nodes")
    if right == 0 or masses[right] == mass:
        values = {key: column[right] for key, column in columns.items()}
    else:
        fraction = (mass - masses[right - 1]) / (masses[right] - masses[right - 1])
        values = {key: (1 - fraction) * column[right - 1] + fraction * column[right]
                  for key, column in columns.items()}
    total = math.fsum(values.values())
    return {key: value / total for key, value in values.items()}


def uses_portable_model1_backend(bench=None):
    """Whether the selected model is a Model-1 deployment with EventCalc row ownership."""
    selected = get_bench() if bench is None else bench
    return (selected in EXHAD_MODELS and selected not in ("dv", "hnl")
            and _release_root() is not None)


def portable_model1_configuration():
    """model_info of the selected Model-1 deployment, or None."""
    return model_info() if uses_portable_model1_backend() else None


def portable_model1_eventcalc_row_label(pdg_list):
    config = portable_model1_configuration()
    if config is None:
        return None
    signature = sorted(int(value) for value in pdg_list if int(value) != -999)
    return next((row["authority_row_id"] for row in config["eventcalc_rows"]
                 if row["pdg_signature"] == signature), None)


def portable_model1_eventcalc_authority_probabilities(mass):
    """Outer-owner row probabilities: piecewise linear, normalized at ``mass``."""
    config = portable_model1_configuration()
    if config is None:
        raise RuntimeError("portable Model 1 deployment is not selected")
    return _owner_probabilities_at(config["owner_probabilities"], mass)


def fixed_partonic_row_supported(pdg_list, bench=None):
    """Whether a fixed-mass row belongs to the dark-photon total-EM pool.

    The pool owns the four diagonal ``u``, ``d``, ``s`` and ``c`` rows; EventCalc
    contributes only their summed, already-selected event count.  Gluon rows
    remain outside this vector-current pool.
    """
    ids = tuple(int(value) for value in pdg_list if int(value) != -999)
    selected = get_bench() if bench is None else bench
    return (selected == "dv" and len(ids) == 2 and ids[0] == -ids[1]
            and abs(ids[0]) in {1, 2, 3, 4})


def generate_explicit_multibody_rest_frame(
        mass, pdgs, masses, charges, stabilities, n_events, seed=1):
    """Compatibility wrapper for EventCalc's native N-body sampler.

    Explicit decay-table rows are independent of the matched-hadronization
    choice; this historical entry point needs no exHad installation.
    """
    import numpy as np
    from funcs import NBodyDecay

    seed = int(seed)
    if seed < 0:
        raise ValueError("explicit multibody seed must be non-negative")
    return NBodyDecay.decay_products(
        mass, pdgs, masses, charges, stabilities, n_events,
        rng=np.random.default_rng(seed))


# ---------------------------------------------------------------------------
# HNL (3-body, per-event hadronic mass W)
# ---------------------------------------------------------------------------
#
# The HNL decay N -> l + (q qbar) is 3-body: the hadronic system carries an
# invariant mass W < m_N distributed by EventCalc's own matrix element.  The
# release replaces ONLY the q-qbar system at each event's exact W and keeps the
# sampled lepton/neutrino and the pair four-momentum; rates, lifetime and the
# W spectrum are untouched.  CC ud/us/cd/cs and NC ud/s are supported; NC c/b
# and CC ub/cb remain inactive through the production endpoint and fail closed.

_CHARGED_LEPTONS = {11, 13, 15}
_NEUTRINOS = {12, 14, 16}
_CC_CURRENT_PAIRS = {
    "CC_ud": (2, 1),
    "CC_us": (2, 3),
    "CC_cd": (4, 1),
    "CC_cs": (4, 3),
    "CC_ub": (2, 5),
    "CC_cb": (4, 5),
}
_NC_CURRENTS = {1: "NC_ud", 2: "NC_ud", 3: "NC_s", 4: "NC_c", 5: "NC_b"}
_HNL_SUPPORTED_CURRENTS = frozenset({
    "CC_ud", "CC_us", "CC_cd", "CC_cs", "NC_ud", "NC_s",
})
_QUARK_CHARGE = {
    1: -1.0 / 3.0, 2: 2.0 / 3.0, 3: -1.0 / 3.0,
    4: 2.0 / 3.0, 5: -1.0 / 3.0, 6: 2.0 / 3.0,
}


def is_hnl_bench():
    return get_bench() == "hnl"


def generation_window(bench=None):
    """The mother-mass interval in which the release generates decays.

    This is ``generation_gev["all"]`` of the selected model.  For the HNL it is
    the interval of HNL mass; the upper edge of ``support_gev`` bounds the
    hadronic invariant mass W instead and is no ceiling on that mass.
    """
    info = model_info(bench)
    window = (info.get("generation_gev") or {}).get("all") or info["support_gev"]
    return float(window[0]), float(window[1])


def hnl_parent_mass_supported(mass):
    """Whether the release generates HNL decays at this HNL mass."""
    try:
        value = float(mass)
    except (TypeError, ValueError):
        return False
    low, high = generation_window("hnl")
    return low <= value <= high


def require_hnl_parent_masses_supported(masses):
    """Fail before generation if an HNL scan leaves the generated mass interval."""
    invalid = [mass for mass in masses if not hnl_parent_mass_supported(mass)]
    if invalid:
        low, high = generation_window("hnl")
        raise RuntimeError(
            "exhad generates HNL decays for HNL mass %g to %g GeV "
            "(offending masses: %s)" %
            (low, high, ", ".join(str(mass) for mass in invalid)))


def hnl_channel_spec(pdg_list):
    """Classify one current-resolved HNL ``Jets-*`` row.

    The return value is a small dictionary with ``current``, ``kind``,
    ``lepton`` and, for charged currents, ``conjugate``.  ``conjugate`` is
    derived from electric charge, never from the numerical sum of PDG ids
    (which gives the wrong answer for u-sbar).
    """
    ids = [int(p) for p in pdg_list if int(p) != -999]
    if len(ids) != 3:
        return None
    quarks = [pid for pid in ids if 1 <= abs(pid) <= 6]
    if len(quarks) != 2:
        return None

    neutrinos = [pid for pid in ids if abs(pid) in _NEUTRINOS]
    charged = [pid for pid in ids if abs(pid) in _CHARGED_LEPTONS]
    if len(neutrinos) == 1 and not charged:
        q1, q2 = quarks
        if q1 != -q2:
            return None
        current = _NC_CURRENTS.get(abs(q1))
        if current is None:
            return None
        return {"current": current, "kind": "NC",
                "lepton": neutrinos[0], "conjugate": False,
                "supported": current in _HNL_SUPPORTED_CURRENTS}

    if len(charged) != 1 or neutrinos:
        return None
    qset = frozenset(quarks)
    current = None
    for label, (up, down) in _CC_CURRENT_PAIRS.items():
        if qset in (frozenset((up, -down)),
                    frozenset((-up, down))):
            current = label
            break
    if current is None:
        return None
    hadronic_charge = sum(
        (1.0 if pid > 0 else -1.0) * _QUARK_CHARGE[abs(pid)]
        for pid in quarks)
    if abs(abs(hadronic_charge) - 1.0) > 1e-12:
        raise RuntimeError("invalid charged-current HNL row: %r" % ids)
    lepton_charge = -1.0 if charged[0] > 0 else 1.0
    if abs(hadronic_charge + lepton_charge) > 1e-12:
        # A neutral (Majorana) HNL must have a charge-neutral final state.
        return None
    return {"current": current, "kind": "CC", "lepton": charged[0],
            # exhad's reference convention has hadronic charge -1.
            "conjugate": hadronic_charge > 0.0,
            "supported": current in _HNL_SUPPORTED_CURRENTS}


def process_hnl_with_exhad(decay_events, pdg_list, mass, seed=1):
    """Replace the q-qbar pair of each sampled N -> l + (q qbar) event at its exact W.

    ``decay_events`` are EventCalc's 8-field three-body rows in ``pdg_list``
    order.  The release checks the W thresholds and endpoint, keeps the lepton
    or neutrino, and returns [lepton, hadrons...]; the result is the six-field
    padded array expected downstream.
    """
    import numpy as np
    ids = [int(p) for p in pdg_list if int(p) != -999]
    spec = hnl_channel_spec(ids)
    if spec is None or not spec["supported"]:
        raise RuntimeError(
            "HNL row %r has no event backend in the installed production "
            "domain" % (ids,))
    rows = np.asarray(decay_events, dtype=np.float64)
    if rows.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    print(f"exhadDecays[HNL]: re-hadronizing {len(rows)} {spec['current']} "
          f"events (row {ids}) at per-event W.")
    events = _model_generator("hnl").hadronize_hnl(
        float(mass), ids, rows.reshape(len(rows), -1).tolist(), seed=int(seed))
    return _pad_six_field_events(
        [[value for particle in event for value in particle] for event in events])


def _pad_six_field_events(events):
    """Rectangularize 6-field records with the EventCalc PDG sentinel."""
    import numpy as np
    maxlen = max((len(event) for event in events), default=0)
    if maxlen % 6:
        raise RuntimeError("invalid exhad six-field event length")
    arr = np.zeros((len(events), maxlen), dtype=np.float64)
    if maxlen:
        arr[:, 5::6] = -999.0
    for index, event in enumerate(events):
        if len(event) % 6:
            raise RuntimeError("invalid exhad six-field event length")
        arr[index, :len(event)] = event
    return arr


def process_events_with_exhad(n_events, mass, seed=1):
    """Replacement for process_events_with_pythia for the owned fixed-mass rows:
    a padded (n_events, 6*max_particles) array of complete rest-frame decays.
    Only the pooled EVENT COUNT of the selected rows is used; the release draws
    the complete conditional composition exactly once."""
    if not enabled(mass):
        raise RuntimeError(
            "exhadDecays: not enabled for this configuration (check "
            "EXHAD_BENCH/EXHAD_LLP, EXHAD_ROOT, and the matched mass support).")
    bench = get_bench()
    model = EXHAD_MODELS[bench]
    print(f"exhadDecays: generating {n_events} hadronic events "
          f"(bench={bench}, exHad model {model}, m={mass:g} GeV) instead of raw Pythia.")
    events = _model_generator(model).generate(float(mass), int(n_events), seed=int(seed))
    return _pad_six_field_events(
        [[value for particle in event for value in particle] for event in events])


# ---------------------------------------------------------------------------
# Mother PDG code, mass windows and exclusive rows (shared funcs.exhad_release)
# ---------------------------------------------------------------------------

_MOTHER_PDG_OVERRIDE = None


def set_mother_pdg(code=None):
    """Host override for the decaying particle's PDG code (``--llp-pdg``).

    ``None`` restores the exHad release default.  EventCalc's *internal*
    hadronization mother code 25 is unrelated and is not affected.
    """
    global _MOTHER_PDG_OVERRIDE
    _MOTHER_PDG_OVERRIDE = None if code is None else int(code)


def mother_pdg(bench=None):
    """The decaying particle's PDG code: the release default unless overridden."""
    selected = get_bench() if bench is None else bench
    if selected is None:
        return _MOTHER_PDG_OVERRIDE
    root = _release_root()
    if root is None:
        return _MOTHER_PDG_OVERRIDE
    return resolve_mother_pdg(root, EXHAD_MODELS[selected], _MOTHER_PDG_OVERRIDE)


def _binding(bench=None):
    """A shared funcs.exhad_release binding for the selected benchmark."""
    selected = get_bench() if bench is None else bench
    root = _release_root()
    if selected is None or root is None:
        raise RuntimeError("exhadDecays: no benchmark or no configured release")
    return exhad_release.bind(root, EXHAD_MODELS[selected],
                              python=_release_python(),
                              llp_pdg=_MOTHER_PDG_OVERRIDE)


def exclusive_enabled(mass):
    """True below the matched support but inside the release's exclusive window.

    There exHad still supplies exact final states row by row through
    ``generate_rows``.  This is what closes the old 1.699-1.700 GeV dark-photon
    gap, where events used to fall through to unfiltered Pythia, and it removes
    the sub-threshold neutral-kaon NaNs with it: a row exHad cannot open is
    never requested.
    """
    bench = get_bench()
    if bench is None or bench == "hnl" or _release_root() is None:
        return False
    window = model_info(bench).get("exclusive_gev")
    return bool(window) and float(window[0]) <= float(mass) < float(window[1])


def exclusive_row_label(pdg_list, row_label=None, bench=None):
    """The exclusive exHad row a host decay row maps onto, or None."""
    return exclusive_label(_binding(bench), pdg_list, row_label=row_label)


def generate_exclusive_rows(mass, counts, seed=1, terminal="pythia", bench=None):
    """Exact final states for named exclusive rows, at exactly the asked counts.

    EventCalc keeps the rates: it decides the counts from its own branching
    ratios and channel selection, and this only realizes them.
    """
    return generate_exclusive(_binding(bench), mass, counts,
                              seed=int(seed), terminal=terminal)


# ---------------------------------------------------------------------------
# Binding one LLP to the release
# ---------------------------------------------------------------------------

def _two_column_table(path):
    """Load a release rate table written either as JSON rows or as text columns."""
    import numpy as np
    with open(path, encoding="utf-8") as stream:
        text = stream.read()
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        return np.loadtxt(path)
    return np.asarray(rows, dtype=float)


def _check_card_windows(particle_path, info):
    """Refuse a card whose declared mass window is not the installed release's.

    The code is the authority and the card is checked against it.  A card that
    names a window the release does not have describes a different installation,
    and following it silently is how a run ends up labelled with a support it
    never had.
    """
    card = _card_config(particle_path)
    if not card:
        return
    for field, (key, position) in _CARD_WINDOW_FIELDS.items():
        if field not in card:
            continue
        released = info.get(key)
        if not released:
            continue
        declared = float(card[field])
        expected = float(released[position])
        if abs(declared - expected) > _CARD_WINDOW_TOLERANCE * max(1.0, abs(expected)):
            raise RuntimeError(
                "exhad card %s declares %s = %g GeV; the installed release has "
                "%s[%d] = %g GeV"
                % (os.path.join(particle_path, "exhad.json"), field, declared,
                   key, position, expected))
    # The HNL card's validated_event_composition states the parent masses its
    # composition was validated over.  The release states two mass intervals
    # and neither is that quantity: generation_gev is the parent masses it
    # generates decays for, 0.02 to 40 GeV, and support_gev's upper edge
    # bounds W, the invariant mass of the hadronic system.  Checking the card's
    # number against support_gev only passed because both trace to the B mass,
    # and reading a W bound as a parent-mass ceiling is the conflation the HNL
    # mass window removed everywhere else.  Nothing is checked here.


#: The two sides of every table comparison below are the same decimal text
#: read by two different parsers -- pandas on the installed file, the JSON and
#: text readers on the release's -- and those disagree in the last bit of a
#: double.  The test is therefore that the two are the same table, at a
#: tolerance a thousand times tighter than the smallest difference any table
#: revision has ever made and ten thousand times looser than that round-off.
_TABLE_RELATIVE_TOLERANCE = 1.0e-9


def _same_table(installed, released):
    import numpy as np

    installed = np.asarray(installed, dtype=float)
    released = np.asarray(released, dtype=float)
    return installed.shape == released.shape and bool(np.allclose(
        installed, released, rtol=_TABLE_RELATIVE_TOLERANCE, atol=0.0))


def _check_installed_tables(llp, binding):
    """The installed lifetime and branching tables must be the release's own.

    EventCalc owns the rates, so the host reads them from ``Distributions``.
    That is only safe while they are the tables the generator was matched to,
    and this is where that is established -- by comparison, not by overwriting
    what the host loaded.
    """
    tables = binding.get('tables') or {}
    decay_path = tables.get('decay')
    if decay_path:
        released = json.loads(open(decay_path, encoding="utf-8").read())
        labels = [str(row[0]) for row in released]
        installed_labels = [str(name) for name in llp.decayChannels]
        if installed_labels != labels:
            raise ValueError(
                "the installed %s decay channels are not the exHad release's: "
                "%d installed rows against %d released rows"
                % (binding['model'], len(installed_labels), len(labels)))
        for index, row in enumerate(released):
            released_pdgs = sorted(int(p) for p in row[1] if int(p) != -999)
            installed_pdgs = sorted(
                int(p) for p in llp.PDGs[index] if int(p) != -999)
            if installed_pdgs != released_pdgs:
                raise ValueError(
                    "installed row %r carries PDG ids %s; the exHad release has %s"
                    % (labels[index], installed_pdgs, released_pdgs))
        installed_rates = getattr(llp, 'BrRatios', None)
        if installed_rates is None:
            raise ValueError(
                "the %s loader did not keep its tabulated branching ratios, so "
                "they cannot be checked against the exHad release"
                % binding['model'])
        for index, row in enumerate(released):
            if not _same_table(list(installed_rates[index]), row[2]):
                raise ValueError(
                    "the installed branching table of row %r is not the exHad "
                    "release's" % labels[index])

    ctau_path = tables.get('ctau')
    if ctau_path and getattr(llp, 'ctau_data', None) is not None:
        if not _same_table(llp.ctau_data, _two_column_table(ctau_path)):
            raise ValueError(
                "the installed %s lifetime table is not the exHad release's"
                % binding['model'])


def _has_hadronic_rows(llp):
    """Whether the LLP's decay table holds a row exHad could hadronize.

    A table with no partonic row leaves an explicit ``on`` nothing to do, so
    the request is reported rather than refused.  A table that is not loaded
    is treated as hadronic, so the fail-closed path is the default.
    """
    rows = getattr(llp, 'PDGs', None)
    if rows is None:
        return True
    from funcs import decayProducts
    return any(decayProducts.channel_is_partonic(row) for row in rows)


def configure_llp(llp, config):
    """Bind one LLP object to the configured release for the whole run.

    Sets the mother PDG code, stores the binding, checks the card against the
    installed release, and checks the lifetime and branching tables the host
    loaded against the release's own.  Nothing is replaced: a disagreement is
    an error, because a run whose rates came from one table set and whose final
    states came from another is neither of the two.
    """
    global _ROOT_OVERRIDE, _PYTHON_OVERRIDE
    llp._exhad_binding = None
    llp._exhad_pooled_indices = ()

    mode = getattr(config, 'exhad_mode', 'auto')
    if mode not in {'auto', 'on', 'off'}:
        raise ValueError("exhad_mode must be 'auto', 'on' or 'off'")
    # The run's release is whatever this run's configuration names; a card that
    # names none falls back to the environment, and never to an earlier card.
    root = getattr(config, 'exhad_root', None)
    python = getattr(config, 'exhad_python', None)
    _ROOT_OVERRIDE = str(exhad_release.resolve_root(root)) if root else None
    _PYTHON_OVERRIDE = str(python) if python else None

    set_force({'on': True, 'off': False, 'auto': None}[mode])
    if mode == 'off':
        return

    bench = get_bench()
    if bench is None or _release_root() is None:
        if mode == 'on' and _has_hadronic_rows(llp):
            raise RuntimeError(
                "exhad was requested for %s, but no usable exhad card or "
                "release resolved" % getattr(llp, 'LLP_name', 'this LLP'))
        return

    model = EXHAD_MODELS[bench]
    binding = exhad_release.bind(
        _release_root(), model, python=_release_python(),
        llp_pdg=_MOTHER_PDG_OVERRIDE)
    _check_card_windows(llp.particle_path, model_info(bench))
    if model == 'hnl':
        exhad_release.check_hnl_inputs(binding, llp.particle_path)
    else:
        _check_installed_tables(llp, binding)
    llp.LLP_pdg = binding['mother_pdg']
    llp._exhad_binding = binding


def metadata(llp):
    """The 'backend'/'mother_pdg' sidecar payload for one configured LLP."""
    binding = getattr(llp, '_exhad_binding', None)
    if binding is None:
        return {'backend': 'raw', 'mother_pdg': getattr(llp, 'LLP_pdg', None)}
    return exhad_release.payload_metadata(binding)


def matched_process_labels(decay_channels, selected_indices, pooled_indices):
    """One output label per selected row, aligned to the existing row order.

    A row the matched generator produced carries its own name, so two pooled
    rows never share a process header and no event is lost when the file is
    read back.  A row EventCalc sampled itself gets ``None`` and keeps its
    physical channel name.  No event is moved: this names rows, it does not
    reorder them.
    """
    from funcs.decayProducts import MATCHED_PROCESS_LABEL
    pooled = {int(index) for index in (pooled_indices or ())}
    labels = []
    for index in selected_indices:
        index = int(index)
        if index in pooled:
            labels.append(
                "%s:%s" % (MATCHED_PROCESS_LABEL, str(decay_channels[index])))
        else:
            labels.append(None)
    return labels


def simulate_decays_weighted(llp, decay_module, mass, pdgs, branching, size,
                             matrix_elements, selected, visible, *, seed=1,
                             weight_floor_fraction=0.):
    """Importance-sampled ALP hadronic events inside the matched support.

    Returns ``(events, sizes, weights)``, where ``weights`` carries the raw
    importance weights and the normalization group of every event.  The weights
    are unnormalized on purpose: they are normalized once over every batch at
    one mass, which only the caller can do.
    """
    import numpy as np

    binding = getattr(llp, '_exhad_binding', None)
    if (isinstance(weight_floor_fraction, bool)
            or not isinstance(weight_floor_fraction, (int, float))
            or not math.isfinite(weight_floor_fraction)
            or not 0. <= weight_floor_fraction <= 1.):
        raise ValueError('weight_floor_fraction must lie in [0, 1]')
    if binding is None or binding['model'] != 'alp-fermion':
        raise ValueError('Weighted sampling requires the explicit ALP exHad binding')
    llp._exhad_pooled_indices = ()
    if check_mass(binding, mass) != 'matched':
        raise ValueError('Weighted sampling is defined only inside the matched support')

    pooled = set()
    for index, row in enumerate(pdgs):
        signature = tuple(sorted(int(p) for p in row if int(p) != -999))
        if signature in binding['signatures']:
            pooled.add(index)
    missing = {index for index in pooled - set(selected) if branching[index] > 0}
    if missing:
        raise ValueError(
            'exHad requires the complete hadronic row pool; select all hadronic rows')

    expected = _owner_probabilities_at(binding['owner_probabilities'], mass)
    observed = {
        binding['signatures'][tuple(sorted(int(p) for p in pdgs[index] if int(p) != -999))]:
            float(branching[index])
        for index in pooled}
    total = math.fsum(observed.values())
    if total > 0 and (set(observed) != set(expected) or any(
            not math.isclose(observed[key] / total, expected[key],
                             rel_tol=2e-8, abs_tol=5e-10)
            for key in expected)):
        raise ValueError(
            'EventCalc hadronic branching ratios disagree with the sealed exHad input')

    sizes = decay_module.distribute_events(
        size, [branching[index] / visible for index in selected])
    needed = sum(int(count) for index, count in zip(selected, sizes) if index in pooled)
    if needed:
        weighted_pool = _generator(binding).generate_weighted(
            float(mass), needed, seed=int(seed),
            weight_floor_fraction=weight_floor_fraction)
        pool = [[value for particle in event for value in particle]
                for event in weighted_pool['events']]
    else:
        weighted_pool = {'raw_weights': [], 'normalization_groups': []}
        pool = []

    result, cursor = [], 0
    raw_weights, groups = [], []
    for index, count in zip(selected, sizes):
        count = int(count)
        if index in pooled:
            result.extend(pool[cursor:cursor + count])
            if count:
                raw_weights.extend(weighted_pool['raw_weights'][cursor:cursor + count])
                groups.extend(weighted_pool['normalization_groups'][cursor:cursor + count])
            cursor += count
        elif count:
            raw_weights.extend([1.] * count)
            groups.extend(['external'] * count)
            rows, _ = decay_module.simulateDecays_rest_frame(
                mass, pdgs, branching, count, matrix_elements, [index],
                float(branching[index]))
            for row in rows:
                result.append([value for j in range(0, len(row), 6)
                               if int(row[j + 5]) != -999
                               for value in row[j:j + 6]])
    if len(result) != size or cursor != needed:
        raise RuntimeError('exHad row pooling lost or duplicated events')
    if len(raw_weights) != size or len(groups) != size:
        raise RuntimeError('Importance weights do not align with pooled/native events')
    llp._exhad_pooled_indices = tuple(sorted(pooled))
    return decay_module.pad_processed_events(result), sizes, {
        'raw_weights': np.asarray(raw_weights),
        'normalization_groups': np.asarray(groups)}
