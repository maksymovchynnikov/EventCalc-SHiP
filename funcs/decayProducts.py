# decayProducts.py

import sys
import os
import ctypes
import importlib
import math
import re
import tempfile
import multiprocessing as mp
from pathlib import Path
import numpy as np
import time  # Import time module for timing

from . import TwoBodyDecay, ThreeBodyDecay, FourBodyDecay, NBodyDecay
from . import PDG
from . import thresholds

_PYTHIA8 = None
_PYTHIA8_XML = None
# One configured Pythia object per process, reused by every decay block.
_PYTHIA_INSTANCE = None
_PYTHIA_INSTANCE_MODULE = None
DEFAULT_PYTHIA8_CHUNK_SIZE = 1000
DEFAULT_PYTHIA8_HADRONIZATION_ATTEMPTS = 10

# Pythia accepts explicit seeds below 9e8.  Keep enough headroom for exhad's
# worker offsets and its separate resonance-decayer stream (+54321).
_EXHAD_SEED_LIMIT = 899_900_000

# Particles that this code either treats as detector-level products directly
# or explicitly keeps stable when routing an event through PYTHIA8.
STABLE_WITHOUT_PYTHIA = {
    11, -11,          # e-/e+
    13, -13,          # mu-/mu+
    12, -12, 14, -14, 16, -16,
    22,               # gamma
    211, -211,        # charged pions
    321, -321,        # charged kaons
    130,              # K_L
}

# A fixed-mass backend replaces the rows that list quarks or gluons.  Matched
# spin-zero cards additionally declare an explicit aggregate set of hadronic
# rows; tau pairs and other rows the backend does not own keep the identity
# EventCalc sampled for them.
QUARK_PDG_IDS = frozenset({1, 2, 3, 4, 5, 6})

# Masses used by the exported HNL partonic matrix elements.  The ordinary
# particle database intentionally stores Pythia-like constituent masses, but
# using those in HNL phase space spuriously removes the physical low-W
# continuum (e.g. W_ud < 0.66 GeV).  They are applied to every partonic HNL
# row at every multiplicity.  The hadronic threshold itself is enforced
# separately by the charge-aware two-hadron rule in funcs.thresholds, which is
# where the constant is stated; this name is the same object read from there.
HNL_CURRENT_QUARK_MASS = thresholds.HNL_CURRENT_QUARK_MASSES

# The stock comparator for the fermion-coupled ALP must not inherit the
# explicit proton- and neutron-pair rows from EventCalc.  Those rows belong to
# the physical decay model.  For the comparator, every hadronic event is
# instead assigned to one of the inclusive partonic rows and passed through
# unmodified Pythia 8.317.  Keeping all five rows in the pooled ownership set
# preserves the total hadronic allocation and, consequently, the exact parent
# records shared with the matched sample.
ALP_STOCK_PYTHIA_PROCESS_LABEL = "Jets-stock"
ALP_STOCK_PYTHIA_MIN_MASS_GEV = 1.911
_ALP_STOCK_POOLED_ROW_BY_SIGNATURE = {
    (21, 21): "Jets-GG",
    (-3, 3): "Jets-ss",
    (-4, 4): "Jets-cc",
    (-2212, 2212): "ppbar",
    (-2112, 2112): "nnbar",
}
_ALP_STOCK_PARTONIC_LABELS = frozenset(
    {"Jets-GG", "Jets-ss", "Jets-cc"}
)
_ALP_STOCK_NONHADRONIC_SIGNATURES = frozenset({
    (-11, 11),
    (-13, 13),
    (-15, 15),
    (22, 22),
})
_ALP_STOCK_POOL_SEED_COORDINATE = 0x53544F43

# The output label of a row whose final states come from the common matched
# exhad pool.  Each such row carries its own name after the colon, so two
# pooled rows never share one output block.
MATCHED_PROCESS_LABEL = "Jets-matched"


def derive_exhad_seed(base_seed=1, batch_index=0, channel_index=0):
    """Return a deterministic Pythia-safe seed for one exhad channel call.

    EventCalc may invoke exhad once for each partonic row.  Giving every row
    the same default seed makes the shorter samples literal prefixes of the
    longer ones.  A SeedSequence keyed by the user/base seed, scan-point
    (batch) index, and original decay-table row keeps reruns reproducible
    while assigning distinct streams to those calls.
    """
    base_seed = int(base_seed)
    batch_index = int(batch_index)
    channel_index = int(channel_index)
    if base_seed < 0 or batch_index < 0 or channel_index < 0:
        raise ValueError("exhad seed coordinates must be non-negative")
    state = np.random.SeedSequence(
        [base_seed, batch_index, channel_index, 0x45584841]
    ).generate_state(1, dtype=np.uint64)[0]
    return 1 + int(state % _EXHAD_SEED_LIMIT)


def seed_decay_generators(seed):
    """Seed Python/Numba decay kinematics for one table channel."""
    seed = int(seed)
    np.random.seed(seed)
    for module in (TwoBodyDecay, ThreeBodyDecay):
        seed_numba = getattr(module, "seed_random", None)
        if seed_numba is not None:
            seed_numba(seed)


def _positive_int_from_env(name, default):
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _pythia_worker_count(n_events, chunk_size):
    # One worker by default, which takes the sequential path and is reproducible
    # from the seed alone.  Above one worker the events of a block still depend on
    # the worker count, so a run that must reproduce keeps this at 1.
    raw_value = os.environ.get("PYTHIA8_N_WORKERS", "1").strip().lower()
    if raw_value == "auto":
        max_workers = os.cpu_count() or 1
        n_chunks = max(1, (n_events + chunk_size - 1) // chunk_size)
        return min(max_workers, n_chunks)

    try:
        workers = int(raw_value)
    except ValueError:
        workers = 1
    return max(1, workers)


def _multiprocessing_context():
    # Fork avoids re-importing the interactive simulate.py top level on macOS/Linux.
    if "fork" not in mp.get_all_start_methods():
        return None
    return mp.get_context("fork")


def _pythia_chunk_seed(seed, chunk_index):
    """Return the stream one chunk of a decay block runs on.

    A block is cut into chunks of a fixed size and every chunk is given the
    stream its index names, so the events a seed produces do not depend on
    how many workers processed them, or on which worker took which chunk.
    The first chunk runs on the block's own seed, so a block that fits in one
    chunk runs exactly the stream its seed names.
    """
    chunk_index = int(chunk_index)
    if chunk_index == 0:
        return int(seed)
    state = np.random.SeedSequence(
        [int(seed), chunk_index, 0x50595448]
    ).generate_state(1, dtype=np.uint64)[0]
    # Pythia accepts explicit seeds below 9e8.
    return 1 + int(state % 899900000)


def _pythia8_xml_version(xmldoc):
    """Read the Pythia version declared by one XML payload."""
    version_file = Path(xmldoc) / "Version.xml"
    try:
        text = version_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            "Pythia XML payload is incomplete: %s is not readable" %
            version_file) from exc
    match = re.search(
        r'name=["\']Pythia:versionNumber["\'][^>]*'
        r'default=["\']([^"\']+)["\']', text)
    if match is None:
        raise RuntimeError(
            "Pythia XML payload does not declare Pythia:versionNumber in %s" %
            version_file)
    return match.group(1)


def _candidate_pythia8_xml_dirs(module):
    """Yield plausible XML trees for the imported Python binding.

    An explicit ``PYTHIA8DATA`` is authoritative and is therefore checked
    without silently falling back.  Otherwise we inspect ``PYTHIA8_DIR``, the
    active Python prefix, and parents of the binding module.  The latter
    covers the layout installed by the ``pythia8mc`` wheel.
    """
    explicit = os.environ.get("PYTHIA8DATA")
    if explicit:
        yield Path(explicit).expanduser()
        return

    pythia_dir = os.environ.get("PYTHIA8_DIR")
    if pythia_dir:
        yield Path(pythia_dir).expanduser() / "share" / "Pythia8" / "xmldoc"

    yield Path(sys.prefix) / "share" / "Pythia8" / "xmldoc"
    module_file = getattr(module, "__file__", None)
    if module_file:
        for parent in Path(module_file).resolve().parents:
            yield parent / "share" / "Pythia8" / "xmldoc"


def _pythia8_accepts_xml(module, xmldoc):
    """Ask the binding itself whether it can run one XML payload.

    Pythia reads the payload named by ``PYTHIA8DATA`` in preference to the
    one a caller passes, so the probe sets that variable to the candidate and
    restores it afterwards.  ``checkVersion`` is the library's own verdict on
    the pairing.  The version Pythia was compiled from is not returned by any
    call, only printed in the abort text, so the probe runs with the process
    standard output captured and reads the version back out of it.

    Returns ``(accepted, binding version)``; the version is ``None`` when the
    library printed no abort text to read it from.
    """
    previous = os.environ.get("PYTHIA8DATA")
    os.environ["PYTHIA8DATA"] = str(xmldoc)
    try:
        with tempfile.TemporaryFile(mode="w+") as sink:
            saved = os.dup(1)
            try:
                sys.stdout.flush()
                os.dup2(sink.fileno(), 1)
                accepted = bool(module.Pythia("", False).checkVersion())
            finally:
                # Pythia ends its abort lines with a newline rather than a
                # flush, so the C++ stream is emptied before fd 1 is restored.
                ctypes.CDLL(None).fflush(None)
                os.dup2(saved, 1)
                os.close(saved)
            sink.seek(0)
            printed = sink.read()
    finally:
        if previous is None:
            os.environ.pop("PYTHIA8DATA", None)
        else:
            os.environ["PYTHIA8DATA"] = previous

    match = re.search(r"in code\s+([0-9.]+)", printed)
    return accepted, (match.group(1) if match else None)


def _resolve_pythia8_xml(module):
    """Return the XML payload to run this binding with, or None for its own.

    EventCalc pins no Pythia version of its own: it runs whichever version
    the installed binding was built from.  What it must not do is pair that
    binding with a payload of another version, which Pythia refuses in
    ``Pythia::checkVersion``.  A candidate is therefore used only if the
    binding accepts it, a candidate the binding rejects is reported with both
    versions and the file that declares the payload's, and a machine that
    offers no payload of its own leaves the binding the one it was compiled
    to find.
    """
    seen = set()
    rejected = []
    for candidate in _candidate_pythia8_xml_dirs(module):
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        version_file = candidate / "Version.xml"
        if not version_file.is_file():
            continue
        declared = _pythia8_xml_version(candidate)
        try:
            accepted, binding_version = _pythia8_accepts_xml(module, candidate)
        except Exception as exc:
            raise RuntimeError(
                "Could not initialize the Pythia binding %s with XML payload "
                "%s" % (getattr(module, "__file__", "<unknown>"),
                        candidate)) from exc
        if accepted:
            return candidate
        rejected.append(
            "%s declares Pythia %s, but the binding %s is Pythia %s" % (
                version_file, declared,
                getattr(module, "__file__", "<unknown>"),
                binding_version or "of another version"))

    if rejected:
        raise RuntimeError(
            "Pythia will not run a binding against an XML payload of another "
            "version: %s.  Set PYTHIA8DATA to the share/Pythia8/xmldoc "
            "directory of the installation that binding came from." %
            "; ".join(rejected))
    return None


def _import_pythia8_binding():
    lib_path = os.environ.get("PYTHIA8_LIB")
    if lib_path and lib_path not in sys.path:
        sys.path.insert(0, lib_path)

    failures = []
    # The PyPI wheel exposes ``pythia8mc``.  CERN's official LCG build
    # exposes the same Pythia API as ``pythia8``.  Accept either spelling and
    # let ``load_pythia8`` below pair it with a payload of its own version.
    for module_name in ("pythia8mc", "pythia8"):
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            failures.append(exc)
    raise ImportError(
        "Pythia is required for the selected decay channel. Install the "
        "pythia8mc wheel, use a CERN LCG Pythia binding, or set PYTHIA8_LIB "
        "to the directory containing one of those modules."
        ) from failures[-1]


def load_pythia8():
    """
    Import pythia8 only when a selected decay channel requires showering,
    hadronization, or decays of unstable products.
    """
    global _PYTHIA8, _PYTHIA8_XML
    if _PYTHIA8 is None:
        pythia8_module = _import_pythia8_binding()
        xmldoc = _resolve_pythia8_xml(pythia8_module)
        if xmldoc is not None:
            # Name the payload on the constructor instead of exporting
            # PYTHIA8DATA: a subprocess spawned later inherits the environment,
            # and a release built against another payload dies when it reads
            # this one.
            _PYTHIA8_XML = str(xmldoc)
        _PYTHIA8 = pythia8_module
    return _PYTHIA8


def _pythia_for_block(seed):
    """Return this process's Pythia object, configured, with its stream set.

    Constructing a Pythia object re-reads the whole XML payload from disk,
    which costs far more than the events of one decay block.  The object
    carries nothing from one event to the next except its random stream, so
    one instance serves the whole process and every block declares its own
    settings and its own stream on it.  A newly built instance takes the seed
    through ``Random:seed`` at ``init``; an instance that is already
    initialized takes the same seed through ``rndm.init``, which is the same
    stream.  A block with no seed keeps the behaviour it had when every block
    built its own object: ``rndm.init(-1)`` is Pythia's default stream.
    """
    global _PYTHIA_INSTANCE, _PYTHIA_INSTANCE_MODULE
    pythia8 = load_pythia8()
    pythia = _PYTHIA_INSTANCE
    reused = pythia is not None and _PYTHIA_INSTANCE_MODULE is pythia8
    if not reused:
        pythia = pythia8.Pythia(_PYTHIA8_XML, False) if _PYTHIA8_XML else pythia8.Pythia()
    #pythia.readString("Print:quiet = on")  # Suppress banners and output
    pythia.readString("ProcessLevel:all = off")
    pythia.readString("PartonLevel:all = on")
    pythia.readString("HadronLevel:all = on")
    # Pythia's own energy-momentum closure test on every completed event, at a
    # tolerance fifty times tighter than the 1e-4 its ErrorChecks.xml defaults
    # to.  An event that fails it is rejected by Pythia and the block retries
    # it rather than consuming the inconsistent record.
    pythia.readString("Check:event = on")
    pythia.readString("Check:epTolErr = 2e-6")
    if seed is not None:
        seed = int(seed)
        if not 1 <= seed <= _EXHAD_SEED_LIMIT:
            raise ValueError(
                "raw-Pythia seed must lie in [1, %d]" %
                _EXHAD_SEED_LIMIT)
        pythia.readString("Random:setSeed = on")
        pythia.readString("Random:seed = %d" % seed)
    # Keep certain particles stable
    stable_particles = [13, -13, 211, -211, 321, -321, 130]
    for pid in stable_particles:
        pythia.readString(f"{pid}:mayDecay = off")
    if reused:
        # ``Random:seed`` is read when Pythia initializes, so an instance that
        # is already initialized is given the block's stream directly.
        pythia.rndm.init(-1 if seed is None else seed)
    else:
        if not pythia.init():
            raise RuntimeError("Pythia initialization failed")
        _PYTHIA_INSTANCE = pythia
        _PYTHIA_INSTANCE_MODULE = pythia8
    return pythia


def _as_2d_decay_array(decay_events):
    decay_events = np.asarray(decay_events, dtype=np.float64)
    if decay_events.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    if decay_events.ndim == 1:
        decay_events = decay_events.reshape(1, -1)
    return decay_events


def channel_requires_pythia(pdg_list):
    """
    Return True if this decay channel has any products that need PYTHIA8.
    """
    for pdg_id in pdg_list:
        pdg_id = int(pdg_id)
        if pdg_id == -999:
            continue
        if pdg_id not in STABLE_WITHOUT_PYTHIA:
            return True
    return False


def channel_is_partonic(pdg_list):
    """Return True only for rows made entirely of quarks and/or gluons.

    This predicate gates the fixed-mass exhad replacement.  Mixed HNL rows
    are deliberately excluded here and continue through their dedicated
    per-event-W handling below.
    """
    ids = [int(pdg_id) for pdg_id in pdg_list if int(pdg_id) != -999]
    return bool(ids) and all(
        pdg_id == 21 or abs(pdg_id) in QUARK_PDG_IDS for pdg_id in ids)


def _row_has_positive_branching(BrRatio, table_index):
    """Return whether one selected decay row has positive physical weight.

    ``None`` preserves the historical structural-only behavior for callers
    that do not have mass-dependent branching ratios available.
    """
    if BrRatio is None:
        return True
    try:
        value = float(BrRatio[table_index])
    except (IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            "branching ratios must contain every selected decay row"
        ) from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "selected branching ratios must be finite and non-negative"
        )
    return value > 0.0


def selected_sample_requires_pythia(
        PDGdecay, BrRatio, selected_decay_indices):
    """Whether a positive-rate selected row needs baseline Pythia.

    This is a rate-level routing statement, independent of a finite Monte
    Carlo allocation.  It lets output provenance distinguish a genuinely
    direct EventCalc sample from the default-fragmentation/decay path.
    """
    for table_index in selected_decay_indices:
        if (
                _row_has_positive_branching(BrRatio, table_index)
                and channel_requires_pythia(
                    _active_pdg_ids(PDGdecay[table_index]))):
            return True
    return False


def _active_pdg_ids(decay_modes):
    """Return the non-padding PDG ids from one decay-table row."""
    decay_modes = np.asarray(decay_modes)
    return decay_modes[decay_modes != -999]


def _pdg_signature(decay_modes):
    """Return the sorted active-PDG signature of one decay-table row."""
    return tuple(sorted(
        int(value) for value in _active_pdg_ids(decay_modes)
    ))


def _alp_stock_pythia_plan(
        mass,
        PDGdecay,
        BrRatio,
        selected_decay_indices,
        size_per_channel,
        llp_name,
):
    """Describe the complete ALP hadronic pool used by stock Pythia.

    The five physical hadronic rows retain their original EventCalc event
    counts solely to preserve the shared parent sample.  Their final states
    are discarded.  The complete pooled count is reassigned to the positive
    inclusive partonic rows according to their relative branching fractions.
    """
    if llp_name != "ALP-fermion":
        raise ValueError(
            "the stock-Pythia pool is defined only for ALP-fermion"
        )
    mass = float(mass)
    if mass < ALP_STOCK_PYTHIA_MIN_MASS_GEV - 1.0e-12:
        raise ValueError(
            "the ALP stock-Pythia pool starts at %.3f GeV"
            % ALP_STOCK_PYTHIA_MIN_MASS_GEV
        )
    if len(selected_decay_indices) != len(size_per_channel):
        raise ValueError(
            "stock-Pythia planning requires one count per selected row"
        )

    pooled_positions = []
    source_positions = []
    unsupported = []
    for position, table_index in enumerate(selected_decay_indices):
        count = int(size_per_channel[position])
        if count <= 0:
            continue
        signature = _pdg_signature(PDGdecay[table_index])
        label = _ALP_STOCK_POOLED_ROW_BY_SIGNATURE.get(signature)
        if label is not None:
            pooled_positions.append(position)
            if label in _ALP_STOCK_PARTONIC_LABELS:
                source_positions.append((position, table_index, label))
            continue
        if signature not in _ALP_STOCK_NONHADRONIC_SIGNATURES:
            unsupported.append((position, table_index, signature))

    if unsupported:
        labels = ", ".join(
            "%d:%s" % (table_index, signature)
            for _, table_index, signature in unsupported
        )
        raise ValueError(
            "stock Pythia received positive ALP rows outside the complete "
            "hadronic pool: " + labels
        )
    if not pooled_positions:
        raise ValueError("stock Pythia received no ALP hadronic events")
    if not source_positions:
        raise ValueError(
            "stock Pythia has no positive ALP partonic source row"
        )

    pool_size = sum(
        int(size_per_channel[position]) for position in pooled_positions
    )
    source_weights = [
        float(BrRatio[table_index])
        for _, table_index, _ in source_positions
    ]
    if (
            any(not math.isfinite(weight) or weight < 0.0
                for weight in source_weights)
            or math.fsum(source_weights) <= 0.0):
        raise ValueError(
            "stock-Pythia partonic source weights are not positive and finite"
        )
    source_allocations = distribute_events(pool_size, source_weights)
    sources = []
    for (position, table_index, label), allocation in zip(
            source_positions, source_allocations):
        allocation = int(allocation)
        if allocation <= 0:
            continue
        pdgs = tuple(
            int(value) for value in _active_pdg_ids(PDGdecay[table_index])
        )
        masses = tuple(float(PDG.get_mass(pdg)) for pdg in pdgs)
        if len(pdgs) != 2 or sum(masses) > mass + 1.0e-12:
            raise ValueError(
                "stock-Pythia source %s is not open at m=%.6g GeV"
                % (label, mass)
            )
        sources.append({
            "position": position,
            "table_index": table_index,
            "label": label,
            "pdgs": pdgs,
            "masses": masses,
            "allocation": allocation,
        })
    if sum(source["allocation"] for source in sources) != pool_size:
        raise AssertionError(
            "stock-Pythia source allocation does not fill the hadronic pool"
        )
    return {
        "pooled_positions": tuple(pooled_positions),
        "pool_size": pool_size,
        "sources": tuple(sources),
    }


def _generate_alp_stock_pythia_pool(
        mass,
        PDGdecay,
        BrRatio,
        selected_decay_indices,
        size_per_channel,
        llp_name,
        seed,
        batch_index,
):
    """Generate the complete ALP stock-Pythia final-state pool."""
    plan = _alp_stock_pythia_plan(
        mass,
        PDGdecay,
        BrRatio,
        selected_decay_indices,
        size_per_channel,
        llp_name,
    )
    partonic_batches = []
    for source in plan["sources"]:
        source_seed = derive_exhad_seed(
            seed,
            batch_index,
            _ALP_STOCK_POOL_SEED_COORDINATE + source["table_index"],
        )
        seed_decay_generators(source_seed)
        pdg1, pdg2 = source["pdgs"]
        mass1, mass2 = source["masses"]
        partonic_batches.append(TwoBodyDecay.decay_products(
            float(mass),
            source["allocation"],
            mass1,
            mass2,
            pdg1,
            pdg2,
            PDG.get_charge(pdg1),
            PDG.get_charge(pdg2),
            PDG.get_stability(pdg1),
            PDG.get_stability(pdg2),
        ))
    partonic = np.vstack(partonic_batches)
    if len(partonic) != plan["pool_size"]:
        raise AssertionError(
            "stock-Pythia input pool has the wrong number of events"
        )

    shuffle_seed = derive_exhad_seed(
        seed,
        batch_index,
        _ALP_STOCK_POOL_SEED_COORDINATE + 0x10000,
    )
    permutation = np.random.default_rng(shuffle_seed).permutation(
        len(partonic)
    )
    pythia_seed = derive_exhad_seed(
        seed,
        batch_index,
        _ALP_STOCK_POOL_SEED_COORDINATE + 0x20000,
    )
    events = process_events_with_pythia(
        partonic[permutation],
        float(mass),
        seed=pythia_seed,
    )
    if len(events) != plan["pool_size"]:
        raise AssertionError(
            "stock Pythia returned the wrong number of ALP events"
        )
    return events, plan


def _row_output_name(table_index, decay_channels):
    """The decay table's own name for one row, when the caller supplied them."""
    if decay_channels is None:
        return None
    try:
        name = decay_channels[table_index]
    except (IndexError, KeyError, TypeError):
        return None
    if name is None:
        return None
    name = str(name).strip()
    return name or None


def _gated_rates(mass, PDGdecay, BrRatio, hnl=False, decay_channels=None,
                 announce=True):
    """The tabulated rates with every row closed at ``mass`` set to zero.

    Between two tabulated masses an interpolant can give a small positive rate
    to a row whose daughters do not fit inside the parent.  Sampling such a row
    produces a particle with a negative squared momentum, so the rate is
    removed here, before the events are distributed, on every route.  A removed
    row is zeroed and never rescaled: EventCalc keeps ownership of the rates.

    Returns ``(gated_rates, removed_indices)``.
    """
    gated, leaked = thresholds.gate_rates(mass, PDGdecay, BrRatio, hnl=hnl)
    if announce and len(leaked):
        names = [_row_output_name(int(index), decay_channels) or str(int(index))
                 for index in leaked]
        print("Threshold gate: %d tabulated row(s) are below their own "
              "threshold at m = %g GeV and were set to exactly zero: %s"
              % (len(leaked), float(mass), ", ".join(names)))
    return gated, leaked


def _exclusive_row_positions(exhad, PDGdecay, selected_decay_indices,
                             decay_channels=None, resolve_labels=True):
    """Map selected-row positions onto the exclusive exhad rows they own.

    Below the matched support exhad still supplies exact final states, one
    named row at a time.  A row it does not own -- a lepton pair, two photons
    -- stays with EventCalc's own sampler.

    Several exclusive rows can carry one PDG signature (the kaon-pion towers
    do), and the decay table's own row name is what names one of them.  With
    ``resolve_labels`` false the answer is only which positions exhad owns,
    which a caller holding no row names can still ask.
    """
    owned = {}
    for position, table_index in enumerate(selected_decay_indices):
        pdg_list = _active_pdg_ids(PDGdecay[table_index])
        try:
            label = exhad.exclusive_row_label(
                pdg_list,
                row_label=_row_output_name(table_index, decay_channels))
        except ValueError:
            if resolve_labels:
                raise
            # The signature names several exclusive rows; the row is owned
            # whichever of them it turns out to be.
            owned[position] = None
            continue
        if label is not None:
            owned[position] = label
    return owned


def _pooled_row_name(exhad, PDGdecay, table_index, decay_channels):
    """A name that tells one pooled row from another in the output.

    The decay table's own row name is the answer when the caller supplied the
    table; the matched backend's own row identifier is the next one, and the
    table row number is the last.  Two pooled rows never share a name, because
    a text export keys its process blocks by name and the second block of a
    repeated name is lost on read-back.
    """
    name = _row_output_name(table_index, decay_channels)
    if name is not None:
        return name
    if exhad is not None and exhad.uses_portable_model1_backend():
        label = exhad.portable_model1_eventcalc_row_label(
            _active_pdg_ids(PDGdecay[table_index]))
        if label:
            return str(label)
    return "row%d" % int(table_index)


def _fixed_exhad_context(mass, PDGdecay, selected_decay_indices,
                         llp_name=None, particle_path=None,
                         exhad_variant=None, *, decay_channels=None):
    """Resolve fixed-mass exhad and the eligible selected-row positions.

    Below the matched support the eligible positions are the rows exhad owns
    as named exclusive rows; inside it they are the rows the fixed-mass pool
    replaces.

    HNL is deliberately excluded: its mixed three-body rows must retain the
    event-by-event hadronic invariant mass and use the dedicated handlers.
    """
    from funcs import exhadDecays as exhad
    exhad.set_selection(llp_name, particle_path, exhad_variant)

    if exhad.is_hnl_bench():
        return exhad, []

    if exhad.exclusive_enabled(mass):
        owned = _exclusive_row_positions(
            exhad, PDGdecay, selected_decay_indices, decay_channels,
            resolve_labels=False)
        if not owned:
            return None, []
        return exhad, sorted(owned)

    if exhad.uses_portable_model1_backend():
        config = exhad.portable_model1_configuration()
        generation_end = exhad.generation_window()[1]
        if float(mass) > generation_end:
            raise ValueError(
                'exhad generates %s decays for a mother mass up to %g GeV; %g GeV was '
                'requested. Select raw Pythia explicitly above that mass.'
                % (exhad.get_bench(), generation_end, float(mass)))
        if not exhad.enabled(mass):
            return None, []
        required = {
            row["authority_row_id"]
            for row in config["eventcalc_rows"]
        }
        row_positions = []
        observed = {}
        for position, table_index in enumerate(selected_decay_indices):
            pdg_list = _active_pdg_ids(PDGdecay[table_index])
            label = exhad.portable_model1_eventcalc_row_label(pdg_list)
            if label is None:
                continue
            if label in observed:
                raise ValueError(
                    "portable Model 1 EventCalc row %r occurs more than once"
                    % label)
            observed[label] = table_index
            row_positions.append(position)
        missing = required - set(observed)
        if missing:
            raise ValueError(
                "portable Model 1 requires the complete hadronic row pool; "
                "selected channels omit: %s" % ", ".join(sorted(missing)))
        if set(observed) != required:
            raise AssertionError(
                "portable Model 1 selected rows exceed the sealed pool")
        return exhad, row_positions

    row_positions = []
    for position, table_index in enumerate(selected_decay_indices):
        pdg_list = _active_pdg_ids(PDGdecay[table_index])
        if (channel_is_partonic(pdg_list)
                and exhad.fixed_partonic_row_supported(pdg_list)):
            row_positions.append(position)
    if not row_positions or not exhad.enabled(mass):
        return None, []
    return exhad, row_positions


def _validate_fixed_exhad_pooled_branching(
        exhad, mass, BrRatio, selected_decay_indices, row_positions, *,
        PDGdecay):
    """Require EventCalc and the selected complete-pool authority to agree."""

    if exhad is None:
        return
    if exhad.uses_portable_model1_backend():
        if BrRatio is None:
            raise ValueError(
                "portable Model 1 pooling requires EventCalc branching ratios")
        expected = exhad.portable_model1_eventcalc_authority_probabilities(
            mass)
        observed = {}
        for position in row_positions:
            table_index = selected_decay_indices[position]
            label = exhad.portable_model1_eventcalc_row_label(
                _active_pdg_ids(PDGdecay[table_index]))
            if label is None:
                raise RuntimeError(
                    "portable Model 1 pooled row lost its PDG identity")
            observed[label] = float(BrRatio[table_index])
        total = math.fsum(observed.values())
        if total == 0.0 and all(value == 0.0 for value in observed.values()):
            return
        if not math.isfinite(total) or total <= 0.0:
            raise RuntimeError(
                "portable Model 1 EventCalc row pool has invalid weight")
        normalized = {key: value / total for key, value in observed.items()}
        if set(normalized) != set(expected):
            raise RuntimeError(
                "portable Model 1 EventCalc row inventory differs from the "
                "outer owner authority")
        mismatch = {
            key: (normalized[key], expected[key])
            for key in expected
            if not math.isclose(normalized[key], expected[key],
                                rel_tol=2.0e-8, abs_tol=5.0e-10)
        }
        if mismatch:
            raise RuntimeError(
                "portable Model 1 EventCalc branching ratios differ from "
                "the outer owner authority: %r" % mismatch)


def effective_exhad_bench(mass, PDGdecay, selected_decay_indices,
                          BrRatio=None, llp_name=None, particle_path=None,
                          exhad_variant=None):
    """Return the matched generator with positive selected rate support.

    For fixed-mass LLPs this is non-``None`` only when the mass is inside the
    benchmark window, or inside the exclusive window below it, and at least
    one selected, owned row has positive branching ratio.  HNL likewise
    requires a selected supported residual current with positive rate.  Merely
    resolving a card, or selecting a zero-weight row, must not label a sample
    as matched exhad.

    The rates are read the way the decay call reads them: a row that is closed
    at this mass carries no rate, so it cannot be the support that names a
    route.

    With ``BrRatio=None`` the answer rests on the row structure alone, for a
    caller that has no rate vector to offer.
    """
    exhad, row_positions = _fixed_exhad_context(
        mass, PDGdecay, selected_decay_indices, llp_name, particle_path,
        exhad_variant)
    if exhad is None:
        return None
    if BrRatio is not None:
        BrRatio, _removed = _gated_rates(
            mass, PDGdecay, BrRatio, announce=False)
    if exhad.is_hnl_bench():
        for table_index in selected_decay_indices:
            spec = exhad.hnl_channel_spec(
                _active_pdg_ids(PDGdecay[table_index]))
            if (
                    spec is not None
                    and spec["supported"]
                    and _row_has_positive_branching(
                        BrRatio, table_index)):
                return exhad.get_bench()
        return None
    if any(
            _row_has_positive_branching(
                BrRatio, selected_decay_indices[position])
            for position in row_positions):
        return exhad.get_bench()
    return None


def prepare_fixed_exhad_pool(mass, size, PDGdecay, selected_decay_indices,
                             BrRatio=None, br_visible_val=None, llp_name=None,
                             particle_path=None, exhad_variant=None, seed=1,
                             batch_index=0):
    """Generate one reusable fixed-mass exhad event pool, when applicable.

    ``size`` is normally the scan's full resampling size.  When ``BrRatio``
    and ``br_visible_val`` are supplied, the pool holds the partonic share of
    that total size plus the headroom of :func:`group_allocation_ceiling`,
    which keeps a fully partonic pool from being generated while covering the
    multinomial spread of the share each individual lifetime draws.  A caller
    can pass the returned rectangular array to
    :func:`simulateDecays_rest_frame` for every lifetime at this mass; each
    point consumes a prefix, so fixed-mass generator startup is paid once.
    ``None`` means this selection has no fixed-mass exhad rows (raw Pythia,
    HNL, out-of-window mass, or no selected partonic channel).
    """
    size = int(size)
    if size < 0:
        raise ValueError("fixed exhad pool size must be non-negative")

    exhad, row_positions = _fixed_exhad_context(
        mass, PDGdecay, selected_decay_indices, llp_name, particle_path,
        exhad_variant)
    if exhad is None or not row_positions:
        return None
    if exhad.exclusive_enabled(mass):
        # Below the matched support every hadronic row is realized as its own
        # named exclusive row at the decay call, so there is no one pool for a
        # whole mass to reuse.
        return None
    _validate_fixed_exhad_pooled_branching(
        exhad, mass, BrRatio, selected_decay_indices, row_positions,
        PDGdecay=PDGdecay)

    pool_size = size
    if BrRatio is not None:
        if br_visible_val is None or br_visible_val <= 0:
            raise ValueError(
                "br_visible_val must be positive when BrRatio is supplied")
        # The decay call removes the rows that are closed at this mass before
        # it distributes events, so the share this pool has to cover is the
        # share of the rates that survive that removal.
        rates, _removed = _gated_rates(
            mass, PDGdecay, BrRatio, announce=False)
        selected_ratios = [float(rates[index])
                           for index in selected_decay_indices]
        if any(not math.isfinite(ratio) or ratio < 0.
               for ratio in selected_ratios):
            raise ValueError(
                "channel rates must all be finite and non-negative")
        visible_rate = math.fsum(selected_ratios)
        if visible_rate <= 0.:
            raise ValueError(
                "no selected decay channel has a positive rate at "
                "m=%.6g GeV" % float(mass))
        partonic_rate = math.fsum(selected_ratios[position]
                                  for position in row_positions)
        pool_size = group_allocation_ceiling(size, partonic_rate / visible_rate)

    if pool_size == 0:
        return np.empty((0, 0), dtype=np.float64)

    first_table_index = selected_decay_indices[row_positions[0]]
    pool_seed = derive_exhad_seed(seed, batch_index, first_table_index)
    pool = np.asarray(
        exhad.process_events_with_exhad(pool_size, mass, seed=pool_seed),
        dtype=np.float64)
    if pool.ndim == 1:
        pool = pool.reshape(1, -1)
    if pool.ndim != 2 or len(pool) < pool_size:
        available = len(pool) if pool.ndim >= 1 else 0
        raise AssertionError(
            "fixed exhad generator returned an insufficient pool: "
            f"need {pool_size}, got {available}")
    return pool[:pool_size]


def decay_event_requires_pythia(decay):
    """
    Return True if PYTHIA8 is needed to turn this internal decay record into
    detector-level final-state products.
    """
    num_particles = len(decay) // 8
    for i_particle in range(num_particles):
        idx = i_particle * 8
        pdg_id = int(decay[idx + 5])
        if pdg_id == -999:
            continue
        if pdg_id not in STABLE_WITHOUT_PYTHIA:
            return True
    return False


def convert_events_without_pythia(decay_events_list):
    """
    Convert internal 8-field particle records to the 6-field event format used
    downstream, preserving only already-stable final-state products.
    """
    decay_events = _as_2d_decay_array(decay_events_list)
    if decay_events.size == 0:
        return np.empty((0, 0), dtype=np.float64)
    if decay_events.shape[1] % 8 != 0:
        raise ValueError("Internal decay records must have 8 fields per particle.")

    num_particles = decay_events.shape[1] // 8
    events_by_particle = decay_events.reshape(decay_events.shape[0], num_particles, 8)
    active_mask = events_by_particle[:, :, 5] != -999

    if np.all(active_mask):
        return events_by_particle[:, :, :6].reshape(decay_events.shape[0], num_particles * 6)

    processed_events = []
    for decay in decay_events:
        final_state_particles = []
        for i_particle in range(num_particles):
            idx = i_particle * 8
            pdg_id = int(decay[idx + 5])
            if pdg_id == -999:
                continue
            final_state_particles.extend([
                decay[idx],
                decay[idx + 1],
                decay[idx + 2],
                decay[idx + 3],
                decay[idx + 4],
                pdg_id,
            ])
        processed_events.append(final_state_particles)

    return pad_processed_events(processed_events)


def process_decay_events(decay_events_list, mass, seed=None):
    """Send each decay event down the path its own products require.

    Whether Pythia is needed is a property of the identities in the record,
    so an event whose products are all stable is completed by the direct
    conversion and never enters a Pythia record at all.  Both paths write the
    same six fields per particle, and the events are returned in their input
    order.
    """
    hadronic_positions = []
    stable_positions = []
    for position, decay in enumerate(decay_events_list):
        if decay_event_requires_pythia(decay):
            hadronic_positions.append(position)
        else:
            stable_positions.append(position)

    if not hadronic_positions:
        print("\nAll selected decay products are stable. Skipping Pythia processing.")
        return convert_events_without_pythia(decay_events_list)
    if not stable_positions:
        print(f"\nStarting Pythia processing of {len(decay_events_list)} decay events...")
        return process_events_with_pythia(decay_events_list, mass, seed=seed)

    print(f"\nStarting Pythia processing of {len(hadronic_positions)} decay events; "
          f"{len(stable_positions)} events with only stable products skip it.")
    hadronic_events = process_events_with_pythia(
        [decay_events_list[position] for position in hadronic_positions], mass,
        seed=seed)
    stable_events = convert_events_without_pythia(
        [decay_events_list[position] for position in stable_positions])
    width = max(hadronic_events.shape[1], stable_events.shape[1])
    processed = np.empty((len(decay_events_list), width), dtype=np.float64)
    processed[hadronic_positions] = pad_processed_array(hadronic_events, width)
    processed[stable_positions] = pad_processed_array(stable_events, width)
    return processed


def pad_processed_array(processed_array, target_width):
    """
    Pad a 2-D processed event array to a target 6-field particle width.
    """
    processed_array = np.asarray(processed_array, dtype=np.float64)
    if processed_array.size == 0:
        return np.empty((0, target_width), dtype=np.float64)
    if processed_array.ndim == 1:
        processed_array = processed_array.reshape(1, -1)

    current_width = processed_array.shape[1]
    if current_width == target_width:
        return processed_array
    if target_width < current_width or (target_width - current_width) % 6 != 0:
        raise ValueError("Processed event arrays must be padded in 6-field particle units.")

    missing_particles = (target_width - current_width) // 6
    padding_particle = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -999.0], dtype=np.float64)
    padding = np.tile(padding_particle, missing_particles)
    padding = np.tile(padding, (processed_array.shape[0], 1))
    return np.hstack((processed_array, padding))


def concatenate_padded_event_arrays(processed_arrays):
    """
    Stack per-channel or per-worker processed event arrays after width padding.
    """
    arrays = []
    for array in processed_arrays:
        array = np.asarray(array, dtype=np.float64)
        if array.ndim == 1 and array.size:
            array = array.reshape(1, -1)
        if array.shape[0] > 0:
            arrays.append(array)

    if not arrays:
        return np.empty((0, 0), dtype=np.float64)

    max_width = max(array.shape[1] for array in arrays)
    padded_arrays = [pad_processed_array(array, max_width) for array in arrays]
    return np.vstack(padded_arrays)


def pad_processed_events(processed_events):
    """
    Pad processed 6-field event records so all events have equal length.
    """
    max_n = max((len(event) // 6 for event in processed_events), default=0)
    padding = [0.0, 0.0, 0.0, 0.0, 0.0, -999]

    padded_events = []
    for event in processed_events:
        n = len(event) // 6
        missing = max_n - n
        if missing > 0:
            event = event + padding * missing
        padded_events.append(event)

    return np.array(padded_events, dtype=np.float64)


def distribute_events(total_events, branching_ratios):
    """Draw how many of ``total_events`` events each decay channel receives.

    Which channel an event decays through is itself a random draw from the
    channel probabilities, so the counts of a sample of ``total_events``
    independent events are one draw from the multinomial distribution with
    those probabilities.  A channel whose expected count is a fraction of an
    event then receives an event in that fraction of the samples, and every
    count carries the spread a user sees when the same point is re-run.

    The draw comes from NumPy's global generator, which ``funcs.seeding``
    seeds together with every other stream, so a seeded run repeats exactly.
    """
    branching_ratios = np.asarray(branching_ratios, dtype=float)
    total_events = int(total_events)
    if total_events < 0:
        raise ValueError(
            'A decay sample cannot contain %d events' % total_events)
    if branching_ratios.ndim != 1:
        raise ValueError(
            'Channel rates must form a one-dimensional array; got shape %r'
            % (branching_ratios.shape,))
    if not np.all(np.isfinite(branching_ratios)) or np.any(branching_ratios < 0.):
        raise ValueError(
            'Channel rates must all be finite and non-negative')
    total_rate = float(np.sum(branching_ratios))
    if not total_rate > 0.:
        # Every rate is zero: there is no distribution to draw from.
        raise ValueError(
            'No selected decay channel has a positive rate, so %d events cannot be '
            'distributed over %d channels' % (total_events, branching_ratios.size))
    return np.random.multinomial(total_events, branching_ratios / total_rate)


def group_allocation_ceiling(total_events, probability, tolerance=1e-12):
    """A count the multinomial share of one group of channels stays below.

    The number of events a group receives out of ``total_events`` is binomial
    with parameter ``probability``, so a pool prepared once for a whole mass
    has to cover the largest share the group can plausibly draw rather than its
    mean.  The count returned is the mean plus the departure from it at which
    the chance of the binomial reaching that far has fallen to ``tolerance``,
    and it is never larger than ``total_events``.  The departure comes from a
    standard bound on how fast the upper tail of a sum of independent terms
    falls: with ``L = ln(1 / tolerance)`` and variance ``v = n p (1 - p)``, a
    departure of ``L / 3 + sqrt(L^2 / 9 + 2 L v)`` is reached with probability
    at most ``tolerance``.
    """
    total_events = int(total_events)
    probability = float(probability)
    if total_events < 0:
        raise ValueError('A decay sample cannot contain %d events' % total_events)
    if not 0. <= probability <= 1.:
        raise ValueError('A channel-group probability must lie in [0, 1]; '
                         'got %r' % (probability,))
    if not 0. < tolerance < 1.:
        raise ValueError('The exceedance probability must lie in (0, 1); '
                         'got %r' % (tolerance,))
    mean = total_events * probability
    variance = mean * (1. - probability)
    log_odds = math.log(1. / tolerance)
    deviation = (log_odds / 3.
                 + math.sqrt(log_odds * log_odds / 9. + 2. * log_odds * variance))
    return min(total_events, int(math.ceil(mean + deviation)))

def simulateDecays_rest_frame(mass, PDGdecay, BrRatio, size,
                              Msquared3BodyLLP, selected_decay_indices,
                              br_visible_val, *,
                              llp_name=None, particle_path=None,
                              exhad_variant=None, seed=1, batch_index=0,
                              fixed_exhad_pool=None,
                              stock_pythia_pool=False,
                              return_process_labels=False,
                              hnl=None, primary_only=False,
                              gate_closed_rows=True, decay_channels=None):
    """
    Simulates the decays of a particle in its rest frame, distributing events among selected decay channels.

    llp_name / particle_path identify the selected LLP so the exhad hook can
    activate automatically from its per-LLP card (Distributions/<LLP>/exhad.json);
    both default to None, in which case exhad falls back to the EXHAD_* env vars.
    ``hnl`` reads those same two arguments when it is None; True or False
    answers the question directly.  A heavy neutral lepton's partonic rows are
    sampled with the current-quark masses the exported matrix elements are
    written in, at every multiplicity.

    seed / batch_index identify this EventCalc scan point.  Fixed-mass
    partonic rows are aggregated into one deterministic exhad call.  An
    optional ``fixed_exhad_pool`` prepared for the same mass may be supplied;
    the required prefix is split back across rows in their original order.
    Below the matched support each hadronic row is instead realized as its own
    named exclusive exhad row at the count this sample asks for.  HNL's
    per-event-W path remains independent and uses its channel seed.

    ``gate_closed_rows`` sets the rate of every row whose daughters do not fit
    inside the parent to exactly zero before the events are distributed, on
    every route.  When that removes rate from a selected row, the visible
    branching ratio is recomputed as the sum of the selected rates that
    survive; a removed row is never rescaled into its neighbours.  A selection
    that loses nothing keeps the number the caller passed, so a point with no
    closed row draws exactly the sample it drew before.

    ``decay_channels`` is the decay table's own row names.  They name one of
    the exclusive exhad rows that share a PDG signature, and they name each
    pooled row in the returned output labels.

    ``primary_only`` returns the raw eight-field rest-frame records of the
    selected rows instead of final-state products; no hadronization of any
    kind runs on them.

    When ``return_process_labels`` is true, a third return value provides one
    output-label override per selected input row.  Fixed-mass rows replaced by
    the common matched exhad pool are labelled ``Jets-matched:<row>``, one
    label per row; other entries are ``None`` and retain their physical
    input-channel labels.  Keeping this metadata separate from
    ``size_per_channel`` preserves the established event ordering while
    avoiding false ``Jets-dd``/``Jets-ss`` provenance.
    """
    if hnl is None:
        is_hnl_selection = (llp_name == "HNL" or
                            (particle_path is not None and
                             os.path.basename(os.path.normpath(
                                 str(particle_path))) == "HNL"))
    else:
        is_hnl_selection = bool(hnl)

    def get_particle_properties(pdg_list):
        masses = [PDG.get_mass(pdg) for pdg in pdg_list]
        if is_hnl_selection:
            # Applied to matched and explicitly raw HNL workflows alike, and
            # at every multiplicity: their only intended difference is the
            # hadronic composition, not the sampled phase space.
            masses = [HNL_CURRENT_QUARK_MASS.get(abs(int(pdg)), value)
                      for pdg, value in zip(pdg_list, masses)]
        charges = [PDG.get_charge(pdg) for pdg in pdg_list]
        stabilities = [PDG.get_stability(pdg) for pdg in pdg_list]
        return masses, charges, stabilities

    # Start total timing
    total_start_time = time.time()

    if gate_closed_rows:
        BrRatio, removed = _gated_rates(
            mass, PDGdecay, BrRatio, hnl=is_hnl_selection,
            decay_channels=decay_channels)
        if set(int(index) for index in removed) & set(
                int(index) for index in selected_decay_indices):
            # The visible branching ratio is the sum of what survives.  Nothing
            # is rescaled to make up the removed rate, and the caller's own
            # number is left alone when this selection lost none of it.
            br_visible_val = math.fsum(
                float(BrRatio[index]) for index in selected_decay_indices)
        if not br_visible_val > 0.:
            raise ValueError(
                "No selected decay channel is open at m = %.6g GeV" %
                float(mass))

    # Get branching ratios and decay modes for selected channels
    selected_BrRatios = [BrRatio[idx] for idx in selected_decay_indices]
    selected_PDGdecay = [PDGdecay[idx] for idx in selected_decay_indices]
    if Msquared3BodyLLP is not None:
        selected_Msquared3BodyLLP = [Msquared3BodyLLP[idx] for idx in selected_decay_indices]
    else:
        selected_Msquared3BodyLLP = [None] * len(selected_BrRatios)

    # Normalize the branching ratios
    normalized_br_ratios = [br / br_visible_val for br in selected_BrRatios]

    # Distribute events among the selected decay channels
    size_per_channel = distribute_events(size, normalized_br_ratios)
    print(f"Events per selected decay channel: {size_per_channel}")
    processed_channel_results = []
    primary_events = []
    stable_event_count = 0
    pythia_event_count = 0
    exhad_event_count = 0
    total_generated_events = 0
    decay_sim_time = 0.0
    final_state_time = 0.0

    # Resolve routing once, then generate every selected row exhad owns in a
    # single call.  Inside the matched support that is one fixed-mass pool,
    # whose prefix is partitioned below in original selected-channel order;
    # below it, it is one call for the named exclusive rows.  HNL is excluded
    # by the resolver, and primary_only asks for no final states at all.
    exhad = None
    exhad_rows = []
    exclusive_exhad_rows = {}
    exclusive_exhad_events = {}
    exclusive_exhad_cursors = {}
    if not primary_only:
        exhad, exhad_rows = _fixed_exhad_context(
            mass, PDGdecay, selected_decay_indices, llp_name, particle_path,
            exhad_variant, decay_channels=decay_channels)
    if exhad is not None and exhad.exclusive_enabled(mass):
        exclusive_exhad_rows = _exclusive_row_positions(
            exhad, PDGdecay, selected_decay_indices, decay_channels)
        fixed_exhad_rows = set()
    else:
        _validate_fixed_exhad_pooled_branching(
            exhad, mass, BrRatio, selected_decay_indices, exhad_rows,
            PDGdecay=PDGdecay)
        fixed_exhad_rows = set(exhad_rows)
    fixed_exhad_required = sum(
        int(size_per_channel[position]) for position in fixed_exhad_rows)
    fixed_exhad_events = None
    fixed_exhad_cursor = 0
    if fixed_exhad_required:
        pool_start_time = time.time()
        if fixed_exhad_pool is None:
            first_position = min(fixed_exhad_rows)
            first_table_index = selected_decay_indices[first_position]
            pool_seed = derive_exhad_seed(
                seed, batch_index, first_table_index)
            fixed_exhad_events = np.asarray(
                exhad.process_events_with_exhad(
                    fixed_exhad_required, mass, seed=pool_seed),
                dtype=np.float64)
        else:
            fixed_exhad_events = np.asarray(
                fixed_exhad_pool, dtype=np.float64)
        if fixed_exhad_events.ndim == 1:
            fixed_exhad_events = fixed_exhad_events.reshape(1, -1)
        available = (len(fixed_exhad_events)
                     if fixed_exhad_events.ndim >= 1 else 0)
        if fixed_exhad_events.ndim != 2 or available < fixed_exhad_required:
            raise AssertionError(
                "fixed exhad pool is insufficient for this scan point: "
                f"need {fixed_exhad_required}, got {available}")
        final_state_time += time.time() - pool_start_time

    exclusive_exhad_required = {}
    if exclusive_exhad_rows:
        for position, label in sorted(exclusive_exhad_rows.items()):
            count = int(size_per_channel[position])
            if count <= 0:
                continue
            exclusive_exhad_required[label] = (
                exclusive_exhad_required.get(label, 0) + count)
    if exclusive_exhad_required:
        pool_start_time = time.time()
        first_position = min(
            position for position in exclusive_exhad_rows
            if int(size_per_channel[position]) > 0)
        exclusive_seed = derive_exhad_seed(
            seed, batch_index, selected_decay_indices[first_position])
        exclusive_exhad_events = exhad.generate_exclusive_rows(
            mass, exclusive_exhad_required, seed=exclusive_seed)
        exclusive_exhad_cursors = {
            label: 0 for label in exclusive_exhad_required}
        final_state_time += time.time() - pool_start_time

    stock_pythia_rows = set()
    stock_pythia_events = None
    stock_pythia_cursor = 0
    stock_pythia_required = 0
    if stock_pythia_pool:
        if fixed_exhad_required:
            raise ValueError(
                "stock-Pythia pooling and matched pooling are mutually "
                "exclusive"
            )
        pool_start_time = time.time()
        stock_pythia_events, stock_plan = (
            _generate_alp_stock_pythia_pool(
                mass,
                PDGdecay,
                BrRatio,
                selected_decay_indices,
                size_per_channel,
                llp_name,
                seed,
                batch_index,
            )
        )
        stock_pythia_rows = set(stock_plan["pooled_positions"])
        stock_pythia_required = int(stock_plan["pool_size"])
        pythia_event_count += stock_pythia_required
        final_state_time += time.time() - pool_start_time

    # Loop through selected decay channels
    for idx_in_selected, (decay_modes, channel_size) in enumerate(zip(selected_PDGdecay, size_per_channel)):
        if channel_size == 0:
            processed_channel_results.append(np.empty((0, 0), dtype=np.float64))
            continue  # Skip if no events are assigned to this decay channel

        pdg_list = _active_pdg_ids(decay_modes)
        i = selected_decay_indices[idx_in_selected]

        if idx_in_selected in exclusive_exhad_rows:
            # Below the matched support exhad realizes this row exactly, at the
            # count EventCalc asked for, so there is no phase-space sample to
            # draw and then throw away.
            final_state_start_time = time.time()
            label = exclusive_exhad_rows[idx_in_selected]
            start = exclusive_exhad_cursors[label]
            stop = start + int(channel_size)
            produced = exclusive_exhad_events[label][start:stop]
            if len(produced) != int(channel_size):
                raise AssertionError(
                    "exhad exclusive row %r supplied %d of the %d events "
                    "this channel asked for"
                    % (label, len(produced), int(channel_size)))
            exclusive_exhad_cursors[label] = stop
            processed_channel_results.append(pad_processed_events(
                [[value for particle in event for value in particle]
                 for event in produced]))
            exhad_event_count += int(channel_size)
            total_generated_events += int(channel_size)
            final_state_time += time.time() - final_state_start_time
            continue

        channel_seed = derive_exhad_seed(seed, batch_index, i)
        seed_decay_generators(channel_seed)

        channel_start_time = time.time()
        if len(pdg_list) == 2:
            # Two-body decay
            pdg1, pdg2 = pdg_list
            masses, charges, stabilities = get_particle_properties([pdg1, pdg2])
            decay_results = TwoBodyDecay.decay_products(
                mass, channel_size, masses[0], masses[1], pdg1, pdg2,
                charges[0], charges[1], stabilities[0], stabilities[1]
            )
        elif len(pdg_list) == 3:
            # Three-body decay
            pdg1, pdg2, pdg3 = pdg_list
            masses, charges, stabilities = get_particle_properties([pdg1, pdg2, pdg3])
            specific_decay_params = (
                pdg1, pdg2, pdg3, masses[0], masses[1], masses[2],
                charges[0], charges[1], charges[2],
                stabilities[0], stabilities[1], stabilities[2],
                selected_Msquared3BodyLLP[idx_in_selected]
            )
            decay_results = ThreeBodyDecay.decay_products(
                mass, channel_size, specific_decay_params
            )
        elif len(pdg_list) == 4:
            # Four-body decay
            pdg1, pdg2, pdg3, pdg4 = pdg_list
            masses, charges, stabilities = get_particle_properties([pdg1, pdg2, pdg3, pdg4])
            specific_decay_params = (
                pdg1, pdg2, pdg3, pdg4, masses[0], masses[1], masses[2], masses[3],
                charges[0], charges[1], charges[2], charges[3],
                stabilities[0], stabilities[1], stabilities[2], stabilities[3]
            )
            decay_results = FourBodyDecay.decay_products(
                mass, specific_decay_params, channel_size
            )
        elif len(pdg_list) > 4:
            # Explicit higher-multiplicity rows are physical decay-table
            # states, independent of the matched/default hadronization
            # choice.  Generate their phase space natively so the default
            # path never depends on an exhad installation.
            masses, charges, stabilities = get_particle_properties(pdg_list)
            decay_results = NBodyDecay.decay_products(
                mass,
                pdg_list,
                masses,
                charges,
                stabilities,
                channel_size,
                rng=np.random.default_rng(channel_seed),
            )
        else:
            print(f"Invalid number of decay products ({len(pdg_list)}) in decay channel {i}. Skipping.")
            processed_channel_results.append(np.empty((0, 0), dtype=np.float64))
            continue  # Skip invalid decay channels

        decay_sim_time += time.time() - channel_start_time
        total_generated_events += len(decay_results)

        if primary_only:
            # The raw rest-frame records themselves are what was asked for.
            primary_events.extend(decay_results)
            continue

        final_state_start_time = time.time()
        handled = False

        # A fixed matched pool is a complete conditional event measure.  It
        # may own stable explicit rows (for example scalar pi-pi or ALP
        # nucleon pairs) as well as partonic rows, so consume it before asking
        # whether the original row itself would have needed Pythia.
        if idx_in_selected in stock_pythia_rows:
            stop = stock_pythia_cursor + len(decay_results)
            processed_channel_results.append(
                stock_pythia_events[stock_pythia_cursor:stop]
            )
            stock_pythia_cursor = stop
            handled = True

        elif idx_in_selected in fixed_exhad_rows:
            stop = fixed_exhad_cursor + len(decay_results)
            processed_channel_results.append(
                fixed_exhad_events[fixed_exhad_cursor:stop])
            fixed_exhad_cursor = stop
            exhad_event_count += len(decay_results)
            handled = True

        elif channel_requires_pythia(pdg_list):
            if exhad is not None and exhad.is_hnl_bench():
                # HNL: classify each signed current independently, retain the
                # sampled lepton/neutrino and exact W, and replace only the
                # pole-subtracted Jets residual. The six installed currents are
                # CC ud/us/cd/cs and NC ud/s. Explicit pole/exclusive rows never
                # enter this branch.
                spec = exhad.hnl_channel_spec(pdg_list)
                if spec is not None:
                    exhad.require_hnl_parent_masses_supported([mass])
                    if not spec["supported"]:
                        raise RuntimeError(
                            "HNL %s is inactive through the installed "
                            "production endpoint and has no event backend."
                            % spec["current"])
                    # The release validates the exact sampled W and the
                    # lepton-dependent endpoint of every event.
                    processed_channel_results.append(
                        exhad.process_hnl_with_exhad(
                            decay_results, pdg_list, mass, seed=channel_seed))
                    exhad_event_count += len(decay_results)
                    handled = True
            if not handled:
                pythia_event_count += len(decay_results)
                processed_channel_results.append(
                    process_events_with_pythia(
                        decay_results, mass, seed=channel_seed))
        else:
            stable_event_count += len(decay_results)
            processed_channel_results.append(convert_events_without_pythia(decay_results))
        final_state_time += time.time() - final_state_start_time

        # Every route returns one final state per sampled decay.  A route that
        # returned a different number would leave this channel's block out of
        # step with size_per_channel, and every later row would be attributed
        # to the wrong channel in silence.
        produced = len(processed_channel_results[-1])
        if produced != len(decay_results):
            raise RuntimeError(
                "final-state routing of channel %d asked for %d event(s) and "
                "returned %d" % (i, len(decay_results), produced))

    if primary_only:
        return primary_events, size_per_channel

    if fixed_exhad_cursor != fixed_exhad_required:
        raise AssertionError(
            "fixed exhad pool split did not consume the expected prefix: "
            f"expected {fixed_exhad_required}, consumed {fixed_exhad_cursor}")
    for label, required in exclusive_exhad_required.items():
        if exclusive_exhad_cursors.get(label, 0) != required:
            raise AssertionError(
                "exclusive exhad row %r was asked for %d event(s) and %d "
                "were consumed"
                % (label, required, exclusive_exhad_cursors.get(label, 0)))
    if stock_pythia_cursor != stock_pythia_required:
        raise AssertionError(
            "stock-Pythia pool split did not consume the expected prefix: "
            f"expected {stock_pythia_required}, "
            f"consumed {stock_pythia_cursor}"
        )

    processed_results = concatenate_padded_event_arrays(processed_channel_results)

    print(f"\nTotal decay events generated: {total_generated_events}")
    print(f"Stable events converted without Pythia: {stable_event_count}")
    print(f"Events processed with Pythia: {pythia_event_count}")
    print(f"Events processed with matched exhad: {exhad_event_count}")
    print(f"Time taken for decay simulations: {decay_sim_time:.2f} seconds")
    print(f"Time taken for final-state processing: {final_state_time:.2f} seconds")

    # Total time
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    print(f"Total time for simulateDecays_rest_frame: {total_time:.2f} seconds")

    if return_process_labels:
        # One label per pooled row, so that two rows of one pool never share
        # an output block.  A row exhad realized as its own exclusive row
        # keeps its physical channel label and is None here.
        process_labels = []
        for position in range(len(selected_decay_indices)):
            if position in stock_pythia_rows:
                process_labels.append(ALP_STOCK_PYTHIA_PROCESS_LABEL)
            elif position in fixed_exhad_rows:
                process_labels.append("%s:%s" % (
                    MATCHED_PROCESS_LABEL,
                    _pooled_row_name(exhad, PDGdecay,
                                     selected_decay_indices[position],
                                     decay_channels)))
            else:
                process_labels.append(None)
        return (processed_results, size_per_channel, process_labels)
    return (processed_results, size_per_channel)


def _process_pythia_chunk(args):
    chunk, mass, seed = args
    return _process_events_with_pythia_sequential(chunk, mass, seed=seed)


def process_events_with_pythia(decay_events_list, mass, n_workers=None,
                               chunk_size=None, seed=None,
                               p4_relative_tolerance=None, p4_audit=None):
    """
    Processes decay events through Pythia. The block is cut into chunks of a
    fixed size, each running on the stream its index names. Set
    PYTHIA8_N_WORKERS to an integer or "auto" to process those chunks in
    parallel; one worker keeps the sequential path, which reproduces from the
    seed alone.

    The optional rest-frame closure guard is a sequential raw-sample audit.
    It retries the same input when Pythia accepts a numerically inconsistent
    final state. It reads one uninterrupted stream, which is the same set of
    events as the default path for a block that fits in one chunk. The
    ndarray return is unchanged.
    """
    decay_events = _as_2d_decay_array(decay_events_list)
    if decay_events.size == 0:
        return np.empty((0, 0), dtype=np.float64)

    if chunk_size is None:
        chunk_size = _positive_int_from_env("PYTHIA8_CHUNK_SIZE", DEFAULT_PYTHIA8_CHUNK_SIZE)
    if n_workers is None:
        n_workers = _pythia_worker_count(len(decay_events), chunk_size)

    if p4_relative_tolerance is not None or p4_audit is not None:
        if n_workers != 1:
            raise ValueError("the optional Pythia p4 guard requires n_workers=1")
        return _process_events_with_pythia_sequential(
            decay_events, mass, seed=seed,
            p4_relative_tolerance=p4_relative_tolerance, p4_audit=p4_audit)

    if seed is None:
        # Without a seed there is no stream to cut into chunk streams, so the
        # block stays one chunk on Pythia's default stream, which is what it
        # ran on when every block built its own object.
        print(f"\nStarting Pythia processing of {len(decay_events)} events...")
        return _process_events_with_pythia_sequential(
            decay_events, mass, seed=None)

    chunk_arguments = [
        (decay_events[start:start + chunk_size], mass,
         _pythia_chunk_seed(seed, index))
        for index, start in enumerate(range(0, len(decay_events), chunk_size))
    ]
    if n_workers > 1 and len(chunk_arguments) > 1:
        context = _multiprocessing_context()
        if context is None:
            print("Pythia multiprocessing requires the fork start method. "
                  "Falling back to sequential processing.")
        else:
            n_workers = min(n_workers, len(chunk_arguments))
            print(f"\nStarting Pythia processing of {len(decay_events)} events "
                  f"with {n_workers} workers and chunk size {chunk_size}...")
            # Build the object here so that every worker inherits it.
            _pythia_for_block(chunk_arguments[0][2])
            with context.Pool(processes=n_workers) as pool:
                return concatenate_padded_event_arrays(
                    pool.map(_process_pythia_chunk, chunk_arguments))
    else:
        print(f"\nStarting Pythia processing of {len(decay_events)} events...")
    return concatenate_padded_event_arrays(
        [_process_pythia_chunk(arguments) for arguments in chunk_arguments])


def _rebuild_manual_pythia_event(pythia, decay, mass, mother_id=25):
    """Restore one external decay record before a hadronization attempt."""
    num_particles = len(decay) // 8
    event_record = pythia.event
    event_record.reset()
    # ``forceHadronLevel`` validates the complete manual event against record
    # 0.  ``Event.reset()`` leaves that system record at zero four-momentum
    # and zero mass, so it must be completed explicitly.  It is bookkeeping
    # only and does not alter the fragmentation state.  Fetch it once: each
    # ``event[0]`` is a crossing of the Python/C++ boundary in its own right.
    system_record = event_record[0]
    system_record.p(0.0, 0.0, 0.0, mass)
    system_record.m(mass)
    system_record.status(-11)

    # Append the mother particle at rest with status -23.
    event_record.append(
        mother_id, -23, 0, 0, 2, num_particles + 1,
        0, 0, 0.0, 0.0, 0.0, mass, mass
    )

    append = event_record.append
    gluon_index = 0
    for i_particle in range(num_particles):
        idx = i_particle * 8
        pdgId = int(decay[idx + 5])

        if pdgId == -999:
            continue  # Skip placeholder entries

        # Assign the same color record on every retry.  Only Pythia's random
        # fragmentation draw is allowed to change.
        col = 0
        acol = 0
        if pdgId in [1, 2, 3, 4, 5]:
            col = 501
            status_code = 23
        elif pdgId in [-1, -2, -3, -4, -5]:
            acol = 501
            status_code = 23
        elif pdgId == 21:
            # Gluons carry a closed color loop. Count only gluons; padding or
            # non-gluon products must not alter its parity.
            if gluon_index % 2 == 0:
                col, acol = 501, 502
            else:
                col, acol = 502, 501
            gluon_index += 1
            status_code = 23
        else:
            status_code = 1

        append(
            pdgId, status_code, 1, 1, 0, 0, col, acol,
            decay[idx], decay[idx + 1], decay[idx + 2],
            decay[idx + 3], decay[idx + 4]
        )


def _process_events_with_pythia_sequential(decay_events_list, mass,
                                            seed=None,
                                            p4_relative_tolerance=None,
                                            p4_audit=None):
    """
    Processes a list of decay events through Pythia sequentially, pads each event to have the same number of particles,
    and writes the processed events to external files.
    """
    decay_events_list = _as_2d_decay_array(decay_events_list)
    guard_enabled = p4_relative_tolerance is not None
    if p4_audit is not None and (not guard_enabled or not isinstance(p4_audit, dict)):
        raise ValueError("p4_audit requires an enabled guard and a dictionary")
    if guard_enabled:
        p4_relative_tolerance = float(p4_relative_tolerance)
        if (not math.isfinite(p4_relative_tolerance) or p4_relative_tolerance <= 0
                or not math.isfinite(mass) or mass <= 0):
            raise ValueError("Pythia p4 tolerance and parent mass must be finite and positive")
        audit = p4_audit if p4_audit is not None else {}
        audit.update({
            "schema": "eventcalc-raw-p4-guard-v1",
            "relative_tolerance": p4_relative_tolerance,
            "absolute_tolerance_gev": p4_relative_tolerance * mass,
            "input_events": len(decay_events_list), "returned_events": 0,
            "total_attempts": 0, "force_hadron_level_failures": 0,
            "closure_rejected_attempts": 0, "recovered_input_events": 0,
            "maximum_attempts_used": 0,
            "maximum_accepted_residual_gev": 0.0,
            "maximum_finite_rejected_residual_gev": 0.0,
            "nonfinite_or_empty_rejections": 0,
            "rejected_attempts": [],
        })
    pythia = _pythia_for_block(seed)
    event_record = pythia.event
    force_hadron_level = pythia.forceHadronLevel
    mother_id = 25  # PDG ID for the mother particle (e.g., Higgs boson)

    processed_events = []
    hadronization_failures = 0
    recovered_hadronization_events = 0
    maximum_attempts_used = 1
    event_counter = 0  # Initialize the event counter

    # Read the block's numbers out of NumPy once, rather than creating a
    # NumPy scalar for each of them at each of up to ten attempts.
    for decay in decay_events_list.tolist():
        event_counter += 1
        # Rare numerical failures can occur inside Pythia's stochastic string
        # fragmentation.  Pythia documents repeated hadronization by restoring
        # the saved parton-level event and trying ``forceHadronLevel`` again.
        # Rebuild the exact same input while allowing only its RNG to advance;
        # never inspect or consume the inconsistent failed record.
        attempts_used = 0
        while attempts_used < DEFAULT_PYTHIA8_HADRONIZATION_ATTEMPTS:
            attempts_used += 1
            if guard_enabled:
                audit["total_attempts"] += 1
                audit["maximum_attempts_used"] = max(
                    audit["maximum_attempts_used"], attempts_used)
            _rebuild_manual_pythia_event(
                pythia, decay, mass, mother_id=mother_id)
            if force_hadron_level():
                if not guard_enabled:
                    break
                # This is the same infinity-norm rest-frame check as the raw
                # campaign worker. Pythia's own default check is less strict.
                # No particle is removed, projected or boosted to repair it.
                final_state_particles = [
                    value for particle_index in range(pythia.event.size())
                    for p in (pythia.event[particle_index],) if p.isFinal()
                    for value in (p.px(), p.py(), p.pz(), p.e(), p.m(), p.id())
                ]
                final_rows = np.asarray(final_state_particles, dtype=np.float64).reshape(-1, 6)
                finite = bool(len(final_rows) and np.all(np.isfinite(final_rows)))
                residual = (float(np.max(np.abs(
                    final_rows[:, :4].sum(axis=0) - [0., 0., 0., mass])))
                    if finite else None)
                if residual is not None and not math.isfinite(residual):
                    finite, residual = False, None
                if finite and residual <= audit["absolute_tolerance_gev"]:
                    audit["maximum_accepted_residual_gev"] = max(
                        audit["maximum_accepted_residual_gev"], residual)
                    break
                audit["closure_rejected_attempts"] += 1
                if finite:
                    audit["maximum_finite_rejected_residual_gev"] = max(
                        audit["maximum_finite_rejected_residual_gev"], residual)
                else:
                    audit["nonfinite_or_empty_rejections"] += 1
                audit["rejected_attempts"].append({
                    "input_event_index": event_counter - 1,
                    "attempt": attempts_used, "max_abs_residual_gev": residual,
                    "final_pdgs": [int(row[5]) for row in final_rows
                                   if math.isfinite(row[5])],
                    "nonfinite_or_empty": not finite,
                })
                continue
            hadronization_failures += 1
            if guard_enabled:
                audit["force_hadron_level_failures"] += 1
        else:
            active = np.asarray(decay, dtype=np.float64).reshape(-1, 8)
            active = active[active[:, 5] != -999]
            summed = active[:, :4].sum(axis=0)
            delta = summed - np.asarray([0.0, 0.0, 0.0, mass])
            shell_residual = (
                active[:, 3] ** 2
                - np.sum(active[:, :3] ** 2, axis=1)
                - active[:, 4] ** 2
            )
            max_shell_residual = (
                float(np.max(np.abs(shell_residual)))
                if len(shell_residual) else 0.0)
            pdgs = [int(value) for value in active[:, 5]]
            if guard_enabled and any(
                    row["input_event_index"] == event_counter - 1
                    for row in audit["rejected_attempts"]):
                raise RuntimeError(
                    "Pythia raw p4 guard failed for event %d/%d after %d "
                    "attempts (seed=%s, relative_tolerance=%.3e, "
                    "input_pdgs=%s); no output event was dropped or repaired"
                    % (event_counter, len(decay_events_list), attempts_used,
                       seed, p4_relative_tolerance, pdgs))
            raise RuntimeError(
                "Pythia forceHadronLevel failed for event %d/%d after %d "
                "attempts; every inconsistent record was rejected "
                "(seed=%s, input_pdgs=%s, input_delta_p4=%s, "
                "max_input_shell_residual=%.3e, total_failed_attempts=%d)" %
                (event_counter, len(decay_events_list), attempts_used,
                 "none" if seed is None else str(seed), pdgs,
                 np.array2string(delta, precision=3),
                 max_shell_residual, hadronization_failures))

        if attempts_used > 1:
            recovered_hadronization_events += 1
            maximum_attempts_used = max(maximum_attempts_used, attempts_used)
            if guard_enabled:
                audit["recovered_input_events"] += 1

        # Extract final state particles
        if not guard_enabled:
            final_state_particles = []
            extend = final_state_particles.extend
            for particle_index in range(event_record.size()):
                p = event_record[particle_index]
                if p.isFinal():
                    extend((p.px(), p.py(), p.pz(), p.e(), p.m(), p.id()))

        processed_events.append(final_state_particles)
        if guard_enabled:
            audit["returned_events"] += 1

    if guard_enabled:
        print("Raw Pythia p4 guard: %d returned events; %d closure-rejected "
              "attempt(s), %d forceHadronLevel failure(s), %d recovered "
              "input event(s); maximum attempts: %d."
              % (audit["returned_events"], audit["closure_rejected_attempts"],
                 audit["force_hadron_level_failures"], audit["recovered_input_events"],
                 audit["maximum_attempts_used"]))
    elif recovered_hadronization_events:
        print(
            "Recovered %d Pythia event(s) after %d rejected "
            "hadronization attempt(s); maximum attempts for one event: %d." %
            (recovered_hadronization_events, hadronization_failures,
             maximum_attempts_used))

    if not processed_events:
        print("\nNo processed events were generated.")

    return pad_processed_events(processed_events)
