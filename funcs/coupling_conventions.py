"""Validated coupling conventions for tabulated EventCalc models.

The fermion-coupled ALP tables use the native SensCalc variable
``g_Y == y == 2*v_h/f_a``.  Keeping that statement in machine-readable
metadata prevents a silent reinterpretation as ``v_h/f_a``, which would
shift coupling-squared values by a factor of four.
"""

import json
import math
import os
import tempfile


ALP_FERMION = "ALP-fermion"
METADATA_FILENAME = "coupling.json"
_EXPECTED_DEFINITION = "2*v_h/f_a"
_EXPECTED_CTAU_COLUMN = "c_tau_times_g_Y_squared_m"
_EXPECTED_YIELD_COLUMN = "production_probability_per_POT_per_g_Y_squared"


def _default_metadata_path():
    eventcalc_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return os.path.join(
        eventcalc_root, "Distributions", ALP_FERMION, METADATA_FILENAME)


def validate_coupling_metadata(metadata, expected_llp_name=ALP_FERMION):
    """Validate and return canonical fermion-ALP coupling metadata.

    Validation is deliberately strict for normalization-bearing fields.  A
    differently normalized table must not be accepted merely because it uses
    a similar coupling symbol.
    """
    if not isinstance(metadata, dict):
        raise ValueError("coupling metadata must be a JSON object")
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported coupling metadata schema_version")
    if metadata.get("llp_name") != expected_llp_name:
        raise ValueError(
            "coupling metadata LLP mismatch: expected %r" % expected_llp_name)

    coupling = metadata.get("coupling")
    if not isinstance(coupling, dict):
        raise ValueError("coupling metadata lacks a coupling object")
    required_coupling = {
        "symbol": "g_Y",
        "alias": "y",
        "definition": _EXPECTED_DEFINITION,
        "squared_output_field": "coupling_squared",
        "squared_output_quantity": "g_Y^2",
    }
    for key, expected in required_coupling.items():
        if coupling.get(key) != expected:
            raise ValueError(
                "invalid fermion-ALP coupling %s: expected %r" %
                (key, expected))
    v_h = coupling.get("v_h_GeV")
    if not isinstance(v_h, (int, float)) or not math.isfinite(float(v_h)):
        raise ValueError("fermion-ALP v_h_GeV must be finite")
    if float(v_h) <= 0.0:
        raise ValueError("fermion-ALP v_h_GeV must be positive")

    benchmark = metadata.get("benchmark")
    if not isinstance(benchmark, dict):
        raise ValueError("coupling metadata lacks BC10 benchmark metadata")
    required_benchmark = {
        "name": "BC10",
        "eft_rg_matching_scale_symbol": "Lambda",
        "eft_rg_matching_scale_GeV": 1000.0,
        "matching_scale_is_f_a": False,
    }
    for key, expected in required_benchmark.items():
        if benchmark.get(key) != expected:
            raise ValueError(
                "invalid fermion-ALP benchmark %s: expected %r" %
                (key, expected))

    provenance = metadata.get("provenance")
    required_provenance = {
        "canonical_definition": (
            "The model chapter defines y=2*v_h/f_a; formulas use y/(2*v_h)."),
        "non_normative_label": (
            "The later proper-lifetime plot label y=v_h/f_a is stale and "
            "non-normative."),
    }
    if not isinstance(provenance, dict):
        raise ValueError("coupling metadata lacks convention provenance")
    for key, expected in required_provenance.items():
        if provenance.get(key) != expected:
            raise ValueError(
                "invalid fermion-ALP provenance %s" % key)

    expected_width = "Gamma_i(m,g_Y)=(g_Y/(2*v_h))^2*W_i(m)"
    if metadata.get("width_scaling") != expected_width:
        raise ValueError("invalid fermion-ALP width scaling")

    tables = metadata.get("tables")
    if not isinstance(tables, dict):
        raise ValueError("coupling metadata lacks table definitions")
    ctau = tables.get("ctau-ALP-fermion.txt")
    total_yield = tables.get("Total-yield-ALP-fermion.txt")
    branching = tables.get("ALP-fermion-decay.json")
    if not isinstance(ctau, dict) or ctau.get("columns") != [
            "mass_GeV", _EXPECTED_CTAU_COLUMN]:
        raise ValueError("invalid fermion-ALP ctau table normalization")
    if ctau.get("relation") != (
            "c_tau(m,g_Y)=c_tau_times_g_Y_squared_m/g_Y^2"):
        raise ValueError("invalid fermion-ALP ctau scaling relation")
    if not isinstance(total_yield, dict) or total_yield.get("columns") != [
            "mass_GeV", _EXPECTED_YIELD_COLUMN]:
        raise ValueError("invalid fermion-ALP yield table normalization")
    if total_yield.get("relation") != (
            "P_prod(m,g_Y)=production_probability_per_POT_per_g_Y_squared*"
            "g_Y^2"):
        raise ValueError("invalid fermion-ALP production scaling relation")
    if (not isinstance(branching, dict) or
            branching.get("quantity") != "branching_ratios" or
            branching.get("coupling_dependence") != "none"):
        raise ValueError("invalid fermion-ALP branching-ratio metadata")
    return metadata


def load_coupling_metadata(particle_path=None):
    """Load the canonical fermion-ALP convention from ``coupling.json``."""
    path = (os.path.join(os.fspath(particle_path), METADATA_FILENAME)
            if particle_path is not None else _default_metadata_path())
    try:
        with open(path, encoding="utf-8") as stream:
            metadata = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("cannot load coupling metadata %s: %s" %
                           (path, exc)) from exc
    return validate_coupling_metadata(metadata)


def _resolved_metadata(llp_name, metadata):
    if llp_name != ALP_FERMION:
        return None
    if metadata is None:
        return load_coupling_metadata()
    return validate_coupling_metadata(metadata, expected_llp_name=llp_name)


def native_coupling_squared(llp_name, ctau_coefficient_m, requested_ctau_m,
                            metadata=None):
    """Return the squared coupling native to a lifetime table.

    For ``ALP-fermion`` this is exactly ``g_Y^2`` with
    ``g_Y = y = 2*v_h/f_a``.  There is intentionally no factor-of-four
    conversion: ``ctau_coefficient_m`` already stores ``c_tau*g_Y^2``.
    """
    if llp_name == ALP_FERMION:
        _resolved_metadata(llp_name, metadata)
    coefficient = float(ctau_coefficient_m)
    requested = float(requested_ctau_m)
    if not math.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError("ctau coupling coefficient must be finite and positive")
    if not math.isfinite(requested) or requested <= 0.0:
        raise ValueError("requested c_tau must be finite and positive")
    return coefficient / requested


def event_header_annotation(llp_name, metadata=None):
    """Return a parser-safe coupling annotation for an event-file header."""
    if llp_name != ALP_FERMION:
        return ""
    _resolved_metadata(llp_name, metadata)
    # A semicolon keeps the annotation outside the legacy reader's permissive
    # floating-point character class, which also accepts a trailing period.
    return ("; Coupling convention: coupling_squared = g_Y^2; "
            "g_Y = y = 2 v_h/f_a.")


def write_output_sidecar(output_path, llp_name, metadata=None):
    """Write canonical coupling metadata next to an ALP output file.

    The sidecar name is ``<output>.coupling.json``.  Non-ALP models return
    ``None`` so existing output paths and formats remain unchanged.
    """
    if llp_name != ALP_FERMION:
        return None
    resolved = _resolved_metadata(llp_name, metadata)
    output_path = os.fspath(output_path)
    sidecar_path = output_path + ".coupling.json"
    payload = {
        "schema_version": 1,
        "applies_to": os.path.basename(output_path),
        "coupling_metadata": resolved,
    }
    serialized = json.dumps(
        payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"

    directory = os.path.dirname(os.path.abspath(sidecar_path))
    os.makedirs(directory, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=directory, delete=False) as stream:
            temporary_path = stream.name
            stream.write(serialized)
        os.replace(temporary_path, sidecar_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass
    return sidecar_path
