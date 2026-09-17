#!/usr/bin/env python3
"""Fail-closed validation for the canonical HNL decay source pair.

The ordinary HNL loader historically trusted two independently generated
files.  That is unsafe for the current-resolved tables: a stale width table,
an omitted current, or a slightly wrong exclusive threshold can all produce
plausible-looking branching ratios.  This module validates the complete pair
before returning it to a caller.  It intentionally depends only on the Python
standard library so the check can run before EventCalc's numerical stack is
imported.

``load_canonical_hnl_source`` is the entry point for the installed tables.
``load_validated_hnl_source`` supports explicit paths and is useful to source
builders and tests.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


EXPECTED_CHANNEL_COUNT = 88
EXPECTED_MASS_NODES = 1724
MIXING_NAMES = ("e", "mu", "tau")

# This is the exact current-resolved inventory produced by the matched source
# notebook.  Replacing or aggregating one of these rows is a physics change and
# must therefore update this contract deliberately.
EXPECTED_CHANNEL_NAMES = frozenset(
    """
    2ev 2muv 2Pie 2Piebar 2Pimu 2Pimubar 2Pitau 2Pitaubar 2Piv
    2tauv a1v De Debar Dmu Dmubar Dse Dsebar Dsmu Dsmubar Dstau
    Dstaubar Dtau Dtaubar emuv emuvbar Etacv EtaPrv etauv etauvbar
    Etav Jets-bbv Jets-cbe Jets-cbebar Jets-cbmu Jets-cbmubar
    Jets-cbtau Jets-cbtaubar Jets-ccv Jets-cde Jets-cdebar Jets-cdmu
    Jets-cdmubar Jets-cdtau Jets-cdtaubar Jets-cse Jets-csebar
    Jets-csmu Jets-csmubar Jets-cstau Jets-cstaubar Jets-ddv Jets-ssv
    Jets-ube Jets-ubebar Jets-ubmu Jets-ubmubar Jets-ubtau
    Jets-ubtaubar Jets-ude Jets-udebar Jets-udmu Jets-udmubar
    Jets-udtau Jets-udtaubar Jets-use Jets-usebar Jets-usmu
    Jets-usmubar Jets-ustau Jets-ustaubar Jets-uuv Ke Kebar Kmu
    Kmubar Ktau Ktaubar mutauv mutauvbar Omegav Phiv Pi0v Pie
    Piebar Pimu Pimubar Pitau Pitaubar
    """.split()
)

# Pole masses used only to enforce the physical lower support of explicit
# EventCalc rows.  Values are current PDG central masses in GeV; the broad
# a1(1260) entry follows EventCalc's nominal 1.23 GeV mass.  Quark rows are
# validated as matched continua and are not subjected to a sum-of-parton-mass
# threshold here.
PDG_MASS_GEV = {
    11: 0.00051099895,
    12: 0.0,
    13: 0.1056583755,
    14: 0.0,
    15: 1.77686,
    16: 0.0,
    111: 0.1349768,
    211: 0.13957039,
    221: 0.547862,
    223: 0.78266,
    321: 0.493677,
    331: 0.95778,
    333: 1.019461,
    411: 1.86966,
    431: 1.96835,
    441: 2.9839,
    20113: 1.23,
}

# Neutral mesons and gauge bosons in the decay-card convention are their own
# antiparticles.  Neutrino PDGs are deliberately absent: conjugate leptonic
# rows must flip their neutrino sign as well.
SELF_CONJUGATE_PDGS = frozenset(
    (21, 22, 111, 113, 221, 223, 331, 333, 441, 20113))

NEUTRAL_PERSISTENT_POLES = (
    "Pi0v", "a1v", "Omegav", "Etav", "EtaPrv", "Phiv", "Etacv")

# exHad's corrected HNL matrix elements take the lepton masses from the
# generator rather than freezing them as decimals, so nine rows (2ev, 2muv,
# 2tauv, emuv(bar), etauv(bar), mutauv(bar)) carry the daughter-mass symbols
# m1/m2/m3 alongside mLLP, E1 and E3.  funcs/initLLP.py and funcs/HNLmerging.py
# bind exactly this set and reject anything else.
_ALLOWED_ME_NAMES = frozenset(("mLLP", "E1", "E3", "m1", "m2", "m3"))
_ALLOWED_ME_AST = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.UAdd,
    ast.USub,
    ast.Constant,
    ast.Name,
    ast.Load,
)

# The notebook's invisible Majorana width for one pure mixing.  It is omitted
# from the event card but retained in HNLdecayWidth.dat.
FERMI_CONSTANT_GEV = 1.166379e-5
VISIBLE_CLOSURE_REL_TOL = 1.0e-8
VISIBLE_CLOSURE_ABS_TOL = 1.0e-12


class HNLSourceValidationError(RuntimeError):
    """Raised when an HNL source pair violates the installed contract."""


@dataclass(frozen=True)
class ValidatedHNLSource:
    """Decoded HNL tables together with their verified provenance."""

    decay_rows: list[list[Any]]
    width_rows: list[tuple[float, float, float, float]]
    mass_grid: tuple[float, ...]
    decay_json_sha256: str
    width_table_sha256: str
    metadata_path: Path | None


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for *path*."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fail(message: str) -> None:
    raise HNLSourceValidationError(message)


def _json_no_nonfinite(token: str) -> None:
    _fail("non-finite JSON token %s is forbidden" % token)


def _read_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as stream:
            return json.load(stream, parse_constant=_json_no_nonfinite)
    except HNLSourceValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _fail("cannot decode %s: %s" % (path, exc))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _metadata_candidates(decay_json_path: Path) -> tuple[Path, ...]:
    directory = decay_json_path.parent
    return (
        directory / "HNL-source.json",
        directory / "HNL-source-metadata.json",
        directory / "hnl-source-metadata.json",
        directory / "source-metadata.json",
    )


def _discover_metadata(decay_json_path: Path) -> Path | None:
    existing = tuple(path for path in _metadata_candidates(decay_json_path)
                     if path.is_file())
    if len(existing) > 1:
        _fail("ambiguous HNL source metadata: %s" %
              ", ".join(str(path) for path in existing))
    return existing[0] if existing else None


def _expected_hashes(metadata: Any, metadata_path: Path) -> tuple[str, str]:
    if not isinstance(metadata, dict):
        _fail("%s must contain a JSON object" % metadata_path)
    source = metadata.get("source", metadata)
    if not isinstance(source, dict):
        _fail("%s source metadata must be an object" % metadata_path)

    expected_json = source.get("decay_json_sha256")
    expected_width = source.get("width_table_sha256")
    for label, digest in (("decay_json_sha256", expected_json),
                          ("width_table_sha256", expected_width)):
        if (not isinstance(digest, str) or len(digest) != 64 or
                any(char not in "0123456789abcdefABCDEF" for char in digest)):
            _fail("%s has no valid %s" % (metadata_path, label))
    return expected_json.lower(), expected_width.lower()


def _validate_metadata(
    metadata: Any,
    metadata_path: Path,
    decay_digest: str,
    width_digest: str,
    mass_grid: Sequence[float] | None = None,
) -> None:
    expected_json, expected_width = _expected_hashes(metadata, metadata_path)
    if decay_digest != expected_json:
        _fail("HNL decay JSON SHA-256 mismatch: expected %s, got %s" %
              (expected_json, decay_digest))
    if width_digest != expected_width:
        _fail("HNL width-table SHA-256 mismatch: expected %s, got %s" %
              (expected_width, width_digest))

    if mass_grid is None:
        return
    inventory = metadata.get("inventory")
    if inventory is None:
        return
    if not isinstance(inventory, dict):
        _fail("%s inventory must be an object" % metadata_path)
    if inventory.get("channels", EXPECTED_CHANNEL_COUNT) != \
            EXPECTED_CHANNEL_COUNT:
        _fail("metadata channel inventory does not equal %d" %
              EXPECTED_CHANNEL_COUNT)
    if inventory.get("mass_nodes", len(mass_grid)) != len(mass_grid):
        _fail("metadata mass-node inventory does not match the source grid")
    stated_range = inventory.get("mass_range_GeV")
    if stated_range is not None:
        if (not isinstance(stated_range, list) or len(stated_range) != 2 or
                not all(_is_number(value) for value in stated_range)):
            _fail("metadata mass_range_GeV must be a numeric two-vector")
        if (not math.isclose(float(stated_range[0]), mass_grid[0],
                             rel_tol=0.0, abs_tol=1.0e-15) or
                not math.isclose(float(stated_range[1]), mass_grid[-1],
                                 rel_tol=0.0, abs_tol=1.0e-12)):
            _fail("metadata mass range does not match the source grid")


def _matrix_element_is_zero(tree: ast.Expression) -> bool:
    if any(isinstance(node, ast.Name) for node in ast.walk(tree)):
        return False
    try:
        value = eval(compile(tree, "<HNL matrix element>", "eval"),
                     {"__builtins__": {}}, {})
    except (ArithmeticError, TypeError, ValueError):
        return False
    return _is_number(value) and float(value) == 0.0


def _validate_matrix_element(
    expression: Any,
    channel: str,
    mixing: str,
    has_positive_branching: bool,
) -> None:
    if not isinstance(expression, str) or not expression.strip():
        _fail("%s/%s has no matrix-element string" % (channel, mixing))
    normalized = expression.replace("***", "e")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as exc:
        _fail("%s/%s has invalid matrix-element syntax: %s" %
              (channel, mixing, exc.msg))
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_ME_AST):
            _fail("%s/%s uses disallowed matrix-element syntax %s" %
                  (channel, mixing, type(node).__name__))
        if isinstance(node, ast.Name) and node.id not in _ALLOWED_ME_NAMES:
            _fail("%s/%s uses unknown matrix-element identifier %s" %
                  (channel, mixing, node.id))
        if (isinstance(node, ast.Constant) and
                (not _is_number(node.value) or not math.isfinite(
                    float(node.value)))):
            _fail("%s/%s has a non-finite/non-numeric matrix-element literal" %
                  (channel, mixing))
    if has_positive_branching and _matrix_element_is_zero(tree):
        _fail("%s/%s has positive branching but a zero matrix element" %
              (channel, mixing))


def _active_threshold(channel: str, pdgs: Sequence[int]) -> float | None:
    if channel.startswith("Jets-"):
        return None
    threshold = 0.0
    for signed_pdg in pdgs:
        if signed_pdg == -999:
            continue
        pdg = abs(signed_pdg)
        if pdg not in PDG_MASS_GEV:
            _fail("%s has no physical threshold mass for PDG %d" %
                  (channel, signed_pdg))
        threshold += PDG_MASS_GEV[pdg]
    return threshold


def _validate_thresholds(rows_by_name: dict[str, list[Any]]) -> None:
    for channel, row in rows_by_name.items():
        threshold = _active_threshold(channel, row[1])
        if threshold is None:
            continue
        for mixing_index, mixing in enumerate(MIXING_NAMES):
            for mass, branching in row[2 + mixing_index]:
                if branching > 0.0 and mass < threshold - 1.0e-12:
                    _fail(
                        "%s/%s is positive below physical threshold: "
                        "m=%g GeV, threshold=%g GeV" %
                        (channel, mixing, mass, threshold))


def _conjugated_pdgs(pdgs: Sequence[int]) -> list[int]:
    return [
        -999 if pdg == -999 else
        pdg if abs(pdg) in SELF_CONJUGATE_PDGS else
        -pdg
        for pdg in pdgs
    ]


def _validate_conjugates(rows_by_name: dict[str, list[Any]]) -> None:
    conjugate_count = 0
    for name, row in rows_by_name.items():
        if not name.endswith("bar"):
            continue
        conjugate_count += 1
        base_name = name[:-3]
        base = rows_by_name.get(base_name)
        if base is None:
            _fail("%s has no charge-conjugate base row %s" %
                  (name, base_name))
        if row[1] != _conjugated_pdgs(base[1]):
            _fail("%s does not have charge-conjugate PDGs" % name)
        for mixing_index, mixing in enumerate(MIXING_NAMES):
            if row[2 + mixing_index] != base[2 + mixing_index]:
                _fail("%s/%s branching table differs from %s" %
                      (name, mixing, base_name))
            if row[5 + mixing_index] != base[5 + mixing_index]:
                _fail("%s/%s matrix element differs from %s" %
                      (name, mixing, base_name))
    if conjugate_count != 36:
        _fail("expected 36 conjugate pairs, found %d" % conjugate_count)


def _charged_row_owners() -> dict[str, int]:
    owners: dict[str, int] = {}
    suffixes = (("e", 0), ("mu", 1), ("tau", 2))
    for suffix, mixing_index in suffixes:
        for prefix in ("Pi", "K", "D", "Ds", "2Pi"):
            name = prefix + suffix
            owners[name] = mixing_index
            owners[name + "bar"] = mixing_index
        for current in ("ud", "us", "cd", "cs", "ub", "cb"):
            name = "Jets-" + current + suffix
            owners[name] = mixing_index
            owners[name + "bar"] = mixing_index
    return owners


CHARGED_ROW_OWNERS = _charged_row_owners()


def _validate_current_and_mixing_ownership(
    rows_by_name: dict[str, list[Any]],
) -> None:
    for channel, owner in CHARGED_ROW_OWNERS.items():
        row = rows_by_name[channel]
        if not any(branching > 0.0 for _, branching in row[2 + owner]):
            _fail("%s has no support in its owning %s mixing" %
                  (channel, MIXING_NAMES[owner]))
        for mixing_index, mixing in enumerate(MIXING_NAMES):
            if mixing_index == owner:
                continue
            if any(branching != 0.0
                   for _, branching in row[2 + mixing_index]):
                _fail("%s leaks into non-owning %s mixing" %
                      (channel, mixing))

    for channel in ("Jets-uuv", "Jets-ddv", "Jets-ssv", "Jets-ccv",
                    "Jets-bbv"):
        row = rows_by_name[channel]
        for mixing_index, mixing in enumerate(MIXING_NAMES):
            if not any(branching > 0.0
                       for _, branching in row[2 + mixing_index]):
                _fail("%s has no support for pure %s mixing" %
                      (channel, mixing))

    # The explicit two-pion row owns the low side of the light-current match;
    # the ud jet residual owns the high side.  They must never overlap.
    for mixing_index, suffix in enumerate(("e", "mu", "tau")):
        low = rows_by_name["2Pi" + suffix][2 + mixing_index]
        high = rows_by_name["Jets-ud" + suffix][2 + mixing_index]
        if any(low_point[1] > 0.0 and high_point[1] > 0.0
               for low_point, high_point in zip(low, high)):
            _fail("charged two-pion and ud residual owners overlap for %s" %
                  suffix)
    neutral_low = rows_by_name["2Piv"]
    for mixing_index, mixing in enumerate(MIXING_NAMES):
        low = neutral_low[2 + mixing_index]
        up = rows_by_name["Jets-uuv"][2 + mixing_index]
        down = rows_by_name["Jets-ddv"][2 + mixing_index]
        if any(low_point[1] > 0.0 and
               (up_point[1] > 0.0 or down_point[1] > 0.0)
               for low_point, up_point, down_point in zip(low, up, down)):
            _fail("neutral two-pion and ud residual owners overlap for %s" %
                  mixing)


def _persistent_pole_owners() -> dict[str, tuple[int, ...]]:
    owners = {name: (0, 1, 2) for name in NEUTRAL_PERSISTENT_POLES}
    for suffix, mixing_index in (("e", 0), ("mu", 1), ("tau", 2)):
        for prefix in ("Pi", "K", "D", "Ds"):
            name = prefix + suffix
            owners[name] = (mixing_index,)
            owners[name + "bar"] = (mixing_index,)
    return owners


PERSISTENT_POLE_OWNERS = _persistent_pole_owners()


def _validate_persistent_poles(rows_by_name: dict[str, list[Any]]) -> None:
    for channel, owners in PERSISTENT_POLE_OWNERS.items():
        row = rows_by_name[channel]
        for mixing_index in owners:
            table = row[2 + mixing_index]
            positive_indices = tuple(
                index for index, (_, value) in enumerate(table) if value > 0.0)
            if not positive_indices:
                _fail("persistent pole %s/%s never opens" %
                      (channel, MIXING_NAMES[mixing_index]))
            first = positive_indices[0]
            if any(value <= 0.0 for _, value in table[first:]):
                _fail("persistent pole %s/%s switches off after opening" %
                      (channel, MIXING_NAMES[mixing_index]))


def validate_decay_rows(rows: Any) -> tuple[float, ...]:
    """Validate decoded HNL decay rows and return their common mass grid."""

    if not isinstance(rows, list):
        _fail("HNL decay JSON root must be a list")
    if len(rows) != EXPECTED_CHANNEL_COUNT:
        _fail("expected %d HNL rows, found %d" %
              (EXPECTED_CHANNEL_COUNT, len(rows)))

    names: list[str] = []
    pdg_rows: list[tuple[int, ...]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 8:
            _fail("HNL row %d must be an 8-element list" % index)
        name = row[0]
        if not isinstance(name, str) or not name:
            _fail("HNL row %d has no valid name" % index)
        names.append(name)
        pdgs = row[1]
        if (not isinstance(pdgs, list) or len(pdgs) != 4 or
                any(not isinstance(pdg, int) or isinstance(pdg, bool)
                    for pdg in pdgs)):
            _fail("%s must contain exactly four integer PDGs" % name)
        pdg_rows.append(tuple(pdgs))

    if len(set(names)) != len(names):
        _fail("HNL channel names are not unique")
    if set(names) != EXPECTED_CHANNEL_NAMES:
        missing = sorted(EXPECTED_CHANNEL_NAMES.difference(names))
        extra = sorted(set(names).difference(EXPECTED_CHANNEL_NAMES))
        _fail("unexpected HNL inventory (missing=%s, extra=%s)" %
              (missing, extra))
    if len(set(pdg_rows)) != len(pdg_rows):
        _fail("HNL decay PDG rows are not unique")

    first_table = rows[0][2]
    if not isinstance(first_table, list):
        _fail("first HNL branching table is not a list")
    if len(first_table) != EXPECTED_MASS_NODES:
        _fail("expected %d HNL mass nodes, found %d" %
              (EXPECTED_MASS_NODES, len(first_table)))
    mass_grid_list: list[float] = []
    for point_index, point in enumerate(first_table):
        if (not isinstance(point, list) or len(point) != 2 or
                not all(_is_number(value) for value in point)):
            _fail("invalid first branching-grid point %d" % point_index)
        mass = float(point[0])
        if not math.isfinite(mass):
            _fail("non-finite HNL mass at grid point %d" % point_index)
        mass_grid_list.append(mass)
    if any(right <= left for left, right in
           zip(mass_grid_list, mass_grid_list[1:])):
        _fail("HNL mass grid is not strictly increasing")
    mass_grid = tuple(mass_grid_list)

    for row in rows:
        name = row[0]
        for mixing_index, mixing in enumerate(MIXING_NAMES):
            table = row[2 + mixing_index]
            if not isinstance(table, list) or len(table) != len(mass_grid):
                _fail("%s/%s does not use the canonical mass-grid length" %
                      (name, mixing))
            has_positive = False
            for point_index, point in enumerate(table):
                if (not isinstance(point, list) or len(point) != 2 or
                        not all(_is_number(value) for value in point)):
                    _fail("%s/%s has invalid point %d" %
                          (name, mixing, point_index))
                mass, branching = map(float, point)
                if mass != mass_grid[point_index]:
                    _fail("%s/%s has a non-canonical mass at point %d" %
                          (name, mixing, point_index))
                if (not math.isfinite(branching) or branching < 0.0 or
                        branching > 1.0):
                    _fail("%s/%s has branching ratio outside [0,1] at m=%g" %
                          (name, mixing, mass))
                has_positive = has_positive or branching > 0.0
            _validate_matrix_element(
                row[5 + mixing_index], name, mixing, has_positive)

    rows_by_name = {row[0]: row for row in rows}
    _validate_thresholds(rows_by_name)
    _validate_conjugates(rows_by_name)
    _validate_current_and_mixing_ownership(rows_by_name)
    _validate_persistent_poles(rows_by_name)
    return mass_grid


def read_width_rows(path: str | Path) -> list[tuple[float, float, float, float]]:
    """Decode a four-column HNL total-width table."""

    rows: list[tuple[float, float, float, float]] = []
    try:
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 4:
                    _fail("HNL width line %d has %d columns, expected 4" %
                          (line_number, len(fields)))
                try:
                    values = tuple(float(field) for field in fields)
                except ValueError as exc:
                    _fail("HNL width line %d is not numeric: %s" %
                          (line_number, exc))
                rows.append(values)  # type: ignore[arg-type]
    except HNLSourceValidationError:
        raise
    except (OSError, UnicodeError) as exc:
        _fail("cannot read HNL width table %s: %s" % (path, exc))
    return rows


def validate_width_rows(
    width_rows: Any,
    mass_grid: Sequence[float],
) -> None:
    """Validate decoded total widths against the exact branching grid."""

    if not isinstance(width_rows, list) or len(width_rows) != len(mass_grid):
        _fail("HNL width table length does not match the branching grid")
    for index, row in enumerate(width_rows):
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            _fail("HNL width row %d must contain four columns" % index)
        if any(not _is_number(value) for value in row):
            _fail("HNL width row %d is not numeric" % index)
        values = tuple(float(value) for value in row)
        if not all(math.isfinite(value) for value in values):
            _fail("HNL width row %d contains a non-finite value" % index)
        if values[0] != mass_grid[index]:
            _fail("HNL width grid differs at point %d" % index)
        if any(value <= 0.0 for value in values[1:]):
            _fail("HNL total width is not positive at m=%g" % values[0])


def _validate_visible_closure(
    decay_rows: Sequence[Sequence[Any]],
    width_rows: Sequence[Sequence[float]],
) -> None:
    coefficient = (2.0 * FERMI_CONSTANT_GEV ** 2 /
                   (192.0 * math.pi ** 3))
    for point_index, width_row in enumerate(width_rows):
        mass = float(width_row[0])
        invisible_width = coefficient * mass ** 5
        for mixing_index, mixing in enumerate(MIXING_NAMES):
            observed = sum(
                float(row[2 + mixing_index][point_index][1])
                for row in decay_rows)
            expected = 1.0 - invisible_width / float(
                width_row[1 + mixing_index])
            tolerance = (VISIBLE_CLOSURE_ABS_TOL +
                         VISIBLE_CLOSURE_REL_TOL *
                         max(abs(observed), abs(expected)))
            if abs(observed - expected) > tolerance:
                _fail(
                    "visible/invisible closure fails for %s at m=%g: "
                    "rows=%g, expected=%g" %
                    (mixing, mass, observed, expected))


def load_validated_hnl_source(
    decay_json_path: str | Path,
    width_table_path: str | Path,
    metadata_path: str | Path | None = None,
) -> ValidatedHNLSource:
    """Load and fully validate one HNL decay JSON/width-table pair.

    If *metadata_path* is supplied it is mandatory.  Otherwise an adjacent
    standard metadata filename is used when present.  Metadata is optional for
    source-development directories, but any discovered metadata must contain
    both expected SHA-256 values and must match exactly.
    """

    decay_path = Path(decay_json_path)
    width_path = Path(width_table_path)
    for label, path in (("decay JSON", decay_path),
                        ("width table", width_path)):
        if not path.is_file():
            _fail("HNL %s does not exist: %s" % (label, path))

    selected_metadata: Path | None
    if metadata_path is None:
        selected_metadata = _discover_metadata(decay_path)
    else:
        selected_metadata = Path(metadata_path)
        if not selected_metadata.is_file():
            _fail("HNL source metadata does not exist: %s" %
                  selected_metadata)

    decay_digest = sha256_file(decay_path)
    width_digest = sha256_file(width_path)
    metadata: Any = None
    if selected_metadata is not None:
        metadata = _read_json(selected_metadata)
        _validate_metadata(metadata, selected_metadata,
                           decay_digest, width_digest)

    decay_rows = _read_json(decay_path)
    mass_grid = validate_decay_rows(decay_rows)
    width_rows = read_width_rows(width_path)
    validate_width_rows(width_rows, mass_grid)
    _validate_visible_closure(decay_rows, width_rows)
    if selected_metadata is not None:
        _validate_metadata(metadata, selected_metadata,
                           decay_digest, width_digest, mass_grid)

    return ValidatedHNLSource(
        decay_rows=decay_rows,
        width_rows=width_rows,
        mass_grid=mass_grid,
        decay_json_sha256=decay_digest,
        width_table_sha256=width_digest,
        metadata_path=selected_metadata,
    )


def load_canonical_hnl_source(
    distribution_directory: str | Path | None = None,
) -> ValidatedHNLSource:
    """Load the installed EventCalc HNL source, validating before return."""

    if distribution_directory is None:
        distribution_directory = (
            Path(__file__).resolve().parents[1] / "Distributions" / "HNL")
    directory = Path(distribution_directory)
    return load_validated_hnl_source(
        directory / "HNL-decay.json",
        directory / "HNLdecayWidth.dat",
    )


def main() -> None:
    source = load_canonical_hnl_source()
    print(
        "Validated HNL source: channels=%d nodes=%d json=%s widths=%s" %
        (len(source.decay_rows), len(source.mass_grid),
         source.decay_json_sha256, source.width_table_sha256))


if __name__ == "__main__":
    main()
