#!/usr/bin/env python3
"""Canonicalize the Mathematica-exported fermion-ALP decay card.

Run this immediately after exporting ``ALP-fermion-decay.json``.  It gives
every PDG multiset a stable ASCII label, requires the canonical gluon id +21,
replaces irrelevant non-three-body dispatcher placeholders by ``1.``, requires
matrix elements (including ``Piecewise`` expressions) to have been exported
in Python/SymPy syntax, supplies phase space for the one nonzero three-body
row whose exported expression vanishes identically, zeros kinematically closed
rows using EventCalc's own runtime particle masses, rescales the accompanying
``ctau`` table by the same surviving-width fraction, and normalizes all listed
branching fractions point by point.

The lifetime normalization is unchanged by this post-processing.  For this
model EventCalc uses ``g_Y == y == 2*v_h/f_a``; the second lifetime-table
column is ``c*tau(m, g_Y=1)`` in metres and scales as ``1/g_Y**2``.  The later
``y=v_h/f_a`` lifetime-plot annotation in the source SensCalc notebook is
stale and is not the normalization of this table.
"""

import json
import math
import os
import sys
import tempfile
from pathlib import Path


CARD = Path(__file__).with_name("ALP-fermion-decay.json")
CTAU = Path(__file__).with_name("ctau-ALP-fermion.txt")
EVENTCALC_ROOT = CARD.parents[2]
if str(EVENTCALC_ROOT) not in sys.path:
    sys.path.insert(0, str(EVENTCALC_ROOT))

from funcs import PDG  # noqa: E402


# Compare masses in GeV.  The tolerance is only for binary roundoff at an
# exactly open threshold; it is far smaller than the 1-MeV source grid.
KINEMATIC_THRESHOLD_TOL = 1.0e-12


def key(ids):
    return tuple(sorted(int(pdg) for pdg in ids))


NAME_BY_PDGS = {
    key([-11, 11]): "ePeM",
    key([-13, 13]): "muPmuM",
    key([-15, 15]): "tauPtauM",
    key([22, 22]): "2gamma",
    key([211, -211, 111]): "piPpiMpi0",
    key([211, -211, 22]): "piPpiMgamma",
    key([211, -211, 111, 111]): "piPpiM2pi0",
    key([211, 211, -211, -211]): "2piP2piM",
    key([221, 111, 111]): "eta2pi0",
    key([221, 211, -211]): "etapiPpiM",
    key([323, -323]): "KstarPKstarM",
    key([313, -313]): "Kstar0Kstar0bar",
    key([223, 211, -211]): "omegapiPpiM",
    key([111, 111, 111]): "3pi0",
    key([331, 111, 111]): "etaPrime2pi0",
    key([331, 211, -211]): "etaPrimepiPpiM",
    key([223, 223]): "2omega",
    key([21, 21]): "Jets-GG",
    key([4, -4]): "Jets-cc",
    key([3, -3]): "Jets-ss",
    key([130, 130, 111]): "2KLpi0",
    key([310, 310, 111]): "2KSpi0",
    key([130, 310, 111]): "KLKSpi0",
    key([-321, 130, 211]): "KMKLpiP",
    key([321, 130, -211]): "KPKLpiM",
    key([-321, 310, 211]): "KMKSpiP",
    key([321, 310, -211]): "KPKSpiM",
    key([321, -321, 111]): "KPKMpi0",
    key([113, 113]): "2rho0",
    key([213, -213]): "rhoPrhoM",
    key([2212, -2212]): "ppbar",
    key([2112, -2112]): "nnbar",
}

# Mathematica currently exports zero for this CP-basis component although its
# tabulated rate is nonzero. No calibrated differential model is supplied for
# it, so EventCalc uses the documented constant-matrix-element phase-space
# approximation instead of crashing on an all-zero rejection envelope.
ISOTROPIC_THREE_BODY = {"KLKSpi0"}


def require_python_matrix_syntax(expression, channel):
    """Reject Mathematica containers left by an incomplete source export."""
    if ("Piecewise({" in expression or "{" in expression or "}" in expression
            or "*^" in expression or "***" in expression):
        raise ValueError(
            "matrix element in row %r still uses Mathematica Piecewise/list "
            "syntax; fix toPython in the source exporter" % channel)
    return expression


def canonical_matrix_element_type(expression, channel, daughter_count):
    """Store source expressions in the string form consumed by EventCalc.

    Mathematica's JSON exporter writes a literal unit matrix element as the
    JSON number ``1.0``.  That value is sufficient for two- and four-body
    phase space, but three-body dynamics must remain an explicit source-
    exported expression.
    """
    if isinstance(expression, str):
        return expression
    if (daughter_count != 3 and isinstance(expression, (int, float)) and
            float(expression) == 1.0):
        return "1."
    raise ValueError("matrix element in row %r is not a supported source "
                     "expression" % channel)


def corrected_ctau(raw_ctau, source_total, allowed_total):
    """Remove a forbidden partial width consistently from the lifetime."""
    if not (math.isfinite(raw_ctau) and raw_ctau > 0.0):
        raise ValueError("ALP ctau must be finite and positive")
    if not (math.isfinite(source_total) and source_total > 0.0 and
            math.isfinite(allowed_total) and allowed_total > 0.0 and
            allowed_total <= source_total * (1.0 + 1.0e-12)):
        raise ValueError("invalid ALP allowed-width fraction")
    return raw_ctau * source_total / allowed_total


def load_ctau(path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError("invalid ctau row %d" % line_number)
            mass, value = map(float, fields)
            if not (math.isfinite(mass) and math.isfinite(value) and
                    value > 0.0):
                raise ValueError("non-finite/nonpositive ctau row %d" %
                                 line_number)
            if rows and mass <= rows[-1][1]:
                raise ValueError("ctau mass grid is not strictly increasing")
            rows.append((fields[0], mass, value))
    if not rows:
        raise ValueError("empty ALP ctau table")
    return rows


def main():
    with CARD.open(encoding="utf-8") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list) or not rows:
        raise ValueError("ALP decay card must be a nonempty row list")

    thresholds = []
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError("invalid row %d" % index)
        ids = row[1]
        if any(int(pdg) == -21 for pdg in ids):
            raise ValueError("row %d uses invalid gluon PDG id -21" % index)
        if any(float(pdg) != int(pdg) for pdg in ids):
            raise ValueError("row %d contains a non-integral PDG id" % index)
        try:
            row[0] = NAME_BY_PDGS[key(ids)]
        except KeyError as exc:
            raise ValueError("unknown PDG multiset in row %d: %r" %
                             (index, ids)) from exc
        row[1] = [float(int(pdg)) for pdg in ids]

        daughter_masses = [PDG.get_mass(int(pdg)) for pdg in ids]
        if any(not isinstance(mass, (int, float))
               for mass in daughter_masses):
            raise ValueError(
                "unknown EventCalc PDG mass in row %r: %r" %
                (row[0], ids))
        thresholds.append(sum(float(mass) for mass in daughter_masses))

        expression = canonical_matrix_element_type(
            row[3], row[0], len(ids))
        if expression.strip().startswith("Msquared3BodyLLP("):
            if len(ids) == 3:
                raise ValueError("three-body row %r has a dispatcher" % row[0])
            expression = "1."
        expression = require_python_matrix_syntax(expression, row[0])
        if expression.strip() in {"0", "0."} and any(
                float(value) > 0.0 for _, value in row[2]):
            if row[0] not in ISOTROPIC_THREE_BODY or len(ids) != 3:
                raise ValueError(
                    "nonzero row %r has an identically zero matrix element" %
                    row[0])
            expression = "1."
        row[3] = expression

    names = [row[0] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("canonical ALP channel names are not unique")
    if set(names) != set(NAME_BY_PDGS.values()):
        raise ValueError("ALP decay card does not contain the expected channels")

    grid = rows[0][2]
    if not grid:
        raise ValueError("empty ALP branching-fraction grid")
    for row in rows:
        if len(row[2]) != len(grid):
            raise ValueError("branching grids have unequal lengths")
        for i, ((mass, _), (reference_mass, _)) in enumerate(zip(row[2], grid)):
            if abs(float(mass) - float(reference_mass)) > 1.0e-10:
                raise ValueError("branching grids differ at index %d" % i)

    ctau_rows = load_ctau(CTAU)
    if len(ctau_rows) != len(grid):
        raise ValueError("ALP branching and ctau grids have unequal lengths")
    for i, ((mass, _), (_, ctau_mass, _)) in enumerate(
            zip(grid, ctau_rows)):
        if abs(float(mass) - ctau_mass) > 1.0e-10:
            raise ValueError("ALP branching and ctau grids differ at index %d" % i)

    largest_correction = 0.0
    thresholded_entries = 0
    corrected_lifetimes = []
    lifetime_corrections = 0
    largest_lifetime_factor = 1.0
    for i in range(len(grid)):
        mass = float(grid[i][0])
        source_values = [float(row[2][i][1]) for row in rows]
        if any(value < 0.0 for value in source_values):
            raise ValueError("negative branching fraction at index %d" % i)
        values = []
        changed = False
        for value, threshold in zip(source_values, thresholds):
            if mass + KINEMATIC_THRESHOLD_TOL < threshold and value != 0.0:
                value = 0.0
                changed = True
                thresholded_entries += 1
            values.append(value)
        source_total = sum(source_values)
        total = sum(values)
        if not total > 0.0:
            raise ValueError("nonpositive total branching fraction at index %d" % i)
        raw_ctau = ctau_rows[i][2]
        if changed:
            value_ctau = corrected_ctau(raw_ctau, source_total, total)
            lifetime_corrections += 1
            largest_lifetime_factor = max(
                largest_lifetime_factor, value_ctau / raw_ctau)
        else:
            value_ctau = raw_ctau
        corrected_lifetimes.append(value_ctau)
        largest_correction = max(largest_correction, abs(total - 1.0))
        # Preserve byte-stable provenance on repeated postprocessing. Totals
        # already closed at floating-point precision need no further rescale.
        if not changed and abs(total - 1.0) <= 1.0e-14:
            continue
        for row, value in zip(rows, values):
            row[2][i][1] = value / total

    card_mode = CARD.stat().st_mode
    ctau_mode = CTAU.stat().st_mode
    card_temporary = None
    ctau_temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=str(CARD.parent),
                delete=False) as stream:
            card_temporary = Path(stream.name)
            json.dump(rows, stream, ensure_ascii=False, indent="\t",
                      allow_nan=False)
            stream.write("\n")
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=str(CTAU.parent),
                delete=False) as stream:
            ctau_temporary = Path(stream.name)
            for (mass_token, _, _), value in zip(
                    ctau_rows, corrected_lifetimes):
                stream.write("%s\t%.17g\n" % (mass_token, value))
        os.chmod(card_temporary, card_mode)
        os.chmod(ctau_temporary, ctau_mode)
        os.replace(card_temporary, CARD)
        card_temporary = None
        os.replace(ctau_temporary, CTAU)
        ctau_temporary = None
    finally:
        for path in (card_temporary, ctau_temporary):
            if path is not None and path.exists():
                path.unlink()
    print("canonicalized %d channels and %d mass rows" %
          (len(rows), len(grid)))
    print("zeroed %d below-threshold branching-fraction entries" %
          thresholded_entries)
    print("rescaled %d lifetime rows; largest ctau factor = %.9g" %
          (lifetime_corrections, largest_lifetime_factor))
    print("largest pre-normalization |sum(BR)-1| = %.9g" %
          largest_correction)


if __name__ == "__main__":
    main()
