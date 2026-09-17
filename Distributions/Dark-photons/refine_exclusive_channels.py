#!/usr/bin/env python3
"""Replace legacy dark-photon aggregate placeholders by DeLiVeR modes.

The legacy EventCalc table carried the correct *sum* of several DeLiVeR
exclusive widths, but attached each sum to one placeholder particle list.
That is harmless for lifetimes and inclusive rates and wrong for event
topologies.  This migration splits only the three affected aggregates below
``m_T = 1.70 GeV``.  At every source mass the sum of the replacement rows is
bit-for-bit the former aggregate value; leptonic, named-exclusive, partonic,
total-width, and lifetime inputs are untouched.

Relative submode weights come from the signed ReD-DeLiVeR tables already
distributed in ``exhad/refdata/red_deliver_rtables/exclusive``.  Intermediate
omega/phi states are kept as such and decayed by the existing Pythia step.
Five/six-pion states are explicit flat phase-space rows, which EventCalc then
boosts and propagates through its existing geometry.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TABLE = HERE / "DP-decay.json"
PROVENANCE = HERE / "DP-decay-detailed-source.json"
SOURCE_DIR = (ROOT / "exhad" / "refdata" / "red_deliver_rtables" /
              "exclusive")
TRANSITION_GEV = 1.70


def _variants(*entries):
    return tuple(entries)


# (source submode, ((row label, PDGs, fraction within source submode), ...))
GROUPS = {
    "KK": (
        ("KK_n", _variants(("KL_KS", [130, 310], 1.0))),
        ("KK_c", _variants(("Kp_Km", [321, -321], 1.0))),
    ),
    "KKpi": (
        ("KKpi_0", _variants(
            ("KKpi_0_KL_KS_Pi0", [130, 310, 111], 1.0))),
        ("KKpi_1", _variants(
            ("Kp_Km_Pi0", [321, -321, 111], 1.0))),
        ("KKpi_2", _variants(
            ("KKpi_2_KS_plus", [310, 321, -211], 0.25),
            ("KKpi_2_KS_minus", [310, -321, 211], 0.25),
            ("KKpi_2_KL_plus", [130, 321, -211], 0.25),
            ("KKpi_2_KL_minus", [130, -321, 211], 0.25))),
    ),
    "rho_other": (
        ("EtaPiPi", _variants(("EtaPiPi", [221, 211, -211], 1.0))),
        ("EtaPrimePiPi", _variants(
            ("EtaPrimePiPi", [331, 211, -211], 1.0))),
        ("6pi_n", _variants(
            ("6pi_n", [211, 211, -211, -211, 111, 111], 1.0))),
        ("6pi_c", _variants(
            ("6pi_c", [211, 211, 211, -211, -211, -211], 1.0))),
        ("OmegaPion", _variants(("OmegaPion", [223, 111], 1.0))),
        ("PhiPi", _variants(("PhiPi", [333, 111], 1.0))),
        ("ppbar", _variants(("ppbar", [2212, -2212], 1.0))),
        ("nnbar", _variants(("nnbar", [2112, -2112], 1.0))),
    ),
    "omega_other": (
        ("EtaGamma", _variants(("EtaGamma", [221, 22], 1.0))),
        ("EtaOmega", _variants(("EtaOmega", [221, 223], 1.0))),
        ("OmPiPi_n", _variants(("OmPiPi_n", [223, 111, 111], 1.0))),
        ("OmPiPi_c", _variants(("OmPiPi_c", [223, 211, -211], 1.0))),
    ),
    "phi_other": (
        ("EtaPhi", _variants(("EtaPhi", [221, 333], 1.0))),
        ("KKpipi_0", _variants(
            ("KKpipi_0", [321, -321, 211, -211], 1.0))),
        # DeLiVeR quotes three independent fits to the same inclusive
        # KS K^+- pi^-+ pi0 topology.  Neutral-kaon propagation requires the
        # physical 50:50 KS/KL mixture, and charge conjugates are equiprobable.
        ("KKpipi_1", _variants(
            ("KKpipi_1_KS_plus", [310, 321, -211, 111], 0.25),
            ("KKpipi_1_KS_minus", [310, -321, 211, 111], 0.25),
            ("KKpipi_1_KL_plus", [130, 321, -211, 111], 0.25),
            ("KKpipi_1_KL_minus", [130, -321, 211, 111], 0.25))),
        ("KKpipi_2", _variants(
            ("KKpipi_2_KS_plus", [310, 321, -211, 111], 0.25),
            ("KKpipi_2_KS_minus", [310, -321, 211, 111], 0.25),
            ("KKpipi_2_KL_plus", [130, 321, -211, 111], 0.25),
            ("KKpipi_2_KL_minus", [130, -321, 211, 111], 0.25))),
        ("KKpipi_3", _variants(
            ("KKpipi_3_KS_plus", [310, 321, -211, 111], 0.25),
            ("KKpipi_3_KS_minus", [310, -321, 211, 111], 0.25),
            ("KKpipi_3_KL_plus", [130, 321, -211, 111], 0.25),
            ("KKpipi_3_KL_minus", [130, -321, 211, 111], 0.25))),
        ("PhiPiPi_n", _variants(("PhiPiPi_n", [333, 111, 111], 1.0))),
        ("PhiPiPi_c", _variants(("PhiPiPi_c", [333, 211, -211], 1.0))),
    ),
}


# Legacy rows whose *sum* is retained for each detailed group.  KK and KKpi
# already had physical-looking rows, but their equal charge-mode splits were
# not the DeLiVeR decomposition and left a residual topology step at m_T.
LEGACY_ROWS = {
    "KK": ("KL_KS", "Kp_Km"),
    "KKpi": ("KS_Km_Pip", "KS_Kp_Pim", "Kp_Km_Pi0"),
    "rho_other": ("rho_other",),
    "omega_other": ("omega_other",),
    "phi_other": ("phi_other,1", "phi_other,2"),
}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(payload):
    encoded = json.dumps(payload, separators=(",", ":"),
                         sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rate_series_sha256(values):
    # JSON summation of many split rows can differ by a few last binary bits
    # solely from addition order.  Fourteen decimal places is substantially
    # tighter than any physical input precision while retaining a stable
    # exact regression digest.
    return _payload_sha256([round(float(value), 14) for value in values])


def _read_curve(name):
    path = SOURCE_DIR / ("R.%s.dat" % name)
    masses, values = [], []
    with open(path) as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split()
            masses.append(float(fields[0]))
            values.append(float(fields[1]))
    if len(masses) < 2 or any(b <= a for a, b in zip(masses, masses[1:])):
        raise RuntimeError("invalid ReD-DeLiVeR curve %s" % path)
    return path, masses, values


def _interp(mass, grid, values):
    if mass <= grid[0]:
        return values[0]
    if mass >= grid[-1]:
        return values[-1]
    index = bisect.bisect_left(grid, mass)
    fraction = ((mass - grid[index - 1]) /
                (grid[index] - grid[index - 1]))
    return values[index - 1] + fraction * (
        values[index] - values[index - 1])


def _group_rows(group, mass_grid, aggregate):
    curves = {}
    for source, _ in GROUPS[group]:
        _, grid, values = _read_curve(source)
        curves[source] = (grid, values)

    series = {}
    pdgs = {}
    source_for_label = {}
    for source, variants in GROUPS[group]:
        if abs(sum(fraction for _, _, fraction in variants) - 1.0) > 1e-14:
            raise RuntimeError("variants of %s do not sum to one" % source)
        for label, ids, _ in variants:
            if label in series:
                raise RuntimeError("duplicate detailed row label %s" % label)
            series[label] = []
            pdgs[label] = ids
            source_for_label[label] = source

    for mass, total in zip(mass_grid, aggregate):
        source_values = {
            source: max(_interp(mass, *curves[source]), 0.0)
            for source, _ in GROUPS[group]
        }
        denominator = sum(source_values.values())
        if total > 1.0e-18 and denominator <= 0.0:
            raise RuntimeError(
                "%s is positive at m=%g but its DeLiVeR submodes vanish" %
                (group, mass))
        for source, variants in GROUPS[group]:
            share = (total * source_values[source] / denominator
                     if denominator > 0.0 else 0.0)
            for label, _, fraction in variants:
                series[label].append([mass, share * fraction])

        reconstructed = sum(series[label][-1][1] for label in series)
        if abs(reconstructed - total) > 2.0e-15:
            raise AssertionError(
                "%s split fails closure at m=%g: %.17g != %.17g" %
                (group, mass, reconstructed, total))

    rows = []
    for source, variants in GROUPS[group]:
        for label, _, _ in variants:
            rows.append([label, pdgs[label], series[label], "1."])
    return rows


def _atomic_json(path, payload):
    descriptor, temporary = tempfile.mkstemp(
        prefix=".%s." % path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent="\t")
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def refine(table_path=TABLE):
    table_path = Path(table_path)
    with open(table_path) as stream:
        table = json.load(stream)
    before_hash = _sha256(table_path)
    by_label = {str(row[0]): row for row in table}
    legacy = {label for labels in LEGACY_ROWS.values() for label in labels}
    if not legacy.issubset(by_label):
        raise RuntimeError(
            "aggregate source rows are absent; table may already be refined")

    mass_grid = [float(point[0]) for point in by_label["rho_other"][2]]
    for label in legacy:
        row_grid = [float(point[0]) for point in by_label[label][2]]
        if row_grid != mass_grid:
            raise RuntimeError("aggregate row grids differ")

    aggregate = {
        group: [
            sum(float(by_label[label][2][index][1]) for label in labels)
            for index in range(len(mass_grid))
        ]
        for group, labels in LEGACY_ROWS.items()
    }
    detailed = {
        group: _group_rows(group, mass_grid, values)
        for group, values in aggregate.items()
    }

    first_legacy = {labels[0]: group
                    for group, labels in LEGACY_ROWS.items()}
    output = []
    for row in table:
        label = str(row[0])
        if label in first_legacy:
            output.extend(detailed[first_legacy[label]])
        elif label in legacy:
            continue
        else:
            output.append(row)

    labels = [str(row[0]) for row in output]
    if len(labels) != len(set(labels)):
        raise RuntimeError("refined dark-photon table has duplicate labels")
    _atomic_json(table_path, output)

    sources = sorted({source for group in GROUPS.values()
                      for source, _ in group})
    provenance = {
        "schema": 1,
        "purpose": "replace aggregate placeholder particle identities below m_T without changing rates",
        "transition_GeV": TRANSITION_GEV,
        "aggregate_source_sha256": before_hash,
        "refined_table_sha256": _sha256(table_path),
        "source_directory": str(SOURCE_DIR.relative_to(ROOT)),
        "source_sha256": {
            ("R.%s.dat" % source): _sha256(
                SOURCE_DIR / ("R.%s.dat" % source))
            for source in sources
        },
        "groups": {
            group: [source for source, _ in specifications]
            for group, specifications in GROUPS.items()
        },
        "replacement_rows": {
            group: [label for _, variants in specifications
                    for label, _, _ in variants]
            for group, specifications in GROUPS.items()
        },
        "aggregate_rate_series_sha256": {
            group: _rate_series_sha256(values)
            for group, values in aggregate.items()
        },
        "all_channel_rate_series_sha256": _rate_series_sha256([
            sum(float(row[2][index][1]) for row in table)
            for index in range(len(mass_grid))
        ]),
        "checkpoints": {
            ("%.3f" % mass): {
                group: aggregate[group][next(
                    index for index, value in enumerate(mass_grid)
                    if abs(value - mass) < 1.0e-10)]
                for group in aggregate
            }
            for mass in (1.600, 1.690, 1.700)
        },
        "rate_contract": "replacement rows sum to the former aggregate at every tabulated mass",
        "kinematics": "flat phase space for direct multibody rows; omega/phi are passed to the existing Pythia decay step",
    }
    _atomic_json(PROVENANCE, provenance)
    return provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default=str(TABLE))
    args = parser.parse_args()
    metadata = refine(args.table)
    print("refined %s" % args.table)
    print("sha256 %s" % metadata["refined_table_sha256"])


if __name__ == "__main__":
    main()
