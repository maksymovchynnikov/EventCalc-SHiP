#!/usr/bin/env python3
# events_pointing_analysis.py
"""
Post-processing of EventCalc event files: daughter-level pointing requirement.

For every decay event, the visible decay products (charged particles, photons,
and K_L; neutrinos are ignored) are propagated along straight lines from the
decay vertex to the plane z = Z_PLANE, which lies downstream of the decay
volume. The event passes if at least two products cross the plane within
|x| < X_HALF, |y| < Y_HALF.

Events are weighted with their decay probability P_decay, so the pointing
efficiency is
    eps_point = sum_{passing events} P_decay / sum_{all events} P_decay,
and the corrected number of events is N_ev^point = N_ev_tot * eps_point,
with N_ev_tot taken from the file header.

Also computed: the same efficiency with an additional momentum threshold
p > 1 GeV per counted product (the SHiP baseline daughter cut).

Usage:
    python3 events_pointing_analysis.py [--pattern GLOB] [--out CSV]
"""

import argparse
import glob
import os
import re
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from funcs.fast_event_io import read_event_file, map_event_files
from events_analysis import (
    DETECTABLE_PDGS,
    DEFAULT_DETECTOR_Z_M,
    DEFAULT_DETECTOR_WIDTH_M,
    DEFAULT_DETECTOR_HEIGHT_M,
)

# The plane on which the daughters are counted sits at z = 95 m, downstream of
# the decay volume, which ends at z = 82 m.
Z_PLANE = DEFAULT_DETECTOR_Z_M
X_HALF = DEFAULT_DETECTOR_WIDTH_M / 2.0
Y_HALF = DEFAULT_DETECTOR_HEIGHT_M / 2.0
P_CUT = 1.0        # GeV, for the "baseline momentum" variant

DETECTABLE = DETECTABLE_PDGS


_DETECTABLE_ARR = np.array(sorted(DETECTABLE))


def pointing_pass(vx, vy, vz, plist, p_cut=0.0):
    """Single-event pointing test (kept for backward compatibility with
    callers that iterate events themselves; the batch path in analyze_file
    uses the vectorized _channel_pass_weights instead)."""
    npass = 0
    for row in plist:
        px, py, pz = row[0], row[1], row[2]
        pdg = int(row[5])
        if pdg == -999 or pdg not in DETECTABLE or pz <= 0.0:
            continue
        if p_cut > 0.0 and np.sqrt(px * px + py * py + pz * pz) < p_cut:
            continue
        dz = Z_PLANE - vz
        if dz <= 0.0:
            continue
        if abs(vx + px / pz * dz) < X_HALF and abs(vy + py / pz * dz) < Y_HALF:
            npass += 1
        if npass >= 2:
            return True
    return npass >= 2


def parse_event_file(path):
    """Backward-compatible per-event view built on the vectorized reader.
    Returns (n_ev_tot, [(P_decay, vx, vy, vz, products[n,6]), ...])."""
    n_ev_tot, channels = read_event_file(path)
    events = []
    for arr in channels.values():
        if arr.size == 0:
            continue
        nprod = (arr.shape[1] - 10) // 6
        prod = arr[:, 10:10 + 6 * nprod].reshape(arr.shape[0], nprod, 6)
        for i in range(arr.shape[0]):
            events.append((arr[i, 6], arr[i, 7], arr[i, 8], arr[i, 9], prod[i]))
    return n_ev_tot, events


def _channel_pass_weights(arr, p_cut):
    """Vectorized over all events of one channel array (n_ev, n_fields).

    Returns (w_sum, w_pass) where the weight is P_decay (column 6) and an
    event passes if >= 2 detectable products cross the z = Z_PLANE aperture.
    """
    if arr.size == 0 or arr.shape[1] < 16:
        return 0.0, 0.0
    P_decay = arr[:, 6]
    vz = arr[:, 9]
    nprod = (arr.shape[1] - 10) // 6
    prod = arr[:, 10:10 + 6 * nprod].reshape(arr.shape[0], nprod, 6)
    px, py, pz = prod[:, :, 0], prod[:, :, 1], prod[:, :, 2]
    pdg = prod[:, :, 5]

    detectable = np.isin(pdg.astype(np.int64), _DETECTABLE_ARR)
    forward = pz > 0.0
    dz = Z_PLANE - vz[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        xf = arr[:, 7][:, None] + np.where(forward, px / pz, 0.0) * dz
        yf = arr[:, 8][:, None] + np.where(forward, py / pz, 0.0) * dz
    in_aper = (np.abs(xf) < X_HALF) & (np.abs(yf) < Y_HALF)
    ok = detectable & forward & (dz > 0.0) & in_aper
    if p_cut > 0.0:
        p = np.sqrt(px * px + py * py + pz * pz)
        ok &= p >= p_cut
    npass = ok.sum(axis=1)
    passed = npass >= 2
    return float(P_decay.sum()), float(P_decay[passed].sum())


def analyze_file(path):
    """Per-file pointing efficiencies; picklable for process-pool sharding."""
    n_ev_tot, channels = read_event_file(path)
    n_sampled = sum(a.shape[0] for a in channels.values())
    if n_sampled == 0:
        return None
    wsum = wpass = wpass_pcut = 0.0
    for arr in channels.values():
        ws, wp = _channel_pass_weights(arr, 0.0)
        _, wpc = _channel_pass_weights(arr, P_CUT)
        wsum += ws
        wpass += wp
        wpass_pcut += wpc
    eps = wpass / wsum if wsum > 0 else 0.0
    eps_pcut = wpass_pcut / wsum if wsum > 0 else 0.0
    return {
        "file": os.path.basename(path),
        "N_ev_tot": n_ev_tot,
        "n_sampled": n_sampled,
        "eps_point": eps,
        "eps_point_pcut": eps_pcut,
        "N_ev_point": n_ev_tot * eps,
        "N_ev_point_pcut": n_ev_tot * eps_pcut,
    }


def parse_name(fname):
    """Extract (mass, ctau, label) from '<LLP>_<mass>_<ctau>_<label>_data.dat'."""
    tokens = fname.replace("_data.dat", "").split("_")
    # tokens: [LLP, mass, ctau, label...]
    mass = float(tokens[1])
    ctau = float(tokens[2])
    label = "_".join(tokens[3:]) if len(tokens) > 3 else ""
    return mass, ctau, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern",
                    default="outputs/Dark-photons/eventData/*_data.dat")
    ap.add_argument("--out", default="outputs/Dark-photons/pointing_summary.csv")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.pattern))
    # worker-sharded over files (EVENTCALC_WORKERS / EXHAD_WORKERS env var)
    results = map_event_files(analyze_file, paths)
    rows = []
    for path, res in zip(paths, results):
        if res is None:
            continue
        mass, ctau, label = parse_name(res["file"])
        res.update({"mass": mass, "c_tau": ctau, "label": label})
        rows.append(res)
        print(f"{res['file']}: eps_point={res['eps_point']:.4f} "
              f"(p>1GeV: {res['eps_point_pcut']:.4f})  "
              f"N_ev={res['N_ev_tot']:.4e} -> N_point={res['N_ev_point']:.4e}")

    if rows:
        import csv
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\nSummary written to {args.out}")


if __name__ == "__main__":
    main()
