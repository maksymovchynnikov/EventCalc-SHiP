"""Print an ALP-photon launch card with mass-dependent coupling ranges.

Run: python3 examples/coupling_scan.py > cards/alp_photon_couplings.json
Then: python3 simulate.py --card cards/alp_photon_couplings.json
The edges below are illustrative scan choices, not calculated eventcalc.sensitivity limits.
"""

import json
import numpy as np


def coupling_grid(mass, lower, upper, n=15, alpha=0.8):
    """Interpolate positive [mass, coupling] edges in log-log space.

    alpha < 1 puts more points toward the upper coupling edge; alpha > 1
    puts more toward the lower edge. No extrapolation is performed.
    """
    if not np.isfinite(mass) or mass <= 0 or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("mass and alpha must be finite and positive")
    if isinstance(n, bool) or not np.isfinite(n) or int(n) != n or n < 1:
        raise ValueError("n must be a positive integer")
    edges = []
    for points in (lower, upper):
        points = np.asarray(points, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
                or not np.all(np.isfinite(points)) or np.any(points <= 0)
                or np.any(np.diff(points[:, 0]) <= 0)):
            raise ValueError("edges need positive [mass, coupling] rows with increasing masses")
        if not points[0, 0] <= mass <= points[-1, 0]:
            raise ValueError("mass lies outside a supplied edge; extrapolation is not allowed")
        edges.append(np.interp(np.log10(mass), np.log10(points[:, 0]),
                               np.log10(points[:, 1])))
    lo, hi = edges
    if lo > hi:
        raise ValueError("lower coupling edge exceeds upper edge")
    u = np.linspace(0, 1, int(n)) ** alpha
    return 10.0 ** (lo + u * (hi - lo))


def example_card():
    masses = np.geomspace(0.02, 2.5, 20)
    lower = [(0.02, 1e-7), (1.0, 1e-8), (2.5, 5e-8)]
    upper = [(0.02, 1e-2), (1.0, 5e-6), (2.5, 5e-7)]
    return {
        "model": "ALP-photon",
        "events": 10000,
        "masses": masses.tolist(),
        "couplings": [coupling_grid(m, lower, upper).tolist() for m in masses],
        "alp_production_mode": "primary",
        "decay_channels": "all",
        "exhad_mode": "off",
        "seed": 12345,
    }


if __name__ == "__main__":
    print(json.dumps(example_card(), indent=2))
