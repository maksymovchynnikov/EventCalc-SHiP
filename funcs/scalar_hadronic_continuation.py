"""Explicit two-meson extrapolation for the portable scalar model only.

Original rate tables are never modified. At their last exclusive point,
hold each two-body amplitude fixed: Gamma(m)/Gamma(m0) =
(m0/m)*beta(m)/beta(m0). This is a model assumption, not an exclusive
calculation above 2 GeV. The original total hadronic, leptonic, nucleon
and charm rates stay fixed. Subtract the two-meson widths before assigning
the remaining light-hadronic width to fragmentation. exHad joins this matched
accounting to showered supplied rows over 4-5 GeV with h = 6x^5-15x^4+10x^3,
x = (m-4 GeV)/GeV; from 5 GeV the supplied rows are returned unchanged.
"""
from bisect import bisect_left
from functools import lru_cache
import json
import math
from pathlib import Path

POLICY = "scalar-endpoint-amplitude-v1"
EDGE = 1.999
SWITCH = 2.0
HBARC = 1.973269804e-16
TWO_MESON = {"PipPim": .13957039, "2Pi0": .1349768,
             "2Kch": .493677, "KLKL": .497611, "KSKS": .497611}
FOUR_PION = ("2Pip2Pim", "PipPim2Pi0")
PARTONS = ("Jets-GG", "Jets-ss")
NONHADRONIC = frozenset(("ePeM", "muPmuM", "tauPtauM"))


def linear(xs, ys, m):
    if not math.isfinite(m) or m < xs[0] or m > xs[-1]:
        raise ValueError("scalar query outside the supplied rate grid")
    i = bisect_left(xs, m)
    if i == 0 or xs[i] == m:
        return ys[i]
    t = (m-xs[i-1])/(xs[i]-xs[i-1])
    return ys[i-1]+t*(ys[i]-ys[i-1])


class ScalarHadronicContinuation:
    def __init__(self, table, lifetime=None):
        self.table = Path(table).resolve()
        self.lifetime = (Path(lifetime).resolve() if lifetime else
                         self.table.with_name(self.table.name.replace("BrRatio-", "ctau-").replace(".json", ".txt")))
        rows = json.loads(self.table.read_text())
        self.labels = tuple(row[0] for row in rows)
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("repeated scalar channel")
        self.curves = {row[0]: tuple(zip(*row[2])) for row in rows}
        nodes = json.loads(self.lifetime.read_text())
        self.total_curve = ([float(x[0]) for x in nodes], [HBARC/float(x[1]) for x in nodes])
        self.edge = self.native(EDGE)
        if any(self.edge[k] <= 0 for k in (*TWO_MESON, *FOUR_PION)):
            raise ValueError("exclusive scalar endpoint is incomplete")
        if any(self.edge[k] != 0 for k in PARTONS):
            raise ValueError("exclusive scalar endpoint contains partons")
        self.edge_width = self.total_width(EDGE)
        norm = math.fsum(self.edge[k] for k in FOUR_PION)
        self.four_pion_ratios = {k: self.edge[k]/norm for k in FOUR_PION}

    def native(self, mass):
        return {k: float(linear(*v, mass)) for k,v in self.curves.items()}

    def total_width(self, mass):
        return float(linear(*self.total_curve, mass))

    def at(self, mass):
        mass = float(mass)
        raw = self.native(mass)
        if mass <= EDGE:
            return raw
        if mass >= 5.0:
            return raw
        if any(v < -1e-15 or not math.isfinite(v) for v in raw.values()):
            raise ValueError("invalid scalar source rates")
        hadronic = math.fsum(v for k,v in raw.items() if k not in NONHADRONIC)
        result = dict(raw)
        total = self.total_width(mass)
        for label, daughter in TWO_MESON.items():
            beta = math.sqrt(1-4*daughter*daughter/mass**2)
            beta0 = math.sqrt(1-4*daughter*daughter/EDGE**2)
            result[label] = self.edge[label]*self.edge_width/total*(EDGE/mass)*beta/beta0
        # NN, charm, inactive-heavy and leptonic rows are copied unchanged.
        residual = hadronic - math.fsum(v for k,v in result.items()
                        if k not in NONHADRONIC and k not in (*FOUR_PION, *PARTONS))
        if residual < 0:
            raise ValueError("continued two-meson rates exceed the inclusive hadronic width")
        for label in (*FOUR_PION, *PARTONS):
            result[label] = 0.0
        if mass < SWITCH:
            # Bridge only the final 1-MeV table gap. As m -> 2-, the entire
            # remaining width is 4pi, exactly as the p=2 runtime gives at 2+.
            weights = self.four_pion_ratios
        else:
            norm = math.fsum(raw[k] for k in PARTONS)
            if norm <= 0:
                raise ValueError("missing inclusive scalar gg/ss mixture")
            weights = {k: raw[k]/norm for k in PARTONS}
        for label, probability in weights.items():
            result[label] = residual*probability
        correction = hadronic-math.fsum(v for k,v in result.items() if k not in NONHADRONIC)
        result[max(weights, key=weights.get)] += correction
        if any(v < 0 for v in result.values()):
            raise ValueError("scalar continuation produced a negative rate")
        if mass > 4.0:
            x = mass-4.0
            h = x*x*x*(10.0+x*(-15.0+6.0*x))
            for label in (*TWO_MESON, *PARTONS):
                result[label] = (1.0-h)*result[label] + h*raw[label]
        return result


@lru_cache(maxsize=8)
def load_continuation(table, lifetime=None):
    return ScalarHadronicContinuation(table, lifetime)
