"""Threshold-aware channel gating: a closed channel has rate exactly zero.

EventCalc's branching-ratio tables are interpolated on a mass grid.  Between
two grid points an interpolant can return a small positive rate for a channel
whose daughters do not fit inside the parent.  Sampling such a row gives the
daughters a negative squared momentum, ``E**2 - m**2 < 0``, so the sampler
either returns a NaN four-vector or, if the square root is clamped, a particle
at rest whose energy is below its own rest mass.  Both records are unphysical,
and the clamped one is the more damaging of the two because it is finite and
therefore survives every downstream check in silence.

exHad's own tables are threshold-aware: a row's rate is exactly zero at and
below its threshold.  This module applies the same rule on the EventCalc side,
which keeps ownership of the rates:

* :func:`channel_threshold` is the smallest parent mass at which a row's final
  state fits;
* :func:`open_mask` marks the rows a parent of mass ``m`` can actually reach;
* :func:`gate_rates` returns the tabulated rates with the closed rows set to
  exactly zero -- this is what channel selection and event distribution use;
* :func:`require_open` is the last line of defence inside the samplers.  A
  closed channel that still reaches a sampler raises; it never returns
  particles.

Partonic rows
-------------
The quarks and gluons a row lists form one colour singlet, and that system
fragments into the hadrons the detector sees.  The colour field between the
partons is a string, which breaks through the creation of quark-antiquark
pairs, and the pieces form hadrons; every string fragmentation therefore ends
in two or more hadrons, and a final state holding a single hadron has zero
phase-space measure.  The smallest invariant mass such a system can carry is
the smallest mass of a two-hadron state that holds its conserved charges --
electric charge, strangeness, charm and beauty -- and that the current making
the pair can populate.  Two identical pseudoscalars are not such a state: a
vector or axial current needs them in ``L = 1``, and Bose symmetry forbids two
identical bosons there.  A charged current ``u dbar`` reaches ``pi+ pi0`` at
0.27455 GeV; ``u sbar`` reaches ``K+ pi0`` at 0.62866 GeV; ``c sbar`` reaches
``Ds+ pi0`` at 2.10333 GeV; a flavourless neutral pair such as ``d dbar``,
``s sbar`` or ``c cbar`` carries the charges of the vacuum and reaches
``pi+ pi-``, the lightest pair left to it, at 0.27914 GeV.  A scalar current
could reach ``pi0 pi0``; the two readings differ only between 0.26996 and
0.27914 GeV, and no shipped table gives a parton row a rate below 1.38 GeV, so
no row sees the difference.

That bound is the smallest mass the *system* can reach.  A hadronizer does not
reach that low: a string breaks by making light quark-antiquark pairs, so the
quark at each end stays in the hadron that end becomes, and the flavour of one
parton never appears in the hadron the other one makes.  ``s sbar`` reaches
``K+ K-`` at 0.98736 GeV that way, not ``pi+ pi-``.
:func:`string_end_two_hadron_mass` is that state, and
:mod:`funcs.ThreeBodyDecay` samples a pair above it because a hadronizer
handed anything lower has no final state to build.  The exHad release states a
minimum of its own on every pair it is handed, per current; those minima are
the release's to state, :mod:`funcs.ThreeBodyDecay` reads them from it when a
release is configured, and nothing here replaces them.

The decays into one meson are physical decays of a portal, and they are
channels of their own: a heavy neutral lepton reaches ``N -> l pi`` and
``N -> nu pi0`` through the axial-current matrix element
``<pi|A|0> = i f_pi p``, EventCalc's tables carry those decays as their own
rows, and exHad generates them without string fragmentation.  The two-hadron
bound applies to a row that holds partons, whose final state comes out of
string fragmentation.

Roundoff policy
---------------
``MOMENTUM_ROUNDOFF_REL`` bounds the *only* clamping that is allowed.  For an
open two-body channel the daughter momentum follows from

    p**2 = E1**2 - m1**2,    E1 = (m**2 + m1**2 - m2**2) / (2*m),

an expression whose double-precision evaluation carries a relative error of a
few units in the last place of ``E1**2`` (each operation contributes at most
2**-53 ~ 1.1e-16).  Exactly at threshold the true ``p**2`` is zero, so the
computed value may come out slightly negative purely from that cancellation.
Clamping a computed ``p**2`` to zero while it is above ``-1e-12 * E1**2``
absorbs about ten thousand ulp -- four orders of magnitude of headroom over the
achievable rounding error -- and nothing else.  A genuinely closed channel sits
far outside it: a 0.990 GeV parent into two 0.497611 GeV kaons gives
``p**2 = -2.5e-3 GeV**2`` against ``E1**2 = 0.245 GeV**2``, a relative deficit
of 1.0e-2, ten orders of magnitude beyond the tolerance.  Such a request
raises instead of returning particles.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import PDG

#: Largest relative negative ``p**2`` attributable to floating-point roundoff.
MOMENTUM_ROUNDOFF_REL = 1e-12

#: HNL rows carry current-quark masses rather than the PDG constituent values.
HNL_CURRENT_QUARK_MASSES = {1: .0047, 2: .0022, 3: .104, 4: 1.27, 5: 4.18}

#: PDG codes that reach the detector as hadrons rather than as themselves.
PARTON_PDGS = frozenset({1, 2, 3, 4, 5, 6, 21})

#: Electric charge of a quark, in thirds of the positron charge.
_QUARK_CHARGE3 = {1: -1, 2: 2, 3: -1, 4: 2, 5: -1, 6: 2}

#: The lightest meson carrying a given set of conserved charges, held as the
#: PDG code whose mass :mod:`funcs.PDG` keeps along with every other mass
#: EventCalc uses.  Keys are ``(strangeness, charm, beauty, electric charge in
#: thirds)``.
_LIGHTEST_MESON = {
    (0, 0, 0, 0): 111,    # pi0
    (0, 0, 0, 3): 211,    # pi+
    (1, 0, 0, 0): 311,    # K0
    (1, 0, 0, 3): 321,    # K+
    (0, 1, 0, 0): 421,    # D0
    (0, 1, 0, 3): 411,    # D+
    (1, 1, 0, 3): 431,    # Ds+
    (0, 0, 1, 0): 511,    # B0
    (0, 0, 1, 3): 521,    # B+
    (-1, 0, 1, 0): 531,   # Bs0
    (0, 1, 1, 3): 541,    # Bc+
}


def _meson_masses():
    """Lightest meson mass for each set of charges and for its mirror.

    A meson and its antiparticle share a mass and carry opposite charges, so
    each entry is read once and stored under both signs.
    """
    masses = {}
    for charges, code in _LIGHTEST_MESON.items():
        mass = float(PDG.get_mass(code))
        masses[charges] = mass
        masses[tuple(-value for value in charges)] = mass
    return masses


_MESON_MASSES = _meson_masses()

_PAD = -999


def daughter_masses(pdg_list, *, hnl=False):
    """Rest masses of a row's daughters, in the row's own convention."""
    masses = []
    for pdg in pdg_list:
        code = int(pdg)
        value = PDG.get_mass(code)
        if hnl and abs(code) in HNL_CURRENT_QUARK_MASSES:
            value = HNL_CURRENT_QUARK_MASSES[abs(code)]
        masses.append(float(value))
    return masses


@lru_cache(maxsize=256)
def _two_hadron_mass(charges):
    """Smallest mass of a two-meson state carrying ``charges``.

    The two mesons share the conserved charges between them in every way the
    table allows, and the lightest such pair the current can populate is the
    answer.  ``None`` says the table holds no pair that reaches these charges.
    """
    lightest = None
    for first, first_mass in _MESON_MASSES.items():
        rest = tuple(total - part for total, part in zip(charges, first))
        # Two identical pseudoscalars are not a state a vector or axial
        # current can populate: such a current needs them in L = 1, and Bose
        # symmetry forbids two identical bosons in an odd orbital state.  The
        # split that hands both mesons the same charges is that pair, so a
        # flavourless neutral system reaches pi+ pi- and not pi0 pi0.
        if rest == first:
            continue
        second_mass = _MESON_MASSES.get(rest)
        if second_mass is None:
            continue
        pair = first_mass + second_mass
        if lightest is None or pair < lightest:
            lightest = pair
    return lightest


@lru_cache(maxsize=256)
def _parton_system_threshold(codes):
    """Smallest invariant mass a row's partons can hadronise into.

    The top quark decays before it binds, so it contributes its electric
    charge and no flavour.
    """
    charge3 = 0
    strangeness = charm = beauty = 0
    quark_number = 0
    for code in codes:
        flavour = abs(code)
        if flavour == 21:
            continue
        sign = 1 if code > 0 else -1
        charge3 += sign * _QUARK_CHARGE3[flavour]
        quark_number += sign
        if flavour == 3:
            strangeness -= sign
        elif flavour == 4:
            charm += sign
        elif flavour == 5:
            beauty -= sign
    if quark_number != 0:
        raise ValueError(
            "Partons %r carry baryon number %+.3f: this module bounds mesonic "
            "systems, and a row with baryons needs its own lightest state."
            % (list(codes), quark_number / 3.))
    # A string breaks into quark-antiquark pairs and every break makes one
    # more hadron, so the final state holds two hadrons or more; one hadron
    # alone has zero phase-space measure and string fragmentation makes none.
    bound = _two_hadron_mass((strangeness, charm, beauty, charge3))
    if bound is None:
        raise ValueError(
            "Partons %r carry strangeness %+d, charm %+d, beauty %+d and "
            "charge %+.0f, which no pair of the mesons in this module holds; "
            "give this combination its own lightest two-hadron state."
            % (list(codes), strangeness, charm, beauty, charge3 / 3.))
    return bound


def _parton_flavour(code):
    """Strangeness, charm, beauty and charge in thirds that one parton carries.

    A gluon carries none of them.  The top quark decays before it binds, so
    the flavour it would carry never reaches a hadron.
    """
    flavour = abs(code)
    if flavour == 21:
        return (0, 0, 0, 0)
    sign = 1 if code > 0 else -1
    return (-sign if flavour == 3 else 0,
            sign if flavour == 4 else 0,
            -sign if flavour == 5 else 0,
            sign * _QUARK_CHARGE3[flavour])


@lru_cache(maxsize=256)
def _string_end_two_hadron_mass(codes):
    """Lightest pair with each parton's flavour kept on its own side."""
    left_charges, right_charges = (_parton_flavour(code) for code in codes)
    total_charge3 = left_charges[3] + right_charges[3]
    lightest = None
    for charge3 in (-3, 0, 3):
        left = _MESON_MASSES.get(left_charges[:3] + (charge3,))
        right = _MESON_MASSES.get(right_charges[:3] + (total_charge3 - charge3,))
        if left is None or right is None:
            continue
        pair = left + right
        if lightest is None or pair < lightest:
            lightest = pair
    return lightest


def string_end_two_hadron_mass(pdg_pair):
    """Lightest two-hadron state a string with these partons at its ends makes.

    The string breaks through the creation of light quark-antiquark pairs, so
    each original parton stays in the hadron its own end becomes and its
    flavour cannot turn up in the other hadron.  That is what a hadronizer
    builds from the pair, and it is at or above
    :func:`parton_system_threshold`, which bounds the same system without
    holding each flavour to its own side: ``s sbar`` reaches ``K+ K-`` here and
    ``pi+ pi-`` there.  ``None`` says the table holds no such pair.
    """
    codes = tuple(int(pdg) for pdg in pdg_pair)
    if len(codes) != 2:
        raise ValueError(
            "A string has two ends; %r partons were given." % (len(codes),))
    return _string_end_two_hadron_mass(codes)


def parton_system_threshold(pdg_list):
    """Smallest invariant mass the partons of ``pdg_list`` can hadronise into.

    This is the one statement of the partonic bound; the samplers read it from
    here so that a row's gate and the phase space it is sampled over agree.
    """
    return _parton_system_threshold(tuple(int(pdg) for pdg in pdg_list))


@lru_cache(maxsize=4096)
def _row_threshold(codes):
    """The threshold of one row, held for the rows a run keeps asking about."""
    partons = tuple(code for code in codes if abs(code) in PARTON_PDGS)
    others = [code for code in codes if abs(code) not in PARTON_PDGS]
    total = sum(float(PDG.get_mass(code)) for code in others)
    if partons:
        total += _parton_system_threshold(partons)
    return total


def channel_threshold(pdg_list, *, hnl=False):
    """Minimum parent mass at which this row is open.

    For a row of ordinary particles this is the sum of the daughter rest
    masses.  A row's partons contribute the lightest two-hadron state they can
    form together, so the quark-mass convention a row is written in leaves a
    threshold unchanged and ``hnl`` is carried only to match
    :func:`daughter_masses`.
    """
    return _row_threshold(tuple(int(pdg) for pdg in pdg_list))


def row_thresholds(pdg_table, *, hnl=False):
    """Threshold of every row of a decay table (padding entries dropped)."""
    out = np.empty(len(pdg_table), dtype=float)
    for index, row in enumerate(pdg_table):
        codes = tuple(int(p) for p in np.asarray(row).ravel() if int(p) != _PAD)
        out[index] = _row_threshold(codes)
    return out


def open_mask(mass, pdg_table, *, hnl=False):
    """Boolean mask of the rows a parent of ``mass`` can reach.

    The comparison is strict: a row is closed *at* its threshold as well as
    below it, matching exHad's convention of a rate that is exactly zero there.
    A parent exactly at threshold produces daughters at rest, which carries no
    phase space and no rate.
    """
    return np.asarray(mass, dtype=float) > row_thresholds(pdg_table, hnl=hnl)


def gate_rates(mass, pdg_table, rates, *, hnl=False):
    """The tabulated rates with every closed row set to exactly zero.

    Returns ``(gated_rates, closed_indices)``.  EventCalc keeps ownership of
    the rates: this only removes rows an interpolant opened below their own
    threshold, it never rescales or redistributes the surviving ones.
    """
    rates = np.asarray(rates, dtype=float).copy()
    closed = ~open_mask(mass, pdg_table, hnl=hnl)
    leaked = np.flatnonzero(closed & (rates > 0.))
    rates[closed] = 0.
    return rates, leaked


def require_open(mass, masses, *, label=''):
    """Raise unless a parent of ``mass`` can produce daughters of ``masses``.

    The samplers call this before drawing anything, so a closed request can
    never turn into particles.  It is deliberately not the place to repair a
    rate: an unreachable row must already have been gated out by
    :func:`gate_rates`, and reaching here means the caller bypassed that.
    """
    total = float(np.sum(masses))
    if not float(mass) > total:
        raise ValueError(
            "Decay channel %sis closed at m = %.6f GeV: the daughters need "
            "%.6f GeV. Threshold-aware rates must zero this row before "
            "sampling (funcs.thresholds.gate_rates); no particles are produced."
            % (('%r ' % (label,)) if label else '', float(mass), total))


def safe_momentum_squared(p_squared, scale):
    """Clamp only floating-point roundoff; anything larger is a defect.

    ``scale`` is the positive quantity ``p_squared`` was obtained from by
    cancellation (``E**2`` for the two-body form).  See the module docstring
    for the size of the tolerance and why it cannot hide a closed channel.
    """
    p_squared = np.asarray(p_squared, dtype=float)
    scale = np.asarray(scale, dtype=float)
    floor = -MOMENTUM_ROUNDOFF_REL * np.maximum(np.abs(scale), 1.)
    if np.any(p_squared < floor):
        worst = float(np.min(p_squared))
        raise ValueError(
            "Negative squared momentum %.6e beyond the %.0e relative roundoff "
            "tolerance: this is a closed decay channel, not rounding. Gate the "
            "row with funcs.thresholds.gate_rates instead of clamping."
            % (worst, MOMENTUM_ROUNDOFF_REL))
    return np.maximum(p_squared, 0.)
