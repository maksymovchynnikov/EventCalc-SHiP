"""Boundary-aware branching-ratio interpolation for the fermionic ALP.

The decay table deliberately changes ownership at two discrete masses:
exclusive *meson* rows -> light partonic rows at ``m_start`` and light ->
charm-aware partonic rows at ``m_charm``.  The controlled exact-nucleon rows
``ppbar`` and ``nnbar`` remain explicit EventCalc decays across both seams.
A single linear interpolator across either partonic seam would mix the meson
and parton descriptions in the preceding 1-MeV bin.
"""

import bisect
import json
import math
import os

from . import PDG


NONHADRONIC = {"ePeM", "muPmuM", "tauPtauM", "2gamma"}
LIGHT_PARTONIC = {"Jets-GG", "Jets-ss"}
CHARM_PARTONIC = {"Jets-cc"}
PARTONIC = LIGHT_PARTONIC | CHARM_PARTONIC
# Exact two-body nucleon modes are separate physical partial widths in the
# authoritative ALP table.  They are not part of the inclusive ``Jets-*``
# remainder and must therefore survive the meson-to-parton handoff.
PERSISTENT_EXCLUSIVE = {"ppbar", "nnbar"}
KINEMATIC_THRESHOLD_TOL = 1.0e-12
SOURCE_CLOSURE_TOLERANCE = 1.0e-12


def prepare_decay_rows(rows):
    """Validate the canonical ALP decay records for EventCalc.

    The authoritative card uses variable-length numeric PDG lists: it has no
    padding sentinels, the gluon has PDG id +21 (never -21), and every matrix
    element is already an explicit Python expression. Reject malformed source
    data rather than silently repairing it at runtime.
    """
    labels, pdgs, br_tables, matrix_elements = [], [], [], []
    for i, row in enumerate(rows):
        if not isinstance(row, list) or len(row) < 4:
            raise ValueError("invalid ALP decay row %d" % i)
        label = row[0]
        normalized_pdgs = []
        for raw_pdg in row[1]:
            pdg = int(raw_pdg)
            if float(raw_pdg) != pdg:
                raise ValueError("non-integral PDG id in ALP row %r" % label)
            if pdg == -21:
                raise ValueError(
                    "invalid gluon PDG id -21 in ALP row %r; use +21" % label)
            if pdg == 0 or abs(pdg) >= 999990:
                raise ValueError(
                    "invalid PDG padding/sentinel in ALP row %r" % label)
            normalized_pdgs.append(pdg)

        matrix_element = row[3]
        is_dispatcher = (isinstance(matrix_element, str) and
                         matrix_element.strip().startswith(
                             "Msquared3BodyLLP("))
        if is_dispatcher:
            raise ValueError(
                "unsupported matrix-element dispatcher in ALP row %r; "
                "export an explicit expression" % label)

        labels.append(label)
        pdgs.append(normalized_pdgs)
        br_tables.append(row[2])
        matrix_elements.append(matrix_element)
    return labels, pdgs, br_tables, matrix_elements


def load_exhad_boundaries(particle_path):
    """Return the card-owned partonic and open-charm ALP boundaries."""
    path = os.path.join(particle_path, "exhad.json")
    with open(path) as f:
        card = json.load(f)
    if card.get("bench") != "alp":
        raise RuntimeError("fermionic-ALP exhad card must use bench 'alp'")
    m_start = float(card["m_start"])
    m_edge = float(card.get("m_edge", m_start))
    m_charm = float(card["m_charm"])
    m_high = float(card["m_high"])
    if abs(m_edge - m_start) > 1e-12:
        raise RuntimeError(
            "fermionic ALP has no residual hybrid interval: "
            "m_edge must equal m_start")
    if not m_start < m_charm <= m_high:
        raise RuntimeError("invalid fermionic-ALP exhad mass window")
    return m_start, m_charm


def _interp(points, mass):
    """Linear interpolation inside a segment, one-sided constant at ends."""
    if not points:
        return 0.0
    if mass <= points[0][0]:
        return points[0][1]
    if mass >= points[-1][0]:
        return points[-1][1]
    masses = [p[0] for p in points]
    i = bisect.bisect_left(masses, mass)
    m0, y0 = points[i - 1]
    m1, y1 = points[i]
    x = (mass - m0) / (m1 - m0)
    return y0 + x * (y1 - y0)


def _one_sided_slope(points, mass, side):
    """Return the exact tangent of a piecewise-linear table at one node."""

    masses = [point[0] for point in points]
    index = bisect.bisect_left(masses, mass)
    if index >= len(points) or points[index][0] != mass:
        raise ValueError("ALP bridge endpoint is absent from a source table")
    if side == "left":
        if index == 0:
            raise ValueError("ALP bridge lacks a left source tangent")
        lower, upper = points[index - 1], points[index]
    elif side == "right":
        if index + 1 >= len(points):
            raise ValueError("ALP bridge lacks a right source tangent")
        lower, upper = points[index], points[index + 1]
    else:
        raise ValueError("ALP source-tangent side must be left or right")
    return (upper[1] - lower[1]) / (upper[0] - lower[0])


def _reduced_width_endpoint(probabilities, ctaus, mass, side):
    """Return ``BR/ctau`` and its exact one-sided table-interpolant slope."""

    probability = _interp(probabilities, mass)
    ctau = _interp(ctaus, mass)
    probability_slope = _one_sided_slope(probabilities, mass, side)
    ctau_slope = _one_sided_slope(ctaus, mass, side)
    reduced_width = probability / ctau
    slope = (
        probability_slope * ctau - probability * ctau_slope
    ) / (ctau * ctau)
    return reduced_width, slope


def _monotone_tangents(left, right, length, left_slope, right_slope):
    """Apply the parameter-free Fritsch--Carlson monotonicity bound."""

    secant = (right - left) / length
    if secant == 0.0:
        return 0.0, 0.0
    if left_slope * secant <= 0.0:
        left_slope = 0.0
    if right_slope * secant <= 0.0:
        right_slope = 0.0
    alpha = left_slope / secant
    beta = right_slope / secant
    norm = math.hypot(alpha, beta)
    if norm > 3.0:
        scale = 3.0 / norm
        left_slope = scale * alpha * secant
        right_slope = scale * beta * secant
    return left_slope, right_slope


def _hermite_value(left, right, left_slope, right_slope, start, end, mass):
    """Evaluate one endpoint-value-and-slope cubic Hermite segment."""

    if mass <= start:
        return left
    if mass >= end:
        return right
    length = end - start
    coordinate = (mass - start) / length
    coordinate2 = coordinate * coordinate
    coordinate3 = coordinate2 * coordinate
    return (
        (2.0 * coordinate3 - 3.0 * coordinate2 + 1.0) * left
        + (coordinate3 - 2.0 * coordinate2 + coordinate)
        * length * left_slope
        + (-2.0 * coordinate3 + 3.0 * coordinate2) * right
        + (coordinate3 - coordinate2) * length * right_slope
    )


def _force_exact_sum(values, target, correction_index):
    """Move one positive entry by ulps until ``math.fsum`` is exact."""

    for _ in range(64):
        observed = math.fsum(values)
        if observed == target:
            return
        direction = math.inf if observed < target else -math.inf
        corrected = math.nextafter(values[correction_index], direction)
        if corrected < 0.0 or corrected == values[correction_index]:
            break
        values[correction_index] = corrected
    raise RuntimeError("failed to enforce exact ALP probability closure")


def _closed_probabilities(values):
    """Return the same exactly closed floating vector as the rate authority."""

    probabilities = [float(value) for value in values]
    if not probabilities or any(
            not math.isfinite(value) or value < 0.0
            for value in probabilities):
        raise ValueError("ALP branching vector contains an invalid probability")
    total = math.fsum(probabilities)
    if abs(total - 1.0) > SOURCE_CLOSURE_TOLERANCE:
        raise ValueError("ALP branching vector does not close: sum=%r" % total)
    correction_index = max(
        range(len(probabilities)), key=probabilities.__getitem__)
    probabilities[correction_index] = 1.0 - math.fsum(
        value for index, value in enumerate(probabilities)
        if index != correction_index)
    _force_exact_sum(probabilities, 1.0, correction_index)
    return probabilities


def _persistent_width_bridges(
        labels, series, ctaus, bridge_start, bridge_end):
    """Build the physical nucleon-width bridge from supplied endpoint data."""

    length = bridge_end - bridge_start
    bridges = {}
    for label in sorted(PERSISTENT_EXCLUSIVE):
        probabilities = series[labels.index(label)]
        left, left_slope = _reduced_width_endpoint(
            probabilities, ctaus, bridge_start, "left")
        right, right_slope = _reduced_width_endpoint(
            probabilities, ctaus, bridge_end, "right")
        left_slope, right_slope = _monotone_tangents(
            left, right, length, left_slope, right_slope)
        bridges[label] = (
            left, right, left_slope, right_slope)
    return bridges


def _apply_persistent_width_bridge(
        probabilities, labels, ctaus, bridges,
        bridge_start, bridge_end, mass):
    """Bridge exact nucleon widths while retaining the hadronic total."""

    if not bridge_start < mass < bridge_end:
        return probabilities
    result = list(probabilities)
    hadronic_indices = [
        index for index, label in enumerate(labels)
        if label not in NONHADRONIC
    ]
    persistent_indices = [
        labels.index(label) for label in sorted(PERSISTENT_EXCLUSIVE)
    ]
    active_indices = [
        index for index in hadronic_indices
        if index not in persistent_indices
    ]
    target_hadronic = math.fsum(
        result[index] for index in hadronic_indices)
    ctau = _interp(ctaus, mass)
    for label, index in zip(
            sorted(PERSISTENT_EXCLUSIVE), persistent_indices):
        left, right, left_slope, right_slope = bridges[label]
        reduced_width = _hermite_value(
            left, right, left_slope, right_slope,
            bridge_start, bridge_end, mass)
        if reduced_width < 0.0:
            raise ValueError("ALP nucleon-width bridge became negative")
        result[index] = reduced_width * ctau
    target_active = target_hadronic - math.fsum(
        result[index] for index in persistent_indices)
    source_active = math.fsum(
        result[index] for index in active_indices)
    if target_active < 0.0 or source_active <= 0.0:
        raise ValueError("ALP nucleon bridge has no positive complement")
    scale = target_active / source_active
    active_values = [result[index] * scale for index in active_indices]
    correction_index = max(
        range(len(active_values)), key=active_values.__getitem__)
    active_values[correction_index] = target_active - math.fsum(
        value for index, value in enumerate(active_values)
        if index != correction_index)
    for index, value in zip(active_indices, active_values):
        result[index] = value
    return _closed_probabilities(result)


def build_br_interpolator(
        labels, pdgs, br_tables, m_start, m_charm, ctau_table):
    """Build an ALP BR interpolator with exact description ownership.

    Complete exclusive and light-partonic vectors remain disjoint at
    ``m_start``.  Across the missing source interval, the exact ``ppbar`` and
    ``nnbar`` partial widths follow the unique monotone cubic Hermite segment
    fixed by their supplied endpoint values and one-sided source tangents.
    The active hadronic complement is rescaled proportionally, preserving its
    internal composition and the supplied total hadronic width.  GG/ss use
    separate pre-charm and charm-aware segments. The charm row is exactly zero
    below ``m_charm`` and turns on at that boundary.
    """
    # ``initLLP`` owns these catalogs as NumPy object arrays.  Normalize them
    # once so the interpolation core has ordinary sequence semantics (notably
    # deterministic ``.index`` lookup) in production and standalone tests.
    labels = [str(label) for label in labels]
    pdgs = tuple(pdgs)
    br_tables = tuple(br_tables)
    if len(labels) != len(pdgs) or len(labels) != len(br_tables):
        raise ValueError("ALP labels/PDGs/BR-table length mismatch")

    thresholds = []
    for label, ids in zip(labels, pdgs):
        daughter_masses = [PDG.get_mass(int(pdg)) for pdg in ids]
        if any(not isinstance(mass, (int, float))
               for mass in daughter_masses):
            raise ValueError("unknown EventCalc PDG mass in ALP row %r" % label)
        thresholds.append(sum(float(mass) for mass in daughter_masses))

    series = []
    global_min = float("inf")
    global_max = -float("inf")
    # Mathematica-exported decimal grid points can arrive as
    # 3.7409999999999997. Canonicalize them before ownership comparisons so
    # the literal card boundary 3.741 selects the intended row exactly.
    m_start = round(float(m_start), 12)
    m_charm = round(float(m_charm), 12)
    for table in br_tables:
        points = [(round(float(m), 12), float(v)) for m, v in table]
        if len(points) < 2 or any(points[i][0] >= points[i + 1][0]
                                  for i in range(len(points) - 1)):
            raise ValueError("ALP BR mass grids must be strictly increasing")
        series.append(points)
        global_min = min(global_min, points[0][0])
        global_max = max(global_max, points[-1][0])
    ctaus = [
        (round(float(mass), 12), float(value))
        for mass, value in ctau_table
    ]
    if (
            len(ctaus) < 2
            or any(
                not math.isfinite(mass) or not math.isfinite(value)
                or mass <= 0.0 or value <= 0.0
                for mass, value in ctaus)
            or any(
                ctaus[index][0] >= ctaus[index + 1][0]
                for index in range(len(ctaus) - 1))):
        raise ValueError("ALP lifetime grid is malformed")

    missing_persistent = PERSISTENT_EXCLUSIVE - set(labels)
    if missing_persistent:
        raise ValueError(
            "ALP decay table lacks persistent exact-nucleon rows: %s" %
            ", ".join(sorted(missing_persistent)))
    exclusive = (set(labels) - NONHADRONIC - PARTONIC -
                 PERSISTENT_EXCLUSIVE)
    segments = []
    for label, points in zip(labels, series):
        if label in NONHADRONIC or label in PERSISTENT_EXCLUSIVE:
            segments.append(("continuous", points))
        elif label in exclusive:
            segments.append(("exclusive",
                             [(m, v) for m, v in points if m < m_start]))
        elif label in LIGHT_PARTONIC:
            middle = [(m, v) for m, v in points
                      if m_start <= m < m_charm]
            high = [(m, v) for m, v in points if m >= m_charm]
            segments.append(("light", (middle, high)))
        elif label in CHARM_PARTONIC:
            segments.append(("charm",
                             [(m, v) for m, v in points if m >= m_charm]))
        else:  # defensive: the sets above are exhaustive
            raise ValueError("unclassified ALP channel %r" % label)

    first_light_nodes = {
        payload[0][0][0]
        for kind, payload in segments
        if kind == "light" and payload[0]
    }
    if len(first_light_nodes) != 1:
        raise ValueError(
            "ALP light-partonic rows must share one first source node")
    first_light_node = first_light_nodes.pop()
    if first_light_node < m_start:
        raise ValueError(
            "ALP first light-partonic source node precedes its boundary")
    last_exclusive_nodes = {
        payload[-1][0]
        for kind, payload in segments
        if kind == "exclusive" and payload
    }
    if len(last_exclusive_nodes) != 1:
        raise ValueError(
            "ALP exclusive rows must share one last source node")
    last_exclusive_node = last_exclusive_nodes.pop()
    if not last_exclusive_node < m_start <= first_light_node:
        raise ValueError("invalid ALP meson-to-parton source gap")
    persistent_bridges = _persistent_width_bridges(
        labels, series, ctaus, last_exclusive_node, first_light_node)

    def get_br(mass):
        mass = round(float(mass), 12)
        if mass < global_min or mass > global_max:
            # An all-zero branching vector says that every decay channel is
            # closed at this mass.  A mass off the end of the source table
            # carries no such information about the ALP, so the request is
            # refused with the mass range the table covers.
            raise ValueError(
                "ALP mass %.12g GeV lies outside the tabulated branching-ratio "
                "range [%.12g, %.12g] GeV" % (mass, global_min, global_max))
        # The authoritative source table deliberately has no nodes from
        # m_start through the point immediately preceding first_light_node.
        # Interpolating rows independently across this gap creates a hybrid
        # meson/parton vector.  Hold one complete non-persistent vector on
        # each side; the physical nucleon-width bridge below then replaces
        # only ppbar/nnbar and rescales that vector's hadronic complement.
        if last_exclusive_node < mass < m_start:
            evaluation_mass = last_exclusive_node
        elif m_start <= mass < first_light_node:
            evaluation_mass = first_light_node
        else:
            evaluation_mass = mass
        out = []
        threshold_changed = False
        for threshold, (kind, payload) in zip(thresholds, segments):
            if kind == "continuous":
                value = _interp(payload, evaluation_mass)
            elif kind == "exclusive":
                value = (
                    _interp(payload, evaluation_mass)
                    if evaluation_mass < m_start
                    else 0.0
                )
            elif kind == "light":
                middle, high = payload
                if evaluation_mass < m_start:
                    value = 0.0
                elif evaluation_mass < m_charm:
                    value = _interp(middle, evaluation_mass)
                else:
                    value = _interp(high, evaluation_mass)
            else:  # charm
                value = (
                    0.0
                    if evaluation_mass < m_charm
                    else _interp(payload, evaluation_mass)
                )
            # The source grid is spaced by 1 MeV.  Even when every closed grid
            # value is zero, linear interpolation would otherwise create a
            # positive ramp between the last closed point and the first open
            # one.  Apply the exact runtime-mass threshold independently of
            # source-card hygiene.
            if (evaluation_mass + KINEMATIC_THRESHOLD_TOL < threshold and
                    float(value) != 0.0):
                value = 0.0
                threshold_changed = True
            out.append(max(float(value), 0.0))
        out = _apply_persistent_width_bridge(
            out,
            labels,
            ctaus,
            persistent_bridges,
            last_exclusive_node,
            first_light_node,
            mass,
        )
        # Removing an interpolated, kinematically forbidden contribution must
        # not create an artificial deficit in the total width or in EventCalc's
        # selected-channel normalization.
        if threshold_changed:
            total = sum(out)
            if not total > 0.0:
                raise ValueError(
                    "all ALP branching fractions vanish below a threshold")
            out = [value / total for value in out]
        return out

    return get_br
