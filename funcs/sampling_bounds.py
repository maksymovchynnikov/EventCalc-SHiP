"""Upper bounds, derived by argument, for the EventCalc decay samplers.

Accept-reject draws a candidate from a region and keeps it with probability
``weight(candidate) / ceiling``.  The accepted candidates follow the
distribution proportional to ``weight`` when ``ceiling`` is at least as large as
``weight`` everywhere the candidate can land.  A ceiling read off a finite trial
sample carries no such guarantee: a narrow peak the trial sample never visited
stays invisible, and every candidate that lands on it is kept with probability
one instead of the correct fraction, which deforms the distribution exactly
where it is largest.  Every ceiling here is derived from the form of the weight
before any event is drawn, so the keep probability stays at or below one and the
accepted sample is exact.

Intermediate masses of a four-or-more-body decay
    ``ChainMassSampler`` cuts the range of each intermediate invariant mass of
    the decay chain into cells and carries, for each combination of cells, a
    value the phase-space density provably never exceeds inside it.  That
    piecewise-constant function is what candidates are drawn from: its mass and
    its conditional structure factorise along the chain, so it costs a table per
    step to build and one search per step to draw from, however many daughters
    there are.  Refining the cells drives it down towards the density itself,
    which is what keeps the acceptance rate high.  ``four_body_weight_bound``
    and ``chain_weight_bound`` read the largest value of that function, which is
    an upper bound on the weight over the whole region.

Three-body decays with a tabulated matrix element
    The weight is an explicit algebraic expression in the energies of two of the
    three daughters.  ``dalitz_upper_bound`` cuts the energy plane into cells
    and works out, for each cell, a value the expression cannot exceed anywhere
    inside it; candidates are drawn from that piecewise-constant function by
    exact inversion of its cumulative and kept with probability weight over the
    value their own cell carries.  The cells start as a regular grid and are
    cut further where the area under the cover stands furthest above the weight
    itself, which is where the candidates are being wasted: either because the
    cell reaches outside the kinematically allowed region, where the weight is
    zero, or because the value it carries stands above the weight inside it.
    The result covers the allowed region with cells that do not overlap, each
    carrying its own such value.  The largest of them is a ceiling for the whole
    region; taken together they form a staircase that lies above the matrix
    element everywhere, which keeps the acceptance rate usable for matrix
    elements whose peak stands orders of magnitude above their mean.

How a value the expression cannot exceed is worked out
    The tabulated matrix elements are long expanded polynomials divided by
    products of meson propagators.  Replacing each energy by the range it runs
    over and carrying ranges through such an expression treats every occurrence
    of an energy as a quantity of its own, so the large cancellations between
    the terms are lost: measured on the shipped dark photon three-pion element,
    plain ranges overstate the maximum by thirteen orders of magnitude and the
    propagator denominators come out straddling zero even though they are sums
    of squares that never vanish.

    Each quantity is therefore carried as a first-order model in the two
    energies: a centre, a slope in each energy, and a radius covering
    everything the linear part does not describe.  Sums and products of such
    models keep the linear dependence on the energies exact and push only the
    genuinely quadratic residue into the radius, so the bound is looser than
    the true range by an amount that falls with the square of the rectangle
    size instead of with its size, and correlated terms cancel as they should.

    The models are complex: several tabulated matrix elements carry meson
    propagators written with a complex pole ``m^2 - i m Gamma``, and the real
    part of the result is what the sampler weights with.  Every operation
    widens its radius by a few units in the last place, so the bound stays
    valid under floating-point rounding.
"""

import functools

import numpy as np

# One double-precision rounding is at most 2^-53 in relative terms, and each
# elementary operation performs a handful of them.
_ROUNDING_SLACK = 16.0 * 2.0 ** -53


class FirstOrderModel:
    """A first-order model of a complex quantity over a rectangle.

    The two independent variables of the rectangle are written as
    ``e1, e3 in [-1, 1]``, one per energy.  The quantity is enclosed by

        centre + slope1 * e1 + slope3 * e3 + [-radius, radius] * (unit disc),

    with a separate centre and pair of slopes for the real and the imaginary
    part and a shared non-negative radius for each part.  Every attribute is an
    array over a batch of rectangles, so a whole generation of rectangles is
    enclosed in one pass.
    """

    __slots__ = ("real", "imag")

    def __init__(self, real, imag=None):
        self.real = real
        self.imag = imag if imag is not None else _zero_like(real)

    # -- construction ----------------------------------------------------
    @staticmethod
    def variable(low, high, index):
        """The model of one energy, exact over the rectangle."""
        low = np.asarray(low, dtype=float)
        high = np.asarray(high, dtype=float)
        centre = 0.5 * (low + high)
        half_width = 0.5 * (high - low)
        zero = np.zeros_like(centre)
        slope1 = half_width if index == 1 else zero
        slope3 = half_width if index == 3 else zero
        return FirstOrderModel(_Part(centre, slope1, slope3, np.zeros_like(centre)))

    @staticmethod
    def enclose(value):
        if isinstance(value, FirstOrderModel):
            return value
        array = np.asarray(value)
        if np.iscomplexobj(array):
            return FirstOrderModel(_constant(array.real.astype(float)),
                          _constant(array.imag.astype(float)))
        return FirstOrderModel(_constant(array.astype(float)))

    def broadcast(self, shape):
        return FirstOrderModel(self.real.broadcast(shape), self.imag.broadcast(shape))

    # -- ranges ----------------------------------------------------------
    @property
    def real_high(self):
        return self.real.centre + self.real.spread()

    @property
    def real_low(self):
        return self.real.centre - self.real.spread()

    @property
    def is_real(self):
        return self.imag.is_zero()

    def __repr__(self):
        return "FirstOrderModel(real in [%g, %g])" % (np.min(self.real_low),
                                             np.max(self.real_high))

    # -- arithmetic ------------------------------------------------------
    def __add__(self, other):
        other = FirstOrderModel.enclose(other)
        return FirstOrderModel(self.real.add(other.real), self.imag.add(other.imag))

    __radd__ = __add__

    def __neg__(self):
        return FirstOrderModel(self.real.negated(), self.imag.negated())

    def __pos__(self):
        return self

    def __sub__(self, other):
        return self + (-FirstOrderModel.enclose(other))

    def __rsub__(self, other):
        return FirstOrderModel.enclose(other) + (-self)

    def __mul__(self, other):
        other = FirstOrderModel.enclose(other)
        if self.is_real and other.is_real:
            return FirstOrderModel(self.real.multiply(other.real))
        real = self.real.multiply(other.real).add(
            self.imag.multiply(other.imag).negated())
        imag = self.real.multiply(other.imag).add(self.imag.multiply(other.real))
        return FirstOrderModel(real, imag)

    __rmul__ = __mul__

    def reciprocal(self):
        """Enclose ``1 / z`` as ``conj(z) / (Re z ^ 2 + Im z ^ 2)``.

        A real quantity is inverted directly.  Routing it through the squared
        modulus would square the radius as well, and the expansion behind
        ``_Part.reciprocal`` then needs a rectangle four times smaller before it
        applies.
        """
        if self.is_real:
            return FirstOrderModel(self.real.reciprocal())
        modulus = self.real.square().add(self.imag.square())
        inverse = modulus.reciprocal()
        conjugate = FirstOrderModel(self.real, self.imag.negated())
        return conjugate * FirstOrderModel(inverse)

    def __truediv__(self, other):
        return self * FirstOrderModel.enclose(other).reciprocal()

    def __rtruediv__(self, other):
        return FirstOrderModel.enclose(other) * self.reciprocal()

    def __pow__(self, exponent):
        if isinstance(exponent, FirstOrderModel):
            raise TypeError("an exponent that depends on the energies is not supported")
        power = int(exponent)
        if power != exponent:
            raise TypeError("a non-integer exponent %r is not supported" % (exponent,))
        if power < 0:
            return (self ** (-power)).reciprocal()
        if power == 0:
            return FirstOrderModel.enclose(np.ones_like(self.real.centre))
        if power == 1:
            return self
        if power == 2:
            if self.is_real:
                return FirstOrderModel(self.real.square())
            return self * self
        half = self ** (power // 2)
        squared = half * half
        return squared if power % 2 == 0 else squared * self

    def sqrt(self):
        if not self.is_real:
            raise TypeError("the square root of a complex quantity is not supported")
        return FirstOrderModel(self.real.sqrt())

    # -- comparison ------------------------------------------------------
    def _compare(self, other, strict, flip):
        other = FirstOrderModel.enclose(other)
        if not (self.is_real and other.is_real):
            raise TypeError("complex quantities cannot be ordered")
        left_low, left_high = self.real_low, self.real_high
        right_low, right_high = other.real_low, other.real_high
        if flip:
            left_low, left_high, right_low, right_high = (
                right_low, right_high, left_low, left_high)
        if strict:
            return Condition(left_high < right_low, left_low < right_high)
        return Condition(left_high <= right_low, left_low <= right_high)

    def __lt__(self, other):
        return self._compare(other, strict=True, flip=False)

    def __le__(self, other):
        return self._compare(other, strict=False, flip=False)

    def __gt__(self, other):
        return self._compare(other, strict=True, flip=True)

    def __ge__(self, other):
        return self._compare(other, strict=False, flip=True)

    def __eq__(self, other):
        other = FirstOrderModel.enclose(other)
        difference = self - other
        exact = (difference.real.is_point() & difference.imag.is_point() &
                 (difference.real.centre == 0.0) & (difference.imag.centre == 0.0))
        possible = ((difference.real_low <= 0.0) & (difference.real_high >= 0.0) &
                    (difference.imag.centre - difference.imag.spread() <= 0.0) &
                    (difference.imag.centre + difference.imag.spread() >= 0.0))
        return Condition(exact, possible)

    def __ne__(self, other):
        equality = self == other
        return Condition(~equality.possible, ~equality.certain)

    __hash__ = None

    # -- NumPy interoperation -------------------------------------------
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _dispatch_ufunc(ufunc, method, inputs, kwargs)

    def __array_function__(self, function, types, args, kwargs):
        return _dispatch_function(function, args, kwargs)


class _Part:
    """One real component of a first-order model."""

    __slots__ = ("centre", "slope1", "slope3", "radius")

    def __init__(self, centre, slope1, slope3, radius):
        self.centre = centre
        self.slope1 = slope1
        self.slope3 = slope3
        self.radius = radius

    def spread(self):
        """Largest departure from the centre."""
        return np.abs(self.slope1) + np.abs(self.slope3) + self.radius

    def magnitude(self):
        return np.abs(self.centre) + self.spread()

    def is_zero(self):
        return (not np.any(self.centre) and not np.any(self.slope1)
                and not np.any(self.slope3) and not np.any(self.radius))

    def is_point(self):
        return self.spread() == 0.0

    def broadcast(self, shape):
        return _Part(np.broadcast_to(self.centre, shape),
                     np.broadcast_to(self.slope1, shape),
                     np.broadcast_to(self.slope3, shape),
                     np.broadcast_to(self.radius, shape))

    def rounded(self):
        """Absorb the rounding error of the operation that just produced this."""
        slack = self.magnitude() * _ROUNDING_SLACK
        return _Part(self.centre, self.slope1, self.slope3, self.radius + slack)

    def negated(self):
        return _Part(-self.centre, -self.slope1, -self.slope3, self.radius)

    def add(self, other):
        return _Part(self.centre + other.centre,
                     self.slope1 + other.slope1,
                     self.slope3 + other.slope3,
                     self.radius + other.radius).rounded()

    def multiply(self, other):
        """Keep the linear dependence exact, bound the rest.

        With ``x = x0 + a1 e1 + a3 e3 + dx`` and ``y = y0 + b1 e1 + b3 e3 + dy``
        the product splits into a linear part and a residue.  The residue holds
        ``a1 b1 e1^2`` and ``a3 b3 e3^2``, whose squares run over ``[0, 1]`` and
        so are recentred by half, the cross term ``(a1 b3 + a3 b1) e1 e3`` which
        runs over ``[-1, 1]``, and everything the two radii reach.
        """
        square_terms = self.slope1 * other.slope1 + self.slope3 * other.slope3
        cross = self.slope1 * other.slope3 + self.slope3 * other.slope1
        residue = (0.5 * (np.abs(self.slope1 * other.slope1) +
                          np.abs(self.slope3 * other.slope3)) +
                   np.abs(cross) +
                   self.radius * other.magnitude() +
                   other.radius * (np.abs(self.centre) + np.abs(self.slope1) +
                                   np.abs(self.slope3)))
        return _Part(self.centre * other.centre + 0.5 * square_terms,
                     self.centre * other.slope1 + self.slope1 * other.centre,
                     self.centre * other.slope3 + self.slope3 * other.centre,
                     residue).rounded()

    def square(self):
        """The square of a real component, which is never negative.

        The residue ``(a1 e1 + a3 e3)^2`` runs over ``[0, s^2]`` with
        ``s = |a1| + |a3|``, so it is recentred at ``s^2 / 2`` with radius
        ``s^2 / 2``; the radius of the component enters through the same
        widening as in a general product.
        """
        linear_spread = np.abs(self.slope1) + np.abs(self.slope3)
        residue = (0.5 * linear_spread ** 2 +
                   self.radius * (2.0 * np.abs(self.centre) +
                                  2.0 * linear_spread + self.radius))
        return _Part(self.centre ** 2 + 0.5 * linear_spread ** 2,
                     2.0 * self.centre * self.slope1,
                     2.0 * self.centre * self.slope3,
                     residue).rounded()

    def reciprocal(self):
        """Enclose ``1 / x`` for a real component.

        Expanding about the centre, ``1 / (x0 + u) = 1 / x0 - u / x0^2 +
        u^2 / (x0^2 (x0 + u))``.  The expansion is used where the departure
        ``u`` cannot reach ``-x0``; elsewhere the component is replaced by its
        plain range, which is unbounded when that range contains zero.
        """
        centre = self.centre
        spread = self.spread()
        low = centre - spread
        high = centre + spread
        safe = spread < np.abs(centre)
        with np.errstate(divide="ignore", invalid="ignore"):
            centre_inverse = np.where(safe, 1.0 / np.where(safe, centre, 1.0), 0.0)
            scale = centre_inverse ** 2
            tail = np.where(safe,
                            spread ** 2 * scale /
                            np.maximum(np.abs(centre) - spread, np.finfo(float).tiny),
                            0.0)
            expansion = _Part(centre_inverse,
                              -self.slope1 * scale,
                              -self.slope3 * scale,
                              self.radius * scale + tail)
            crosses_zero = (low <= 0.0) & (high >= 0.0)
            range_low = np.where(crosses_zero, -np.inf, 1.0 / high)
            range_high = np.where(crosses_zero, np.inf, 1.0 / low)
        fallback = _from_range(range_low, range_high, np.zeros_like(centre))
        return _Part(np.where(safe, expansion.centre, fallback.centre),
                     np.where(safe, expansion.slope1, 0.0),
                     np.where(safe, expansion.slope3, 0.0),
                     np.where(safe, expansion.radius, fallback.radius)).rounded()

    def sqrt(self):
        """Enclose the square root of a non-negative real component.

        The error of the tangent line at the centre is concave and vanishes to
        first order there, so its largest magnitude over the range sits at one
        of the two ends.
        """
        centre = self.centre
        spread = self.spread()
        low = centre - spread
        high = centre + spread
        if np.any(low < 0.0):
            raise TypeError("the square root of a negative quantity is not supported")
        positive = centre > 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            root = np.sqrt(np.where(positive, centre, 1.0))
            derivative = np.where(positive, 0.5 / root, 0.0)
            error_low = np.abs(np.sqrt(low) - root - (low - centre) * derivative)
            error_high = np.abs(np.sqrt(high) - root - (high - centre) * derivative)
            tangent = _Part(np.where(positive, root, 0.0),
                            self.slope1 * derivative,
                            self.slope3 * derivative,
                            self.radius * derivative +
                            np.maximum(error_low, error_high))
        degenerate = _from_range(np.sqrt(low), np.sqrt(high), np.zeros_like(centre))
        return _Part(np.where(positive, tangent.centre, degenerate.centre),
                     np.where(positive, tangent.slope1, 0.0),
                     np.where(positive, tangent.slope3, 0.0),
                     np.where(positive, tangent.radius, degenerate.radius)).rounded()


def _zero_like(part):
    zero = np.zeros_like(part.centre)
    return _Part(zero, zero, zero, zero)


def _constant(value):
    zero = np.zeros_like(value)
    return _Part(value, zero, zero, zero)


def _from_range(low, high, template):
    centre = 0.5 * (low + high)
    radius = 0.5 * (high - low)
    centre = np.where(np.isfinite(centre), centre, 0.0)
    radius = np.where(np.isfinite(radius), radius, np.inf)
    zero = np.zeros_like(template)
    return _Part(centre + zero, zero, zero, radius + zero)


class Condition:
    """The truth value of a comparison between enclosed quantities.

    A comparison over a rectangle has three outcomes: it holds everywhere in
    the rectangle, it fails everywhere, or it holds in part of it.  ``certain``
    records the first, and ``possible`` the union of the first and the third.
    """

    __slots__ = ("certain", "possible")

    def __init__(self, certain, possible):
        self.certain = np.asarray(certain, dtype=bool)
        self.possible = np.asarray(possible, dtype=bool)

    @staticmethod
    def enclose(value):
        if isinstance(value, Condition):
            return value
        truth = np.asarray(value, dtype=bool)
        return Condition(truth, truth)

    def __invert__(self):
        return Condition(~self.possible, ~self.certain)

    def __and__(self, other):
        other = Condition.enclose(other)
        return Condition(self.certain & other.certain,
                         self.possible & other.possible)

    __rand__ = __and__

    def __or__(self, other):
        other = Condition.enclose(other)
        return Condition(self.certain | other.certain,
                         self.possible | other.possible)

    __ror__ = __or__

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        return _dispatch_ufunc(ufunc, method, inputs, kwargs)

    def __array_function__(self, function, types, args, kwargs):
        return _dispatch_function(function, args, kwargs)


_BINARY_UFUNCS = {
    "add": lambda a, b: FirstOrderModel.enclose(a) + FirstOrderModel.enclose(b),
    "subtract": lambda a, b: FirstOrderModel.enclose(a) - FirstOrderModel.enclose(b),
    "multiply": lambda a, b: FirstOrderModel.enclose(a) * FirstOrderModel.enclose(b),
    "true_divide": lambda a, b: FirstOrderModel.enclose(a) / FirstOrderModel.enclose(b),
    "divide": lambda a, b: FirstOrderModel.enclose(a) / FirstOrderModel.enclose(b),
    "power": lambda a, b: FirstOrderModel.enclose(a) ** b,
    "less": lambda a, b: FirstOrderModel.enclose(a) < FirstOrderModel.enclose(b),
    "less_equal": lambda a, b: FirstOrderModel.enclose(a) <= FirstOrderModel.enclose(b),
    "greater": lambda a, b: FirstOrderModel.enclose(a) > FirstOrderModel.enclose(b),
    "greater_equal": lambda a, b: FirstOrderModel.enclose(a) >= FirstOrderModel.enclose(b),
    "equal": lambda a, b: FirstOrderModel.enclose(a) == FirstOrderModel.enclose(b),
    "not_equal": lambda a, b: FirstOrderModel.enclose(a) != FirstOrderModel.enclose(b),
    "logical_and": lambda a, b: Condition.enclose(a) & Condition.enclose(b),
    "logical_or": lambda a, b: Condition.enclose(a) | Condition.enclose(b),
}


def _heaviside(argument, value_at_zero):
    argument = FirstOrderModel.enclose(argument)
    if not argument.is_real:
        raise TypeError("a step of a complex quantity is not supported")
    below = argument.real_high < 0.0
    above = argument.real_low > 0.0
    low = np.where(above, 1.0, 0.0)
    high = np.where(below, 0.0, 1.0)
    return FirstOrderModel(_from_range(low, high, argument.real.centre))


def _dispatch_ufunc(ufunc, method, inputs, kwargs):
    if method != "__call__" or kwargs.get("out") is not None:
        raise TypeError("only direct calls of NumPy operations are supported here")
    name = ufunc.__name__
    if name in _BINARY_UFUNCS:
        return _BINARY_UFUNCS[name](inputs[0], inputs[1])
    if name == "negative":
        return -FirstOrderModel.enclose(inputs[0])
    if name == "positive":
        return FirstOrderModel.enclose(inputs[0])
    if name == "sqrt":
        return FirstOrderModel.enclose(inputs[0]).sqrt()
    if name == "square":
        return FirstOrderModel.enclose(inputs[0]) ** 2
    if name == "reciprocal":
        return FirstOrderModel.enclose(inputs[0]).reciprocal()
    if name == "heaviside":
        return _heaviside(inputs[0], inputs[1])
    if name == "logical_not":
        return ~Condition.enclose(inputs[0])
    if name == "conjugate":
        value = FirstOrderModel.enclose(inputs[0])
        return FirstOrderModel(value.real, value.imag.negated())
    raise TypeError(
        "the matrix element uses %r, which the bound derived here does not "
        "cover; extend funcs/sampling_bounds.py before shipping a table that "
        "needs it"
        % name)


def _dispatch_function(function, args, kwargs):
    if function is np.select:
        return _select(*args, **kwargs)
    raise TypeError(
        "the matrix element uses %r, which the bound derived here does not "
        "cover; extend funcs/sampling_bounds.py before shipping a table that "
        "needs it"
        % function.__name__)


def _select(condlist, choicelist, default=0):
    """Enclose a branch chosen by conditions that the rectangle may straddle.

    Where a rectangle settles the conditions, the chosen branch is returned
    unchanged.  Where it straddles a switching surface, more than one branch is
    reachable and the result is the smallest box containing all of them.
    """
    conditions = [Condition.enclose(item) for item in condlist]
    choices = [FirstOrderModel.enclose(item) for item in choicelist]
    shape = ()
    for candidate in choices:
        if np.shape(candidate.real.centre):
            shape = np.shape(candidate.real.centre)
            break
    else:
        for candidate in conditions:
            if candidate.certain.shape:
                shape = candidate.certain.shape
                break
    settled = np.zeros(shape, dtype=bool)
    reachable = []
    for condition, choice in zip(conditions, choices):
        can_reach = np.broadcast_to(condition.possible, shape) & ~settled
        if np.any(can_reach):
            reachable.append((can_reach, choice.broadcast(shape)))
        settled = settled | np.broadcast_to(condition.certain, shape)
    if not np.all(settled):
        reachable.append((~settled, FirstOrderModel.enclose(default).broadcast(shape)))
    if not reachable:
        raise ValueError("no branch of the matrix element is reachable")
    if len(reachable) == 1:
        return reachable[0][1]
    return _branch_hull(reachable, shape)


def _branch_hull(reachable, shape):
    real_low = np.full(shape, np.inf)
    real_high = np.full(shape, -np.inf)
    imag_low = np.full(shape, np.inf)
    imag_high = np.full(shape, -np.inf)
    for mask, choice in reachable:
        real_low = np.where(mask, np.minimum(real_low, choice.real_low), real_low)
        real_high = np.where(mask, np.maximum(real_high, choice.real_high), real_high)
        imaginary_spread = choice.imag.spread()
        imag_low = np.where(mask, np.minimum(imag_low, choice.imag.centre - imaginary_spread), imag_low)
        imag_high = np.where(mask, np.maximum(imag_high, choice.imag.centre + imaginary_spread), imag_high)
    template = np.zeros(shape)
    return FirstOrderModel(_from_range(real_low, real_high, template),
                  _from_range(imag_low, imag_high, template))


# ---------------------------------------------------------------------------
# Intermediate masses of a decay chain
# ---------------------------------------------------------------------------

def two_body_momentum(parent_mass, first_mass, second_mass):
    """Momentum of either daughter of a two-body decay in the parent frame.

    Zero where the split is closed.  The Kallen product is positive both above
    the threshold ``parent_mass >= first_mass + second_mass`` and below
    ``parent_mass <= |first_mass - second_mass|``, so the threshold is tested on
    the masses themselves and the sign of the product settles nothing.

    The arguments may be arrays, and the result broadcasts over them.
    """
    parent_mass = np.asarray(parent_mass, dtype=float)
    first_mass = np.asarray(first_mass, dtype=float)
    second_mass = np.asarray(second_mass, dtype=float)
    product = ((parent_mass ** 2 - (first_mass + second_mass) ** 2) *
               (parent_mass ** 2 - (first_mass - second_mass) ** 2))
    open_split = (parent_mass > first_mass + second_mass) & (product > 0.0)
    product = np.where(open_split, product, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        momentum = np.sqrt(product) / (2.0 * parent_mass)
    return np.where(np.isfinite(momentum), momentum, 0.0)


# How finely each intermediate-mass range is cut.  The table costs one matrix of
# cells by cells per step of the chain, so cutting more finely raises the share
# of candidates kept and at the same time makes each candidate cost more, since
# the search that places it walks a larger table.  Measured on this machine with
# the two counts interleaved so that the load cancels, 20000 events at a time:
# four pions 1.83e6 events per second at 128 cells against 1.60e6 at 256, six
# pions 7.00e5 against 6.13e5, eight pions 4.80e5 against 4.19e5, for a share
# kept of 94.5, 81.1 and 61.7 per cent at 128 against 97.2, 89.9 and 78.3 at
# 256.  The throughput is flat below 128 and the share kept keeps falling, so
# 128 sits where the two meet.  The table is then 1.3 MB at eight daughters and
# takes under two milliseconds to build.
_CELLS_PER_RANGE = 128


class ChainMassSampler:
    """Draws the intermediate invariant masses of a decay chain exactly.

    A decay of a parent of mass ``M`` into daughters ``m_0 ... m_{n-1}`` is
    generated as a chain: daughters 0 and 1 form a system of invariant mass
    ``M_1``, daughter 2 joins it to make ``M_2``, and so on up to
    ``M_{n-1} = M``.  Flat n-body phase space, written in those invariant
    masses, has density

        f(M_1, ..., M_{n-2}) = prod_k p(M_k; m_k, M_{k-1}),   M_0 = m_0,

    where ``p(a; b, c)`` is the momentum either daughter carries in a two-body
    decay of a mass ``a`` into masses ``b`` and ``c``.  Each ``M_k`` runs over
    ``[m_0 + ... + m_k, m_0 + ... + m_k + T]``, with ``T = M - sum(m_i)`` the
    kinetic energy the parent has to share; the steps of the chain telescope, so
    every configuration of the decay has all of its ``M_k`` inside those ranges.

    Each range is cut into cells.  Over a cell of ``M_k`` running up to ``high``
    and a cell of ``M_{k-1}`` starting at ``low``, the step momentum is at most
    ``p(high; m_k, low)``, because ``p(a; b, c)`` rises with the parent mass
    ``a`` and falls with a daughter mass ``c``:

    * with ``s = a^2`` the squared momentum is
      ``(s - (b+c)^2)(s - (b-c)^2) / (4 s)``, which is
      ``(s - (b+c)^2 - (b-c)^2 + (b+c)^2 (b-c)^2 / s) / 4`` and so has
      derivative ``(1 - (b+c)^2 (b-c)^2 / s^2) / 4`` in ``s``, non-negative
      because ``s >= (b+c)^2 >= |b^2 - c^2|`` and ``(b+c)(b-c) = b^2 - c^2``;
    * the derivative of the Kallen product in ``c`` is ``-4 c (a^2 + b^2 - c^2)``,
      negative because ``a >= b + c`` forces ``a^2 + b^2 > c^2``.

    Both statements hold along the path from the point to the corner of the two
    cells, since a step that is open at ``(M_k, M_{k-1})`` stays open as the
    parent mass grows to ``high`` and the daughter mass falls to ``low``.  Where
    the step is closed the density is zero, which any non-negative value bounds.

    Multiplying the per-step values gives, for each combination of cells, one
    value the density provably never exceeds anywhere inside it.  That
    piecewise-constant function is what candidates come from: a combination of
    cells is drawn with probability proportional to its value times its volume,
    a point is drawn uniformly inside it, and the point is kept with probability
    ``f(point) / value``.  The kept points then follow ``f`` exactly over the
    whole region, cells where the density vanishes included.

    The sum of the piecewise-constant function over all combinations of cells
    factorises along the chain, as do the conditional distributions of one cell
    given the previous one, so the table costs
    ``sum_k cells_k * cells_{k-1}`` to build whatever the number of daughters,
    and a draw costs one search per step.  Cutting the ranges more finely drives
    the values down towards the density itself and the acceptance rate towards
    one.
    """

    def __init__(self, parent_mass, masses, cells=None):
        self.parent_mass = float(parent_mass)
        self.masses = np.asarray(masses, dtype=float)
        count = len(self.masses)
        if count < 4:
            raise ValueError(
                "a decay chain of intermediate masses needs at least four "
                "daughters, and this one has %d" % count)
        self.kinetic = self.parent_mass - float(self.masses.sum())
        if self.kinetic <= 0.0:
            raise ValueError(
                "the decay of a parent of mass %g is closed for daughters "
                "summing to %g" % (self.parent_mass, float(self.masses.sum())))
        self.free = count - 2
        if cells is None:
            cells = _CELLS_PER_RANGE
        cells = int(cells)
        if cells < 1:
            raise ValueError("each intermediate-mass range needs at least one cell")

        cumulative = np.cumsum(self.masses)
        self.cell_low = []
        self.cell_high = []
        for index in range(self.free):
            edges = np.linspace(cumulative[index + 1],
                                cumulative[index + 1] + self.kinetic, cells + 1)
            self.cell_low.append(edges[:-1].copy())
            self.cell_high.append(edges[1:].copy())
        self.cell_width = [high - low
                           for low, high in zip(self.cell_low, self.cell_high)]

        # The largest each factor of the density can be on each cell, or on each
        # pair of neighbouring cells for the steps that join two free masses.
        # Each is widened by a few units in the last place, so that the rounding
        # of a candidate whose masses sit against the edge of its cell cannot
        # put the density above the value its own cell carries.
        widen = 1.0 + _ROUNDING_SLACK
        self.opening = widen * two_body_momentum(self.cell_high[0],
                                                 self.masses[1], self.masses[0])
        self.step = []
        for index in range(1, self.free):
            self.step.append(widen * two_body_momentum(
                self.cell_high[index][:, None], self.masses[index + 1],
                self.cell_low[index - 1][None, :]))
        self.closing = widen * two_body_momentum(
            self.parent_mass, self.masses[count - 1],
            self.cell_low[self.free - 1])

        # Summing the piecewise-constant function over the cells of the later
        # masses, one step at a time, leaves the mass each cell of M_k carries
        # to the total.  These also give the conditional distribution of the
        # cell of M_k once the cell of M_{k-1} is known.
        self._suffix = [None] * self.free
        self._suffix[self.free - 1] = self.closing.copy()
        for index in range(self.free - 2, -1, -1):
            spread = self.step[index] * self.cell_width[index + 1][:, None]
            self._suffix[index] = (spread * self._suffix[index + 1][:, None]).sum(axis=0)
        head = self.opening * self.cell_width[0] * self._suffix[0]
        self.total = float(head.sum())
        if not self.total > 0.0:
            raise ValueError(
                "flat phase space has no volume for a parent of mass %g "
                "decaying to masses %s"
                % (self.parent_mass,
                   ", ".join("%g" % value for value in self.masses)))

        self._first_cumulative = np.cumsum(head) / self.total
        self._first_cumulative[-1] = 1.0
        # One cumulative table per step, laid out so that a single search finds
        # the cell of M_k for every candidate at once: each row is shifted by
        # its own index, which makes the flattened table increase throughout.
        self._step_cumulative = []
        for index in range(1, self.free):
            spread = self.step[index - 1] * self.cell_width[index][:, None]
            weighted = (spread * self._suffix[index][:, None]).T
            row_total = weighted.sum(axis=1)
            reachable = row_total > 0.0
            weighted = np.where(reachable[:, None],
                                weighted / np.where(reachable, row_total, 1.0)[:, None],
                                1.0 / weighted.shape[1])
            cumulative = np.cumsum(weighted, axis=1)
            cumulative[:, -1] = 1.0
            self._step_cumulative.append(
                (cumulative + np.arange(cumulative.shape[0])[:, None]).ravel())
        self.cells = cells

    def __repr__(self):
        return ("ChainMassSampler(mass %g, daughters %s, %d cells per range)"
                % (self.parent_mass,
                   ", ".join("%g" % value for value in self.masses), self.cells))

    def density(self, point):
        """The flat-phase-space density at intermediate masses ``point``.

        ``point`` has one column per free intermediate mass.  The density is
        zero wherever a step of the chain is closed.
        """
        point = np.asarray(point, dtype=float)
        count = len(self.masses)
        value = two_body_momentum(point[:, 0], self.masses[1], self.masses[0])
        for index in range(1, self.free):
            value = value * two_body_momentum(point[:, index],
                                              self.masses[index + 1],
                                              point[:, index - 1])
        return value * two_body_momentum(self.parent_mass,
                                         self.masses[count - 1],
                                         point[:, self.free - 1])

    def ceiling_over_region(self, last_factor=None):
        """The largest value the piecewise-constant function takes.

        This is a value the density provably never exceeds anywhere in the
        region.  ``last_factor`` multiplies each cell of the last free mass by a
        value that is itself an upper bound on some extra factor over that cell,
        which is what ``four_body_weight_bound`` needs.
        """
        best = self.opening
        for index in range(1, self.free):
            best = (self.step[index - 1] * best[None, :]).max(axis=1)
        closing = self.closing if last_factor is None else self.closing * last_factor
        return float((best * closing).max())

    def region_integral(self):
        """A midpoint estimate of the integral of the density over the region.

        Evaluating the density at the centre of each combination of cells and
        summing over volumes is the midpoint rule, and it factorises along the
        chain exactly as the piecewise-constant function does.  Dividing it by
        ``total`` estimates the acceptance rate, which is what the extra pass is
        for; nothing in the sampler depends on the value.
        """
        count = len(self.masses)
        centre = [0.5 * (low + high)
                  for low, high in zip(self.cell_low, self.cell_high)]
        value = (two_body_momentum(centre[0], self.masses[1], self.masses[0])
                 * self.cell_width[0])
        for index in range(1, self.free):
            spread = two_body_momentum(centre[index][:, None],
                                       self.masses[index + 1],
                                       centre[index - 1][None, :])
            value = (spread * value[None, :]).sum(axis=1) * self.cell_width[index]
        closing = two_body_momentum(self.parent_mass, self.masses[count - 1],
                                    centre[self.free - 1])
        return float((value * closing).sum())

    def estimated_acceptance(self):
        """The share of candidates the accept-reject step is expected to keep."""
        return self.region_integral() / self.total

    def draw_candidates(self, size, rng):
        """Draw candidate intermediate masses with the ceiling of their cells."""
        size = int(size)
        index = np.empty((size, self.free), dtype=np.intp)
        index[:, 0] = np.minimum(
            np.searchsorted(self._first_cumulative, rng.random(size), side="right"),
            len(self._first_cumulative) - 1)
        for step in range(1, self.free):
            columns = len(self.cell_low[step])
            previous = index[:, step - 1]
            position = np.searchsorted(self._step_cumulative[step - 1],
                                       previous + rng.random(size), side="right")
            index[:, step] = np.minimum(position - previous * columns, columns - 1)

        point = np.empty((size, self.free), dtype=float)
        for step in range(self.free):
            chosen = index[:, step]
            point[:, step] = (self.cell_low[step][chosen] +
                              rng.random(size) * self.cell_width[step][chosen])

        ceiling = self.opening[index[:, 0]]
        for step in range(1, self.free):
            ceiling = ceiling * self.step[step - 1][index[:, step], index[:, step - 1]]
        ceiling = ceiling * self.closing[index[:, self.free - 1]]
        return point, ceiling

    def draw(self, size, rng=None):
        """Return ``size`` independent draws of the intermediate masses.

        Every returned row is its own draw from flat phase space: none repeats
        another, and the number asked for sets the resolution.
        """
        size = int(size)
        if rng is None:
            rng = np.random
        point = np.empty((size, self.free), dtype=float)
        filled = drawn = 0
        while filled < size:
            missing = size - filled
            rate = max(filled / drawn, 1.0e-3) if drawn else 0.5
            batch = int(min(max(1.2 * missing / rate, 1024), 2_000_000))
            candidate, ceiling = self.draw_candidates(batch, rng)
            density = self.density(candidate)
            drawn += batch
            if np.any(density > ceiling):
                worst = int(np.argmax(density - ceiling))
                raise RuntimeError(
                    "The flat-phase-space density reached %.17g at intermediate "
                    "masses %s, above the %.17g the cells of %s allow there. "
                    "Each cell carries the product of the largest value every "
                    "step of the chain can take inside it, which the density "
                    "cannot exceed, so the derivation in "
                    "funcs/sampling_bounds.py and the density evaluated here "
                    "have drifted apart."
                    % (density[worst],
                       ", ".join("%.17g" % value for value in candidate[worst]),
                       ceiling[worst], self))
            keep = rng.random(batch) * ceiling < density
            take = min(int(keep.sum()), missing)
            point[filled:filled + take] = candidate[keep][:take]
            filled += take
            if drawn > 2000 * size + 10_000_000:
                raise RuntimeError(
                    "Accept-reject on the intermediate masses of %s kept %d of "
                    "%d candidates, far below the rate the cells predict."
                    % (self, filled, drawn))
        return point


def chain_mass_sampler(parent_mass, masses):
    """The sampler for one parent mass and one set of daughter masses.

    The table behind it depends on nothing else, so it is built once and kept.
    A different parent mass or a different set of daughter masses is a different
    key and builds its own table.
    """
    return _chain_mass_sampler(float(parent_mass),
                               tuple(float(value) for value in masses))


@functools.lru_cache(maxsize=32)
def _chain_mass_sampler(parent_mass, masses):
    return ChainMassSampler(parent_mass, masses)


def four_body_mass_sampler(parent_mass, mass1, mass2, mass3, mass4):
    """The sampler for a four-body decay.

    The four-body chain pairs daughters 3 and 4 into a system of invariant mass
    ``m34``, adds daughter 2 to make ``m234``, and adds daughter 1 to make the
    parent, so the chain reads the daughter masses in the order
    ``m3, m4, m2, m1``.  Column 0 of a draw is then ``m34`` and column 1 is
    ``m234``.
    """
    return chain_mass_sampler(parent_mass, (mass3, mass4, mass2, mass1))


def four_body_weight_bound(parent_mass, mass1, mass2, mass3, mass4):
    """A value the four-body weight of uniform mass draws never exceeds.

    One way to reach flat four-body phase space is to draw ``m234`` uniformly
    over ``[m2 + m3 + m4, M - m1]``, then ``m34`` uniformly over
    ``[m3 + m4, m234 - m2]``, and weight the pair by

        w = (m234 - m2 - m3 - m4) * p(M; m1, m234)
                                  * p(m234; m2, m34)
                                  * p(m34; m3, m4),

    where the first factor is the width of the ``m34`` range, which restores the
    measure of the variable-width range the second draw came from, and the other
    three are the density ``ChainMassSampler`` describes.

    Every drawn pair lies in the rectangle ``[m2 + m3 + m4, M - m1]`` by
    ``[m3 + m4, M - m1 - m2]``, which the cells of that sampler cover.  Inside
    one combination of cells the density is at most the value the cells carry
    and the width factor is at most ``m234`` at the top of its cell less
    ``m2 + m3 + m4``, so their product bounds the weight there; the largest such
    product over the cells bounds it everywhere.
    """
    sampler = four_body_mass_sampler(parent_mass, mass1, mass2, mass3, mass4)
    width = np.maximum(sampler.cell_high[1] - (mass2 + mass3 + mass4), 0.0)
    bound = sampler.ceiling_over_region(last_factor=width)
    if not bound > 0.0:
        raise ValueError(
            "the four-body phase-space weight vanishes identically for a parent "
            "of mass %g" % parent_mass)
    return bound


def chain_weight_bound(parent_mass, masses):
    """A value the flat-phase-space density of a decay chain never exceeds.

    The density is the product of the momenta released at each step of the
    chain, as a function of the intermediate invariant masses; see
    ``ChainMassSampler`` for it and for why the value each combination of cells
    carries bounds it inside that combination.  The largest of those values
    bounds the density over the whole region.
    """
    sampler = chain_mass_sampler(parent_mass, masses)
    bound = sampler.ceiling_over_region()
    if not bound > 0.0:
        raise ValueError(
            "the chain phase-space weight vanishes identically for a parent of "
            "mass %g" % parent_mass)
    return bound


# ---------------------------------------------------------------------------
# Three-body decays with a tabulated matrix element
# ---------------------------------------------------------------------------

class DalitzRectangleBounds:
    """Rectangles covering the allowed region, each with a value on it.

    The kinematically allowed region of the plane spanned by the energies of
    daughters 1 and 3 is covered by disjoint rectangles.  ``ceilings[i]`` is a
    value the matrix element provably never exceeds inside rectangle ``i``, so
    drawing a rectangle with probability proportional to its area times its
    ceiling, then a point uniformly inside that rectangle, then keeping the
    point with probability ``weight / ceiling`` returns points distributed
    exactly as the matrix element over the region.
    """

    __slots__ = ("energy1_low", "energy1_high", "energy3_low", "energy3_high",
                 "ceilings", "areas", "global_ceiling", "attained",
                 "rectangles_examined", "estimated_acceptance", "_cumulative")

    def __init__(self, energy1_low, energy1_high, energy3_low, energy3_high,
                 ceilings, attained, rectangles_examined, region_integral=0.0):
        self.energy1_low = np.asarray(energy1_low, dtype=float)
        self.energy1_high = np.asarray(energy1_high, dtype=float)
        self.energy3_low = np.asarray(energy3_low, dtype=float)
        self.energy3_high = np.asarray(energy3_high, dtype=float)
        self.ceilings = np.asarray(ceilings, dtype=float)
        self.areas = ((self.energy1_high - self.energy1_low) *
                      (self.energy3_high - self.energy3_low))
        self.global_ceiling = float(self.ceilings.max()) if self.ceilings.size else 0.0
        self.attained = float(attained)
        self.rectangles_examined = int(rectangles_examined)
        weights = self.areas * self.ceilings
        total = weights.sum()
        if not total > 0.0:
            raise ValueError(
                "the rectangles of this channel carry no area under them")
        self._cumulative = np.cumsum(weights) / total
        # The share of candidates the accept-reject step is expected to keep:
        # the integral of the matrix element over the region, estimated on the
        # rectangles, divided by the area under the cover.  It sizes the batch
        # a caller draws and nothing else; a batch too small is followed by
        # another, and the sampled distribution does not depend on it.
        self.estimated_acceptance = min(max(region_integral / total, 1.0e-3), 1.0)

    def __len__(self):
        return len(self.ceilings)

    @property
    def area_under_bound(self):
        """Area under the piecewise-constant bound, which sets the acceptance
        rate."""
        return float(np.sum(self.areas * self.ceilings))

    @property
    def tightness(self):
        """The ceiling divided by the largest value actually found."""
        if self.attained <= 0.0:
            return float("inf")
        return self.global_ceiling / self.attained

    def draw_candidates(self, count):
        """Draw candidate energies and return them with their rectangle ceiling."""
        draw = np.random.random(count)
        index = np.searchsorted(self._cumulative, draw, side="right")
        index = np.minimum(index, len(self.ceilings) - 1)
        energy1 = self.energy1_low[index] + np.random.random(count) * (
            self.energy1_high[index] - self.energy1_low[index])
        energy3 = self.energy3_low[index] + np.random.random(count) * (
            self.energy3_high[index] - self.energy3_low[index])
        return energy1, energy3, self.ceilings[index]


def dalitz_region_mask(parent_mass, mass1, mass2, mass3, energy1, energy3):
    """Whether energies land inside the kinematically allowed region.

    Daughter 2 takes the energy the other two leave, and the three momenta close
    into a triangle exactly when the squared momentum of daughter 2 lies between
    the smallest and the largest value the other two momenta can combine to.
    """
    energy2 = parent_mass - energy1 - energy3
    momentum1_squared = energy1 ** 2 - mass1 ** 2
    momentum3_squared = energy3 ** 2 - mass3 ** 2
    imbalance = (energy2 ** 2 - mass2 ** 2 - momentum1_squared - momentum3_squared) ** 2
    closure = 4.0 * momentum1_squared * momentum3_squared
    return (energy2 > mass2) & (imbalance < closure)


def _region_reachable(parent_mass, mass1, mass2, mass3, energy1, energy3):
    """Whether a rectangle can contain a point of the allowed region."""
    energy2 = parent_mass - energy1 - energy3
    momentum1_squared = energy1 ** 2 - mass1 ** 2
    momentum3_squared = energy3 ** 2 - mass3 ** 2
    imbalance = (energy2 ** 2 - mass2 ** 2 - momentum1_squared - momentum3_squared) ** 2
    closure = 4.0 * momentum1_squared * momentum3_squared
    return ((energy2 > mass2) & (imbalance < closure)).possible


def _enclose_matrix_element(matrix_element, parent_mass, masses, boxes):
    energy1 = FirstOrderModel.variable(boxes[0], boxes[1], 1)
    energy3 = FirstOrderModel.variable(boxes[2], boxes[3], 3)
    value = matrix_element(parent_mass, energy1, energy3, *masses)
    if isinstance(value, np.ndarray) and value.dtype == object:
        value = value.item()
    if not isinstance(value, FirstOrderModel):
        value = FirstOrderModel.enclose(np.broadcast_to(np.asarray(value, dtype=float),
                                               np.shape(boxes[0])))
    return value


def _rectangle_ceilings(matrix_element, parent_mass, masses, boxes):
    """A value the matrix element cannot exceed on each rectangle.

    A rectangle on which the expression has no finite bound produces
    infinities, and combining those gives undefined values; both outcomes are
    read below as "not bounded here", so the floating-point flags they raise
    are expected and are silenced rather than reported.
    """
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        energy1 = FirstOrderModel.variable(boxes[0], boxes[1], 1)
        energy3 = FirstOrderModel.variable(boxes[2], boxes[3], 3)
        reachable = _region_reachable(parent_mass, masses[0], masses[1],
                                      masses[2], energy1, energy3)
        value = _enclose_matrix_element(matrix_element, parent_mass, masses,
                                        boxes)
        ceiling = np.where(reachable, value.real_high, 0.0)
    # A rectangle whose bound came out undefined is treated as having none, so
    # that it is split rather than trusted.  The sampler reads a negative matrix
    # element as zero, so a rectangle on which the expression never rises above
    # zero contributes nothing.
    ceiling = np.where(np.isnan(ceiling), np.inf, ceiling)
    return np.maximum(ceiling, 0.0)


# Points per side of the lattice each rectangle is sampled on.  Three is
# enough to tell a rectangle that lies inside the allowed region from one that
# hangs half outside it, which is what the estimate is for.
_LATTICE_PER_SIDE = 3

# Cells per side of the grid the subdivision starts from.  Measured against 16
# and 64 on the shipped channels: 16 leaves more of the work to the rounds,
# which is what costs on the expensive matrix elements, and 64 spends half the
# rectangle budget before a single round and leaves a matrix element with a
# pole against the edge of the region too little of it to resolve the pole.
_SEED_CELLS = 32


def _interior_values(matrix_element, parent_mass, masses, boxes):
    """The matrix element on a lattice inside each rectangle.

    These are the matrix element at single points, not bounds over a
    rectangle.  The largest of them is a lower bound on the true maximum, which
    says how much slack is left in the ceiling.  Their mean, with points
    outside the allowed region counted as zero, estimates the mean of the
    weight over the rectangle, and the area-weighted sum of those means
    estimates the integral the rectangles have to cover, which is what says how
    much of the area under them is wasted.  A lattice rather than the centre
    alone is what makes that estimate see the region: a rectangle that hangs
    half outside it wastes half its candidates, and a single centre point
    inside the region reports no waste at all.

    Returns the mean and the largest value, one of each per rectangle.
    """
    offsets = (np.arange(_LATTICE_PER_SIDE) + 0.5) / _LATTICE_PER_SIDE
    low1, high1, low3, high3 = boxes
    shape = np.shape(low1)
    energy1 = np.concatenate(
        [np.broadcast_to(low1 + step * (high1 - low1), shape)
         for step in offsets for _ in offsets])
    energy3 = np.concatenate(
        [np.broadcast_to(low3 + step * (high3 - low3), shape)
         for _ in offsets for step in offsets])
    inside = dalitz_region_mask(parent_mass, masses[0], masses[1], masses[2],
                                energy1, energy3)
    values = np.zeros(energy1.shape, dtype=float)
    if np.any(inside):
        sampled = np.asarray(matrix_element(parent_mass, energy1[inside],
                                            energy3[inside], *masses))
        if np.iscomplexobj(sampled):
            sampled = sampled.real
        sampled = np.broadcast_to(sampled, values[inside].shape)
        values[inside] = np.where(np.isfinite(sampled) & (sampled > 0.0),
                                  sampled, 0.0)
    values = values.reshape((_LATTICE_PER_SIDE ** 2,) + shape)
    return values.mean(axis=0), values.max(axis=0)


def dalitz_upper_bound(matrix_element, parent_mass, mass1, mass2, mass3,
                       energy1_range, energy3_range, ceiling_tolerance=0.05,
                       waste_target=1.15, max_rectangles=8192):
    """Cover a matrix element with cells it cannot rise above.

    The rectangle ``energy1_range x energy3_range`` covers every pair of
    energies a candidate can land on.  It is cut into a regular grid of cells
    and those are cut further; a cell that provably holds no allowed point, and
    a cell on which the matrix element provably never rises above zero, are
    dropped.  At every stopping point each surviving cell carries a value the
    matrix element cannot exceed inside it, so the cover is valid throughout and
    refinement only decides how many candidates are wasted.

    Refinement runs in two stages, because the cover is asked for two different
    things.  The first stage cuts the cells whose bound stands highest, which
    drives the largest bound down towards the true maximum; it stops once that
    bound is within ``ceiling_tolerance`` of the largest value the matrix
    element was seen to take, since the true maximum lies between the two.  The
    second stage cuts the cells that waste most: the area each carries under the
    cover, less the part of it the matrix element fills.  That difference is
    what a candidate drawn from the cell pays for, whether it is wasted outside
    the allowed region or under a ceiling that stands above the weight, and the
    total of it over the cells is what sets the acceptance rate.  The stage
    stops once the area under the cover is within ``waste_target`` of the
    integral of the matrix element estimated on the cells, or when a further
    round takes less than two per cent off the waste, or when the cell budget is
    spent.
    """
    masses = (mass1, mass2, mass3)
    low1, high1 = float(energy1_range[0]), float(energy1_range[1])
    low3, high3 = float(energy3_range[0]), float(energy3_range[1])
    if not (high1 > low1 and high3 > low3):
        raise ValueError("the three-body energy ranges are empty")

    state = _Subdivision(matrix_element, parent_mass, masses)
    state.seed(low1, high1, low3, high3)

    ceiling_budget = max(2, max_rectangles // 2)
    while len(state.ceilings) < ceiling_budget:
        highest = float(state.ceilings.max())
        if state.attained > 0.0 and highest <= state.attained * (1.0 + ceiling_tolerance):
            break
        state.refine(state.ceilings, ceiling_budget)

    previous_waste = float("inf")
    while len(state.ceilings) < max_rectangles:
        areas = state.areas()
        contributions = areas * state.ceilings
        contributions = np.where(np.isfinite(contributions), contributions, np.inf)
        wasted = contributions - areas * state.estimates
        current_mass = float(np.sum(contributions))
        if np.isfinite(current_mass):
            current_waste = float(np.sum(wasted))
            if not current_mass > 0.0 or current_waste > previous_waste * 0.98:
                break
            if current_mass <= waste_target * state.region_integral():
                break
            previous_waste = current_waste
        state.refine(wasted, max_rectangles)

    return DalitzRectangleBounds(state.boxes[0], state.boxes[1], state.boxes[2],
                          state.boxes[3], state.ceilings, state.attained,
                          state.examined, state.region_integral())


class _Subdivision:
    """Working set of rectangles during the subdivision."""

    def __init__(self, matrix_element, parent_mass, masses):
        self.matrix_element = matrix_element
        self.parent_mass = parent_mass
        self.masses = masses
        self.boxes = None
        self.ceilings = None
        self.estimates = None
        self.attained = 0.0
        self.examined = 0

    def seed(self, low1, high1, low3, high3, cells=None):
        """Start from a regular grid of cells over the whole rectangle.

        One cell over the whole rectangle would be the cheapest start, but the
        rounds that follow each cost a pass of the model arithmetic over the
        cells they cut, and on the larger tabulated matrix elements one such
        pass costs a tenth of a second whatever its size.  Laying down the grid
        in a single pass buys the first several rounds' worth of cells for the
        price of one.
        """
        cells = _SEED_CELLS if cells is None else cells
        edges1 = np.linspace(low1, high1, cells + 1)
        edges3 = np.linspace(low3, high3, cells + 1)
        boxes = (np.repeat(edges1[:-1], cells), np.repeat(edges1[1:], cells),
                 np.tile(edges3[:-1], cells), np.tile(edges3[1:], cells))
        ceilings = _rectangle_ceilings(self.matrix_element, self.parent_mass,
                                       self.masses, boxes)
        estimates, largest = _interior_values(self.matrix_element,
                                              self.parent_mass, self.masses,
                                              boxes)
        self.examined = int(ceilings.size)
        if not np.any(ceilings > 0.0):
            raise ValueError(
                "the matrix element never rises above zero over the allowed "
                "region of this channel")
        alive = ceilings > 0.0
        self.boxes = tuple(part[alive] for part in boxes)
        self.ceilings = ceilings[alive]
        self.estimates = estimates[alive]
        self.attained = float(largest.max())

    def areas(self):
        return (self.boxes[1] - self.boxes[0]) * (self.boxes[3] - self.boxes[2])

    def region_integral(self):
        return float(np.sum(self.areas() * self.estimates))

    def refine(self, priority, budget):
        """Split the highest-priority rectangles, choosing the cut direction.

        The rectangles split are the ones that carry the first half of the
        priority, between a thirty-second and a quarter of the working set.  A
        matrix element whose peak sits against the edge of the region has its
        priority in a handful of rectangles, and splitting a fixed share of the
        set each round would spend the budget on the rest of the plane long
        before those were cut deeply enough; one whose priority is spread over
        the whole plane needs the budget spent quickly instead, because it is
        the number of rounds, not the number of rectangles, that costs on the
        larger matrix elements.  The lower limit keeps the budget shrinking
        whatever the shape, so the rounds end.

        Each rectangle is cut into four strips, and both directions are tried
        before one is chosen.  The direction that leaves fewer strips without a
        finite bound wins, and between equals the direction that leaves less
        area under the cover wins; only when neither direction bounds anything
        does the longer side decide.  Cutting along the longer side throughout
        would be wrong wherever the bound is governed by one energy alone: the set where a matrix element approaches a pole is a
        thin strip along one direction, and cornering a strip by cutting both
        directions in turn costs exponentially more rectangles than cutting the
        one direction that matters.  Four strips rather than two are used
        because one cut is often not enough to tell the two directions apart,
        and the strips that were evaluated to make the choice are exactly the
        ones kept.
        """
        count = len(self.ceilings)
        order = np.argsort(priority)
        running = np.cumsum(priority[order][::-1])
        total = running[-1] if running.size else 0.0
        if np.isfinite(total) and total > 0.0:
            carrying = int(np.searchsorted(running, 0.5 * total) + 1)
        else:
            # An infinite priority is a rectangle with no finite bound on it;
            # those are the ones to cut, and there is no half of an infinite
            # total to take.
            carrying = int(np.count_nonzero(~np.isfinite(priority))) or 1
        chosen_count = min(max(carrying, count // 32 + 1), count // 4 + 1)
        chosen_count = max(1, min(chosen_count, max(1, (budget - count) // 3)))
        chosen = order[-chosen_count:]
        keep = np.ones(count, dtype=bool)
        keep[chosen] = False

        low1, high1, low3, high3 = (part[chosen] for part in self.boxes)
        step1 = 0.25 * (high1 - low1)
        step3 = 0.25 * (high3 - low3)
        candidates = []
        for strip in range(4):
            candidates.append((low1 + strip * step1, low1 + (strip + 1) * step1,
                               low3, high3))
        for strip in range(4):
            candidates.append((low1, high1,
                               low3 + strip * step3, low3 + (strip + 1) * step3))
        stacked = tuple(np.concatenate([part[i] for part in candidates])
                        for i in range(4))
        ceilings = _rectangle_ceilings(self.matrix_element, self.parent_mass,
                                       self.masses, stacked)
        # A strip lies inside its parent, so the parent's ceiling also holds
        # there.  Keeping the smaller of the two makes every round an
        # improvement: the bound on a small rectangle is not always the
        # tighter one, because the rounding error of evaluating a long
        # cancelling expression depends on where in the plane it is evaluated.
        ceilings = np.minimum(ceilings, np.tile(self.ceilings[chosen], 8))
        self.examined += len(ceilings)
        strips = np.split(ceilings, 8)
        areas = np.split((stacked[1] - stacked[0]) * (stacked[3] - stacked[2]), 8)

        def cost(first):
            unbounded = np.zeros(chosen_count, dtype=int)
            mass = np.zeros(chosen_count, dtype=float)
            for offset in range(4):
                value = strips[first + offset]
                bounded = np.isfinite(value)
                unbounded += (~bounded).astype(int)
                mass += np.where(bounded, value * areas[first + offset], 0.0)
            return unbounded, mass

        unbounded_1, mass_1 = cost(0)
        unbounded_3, mass_3 = cost(4)
        # Where neither direction bounds anything, the unbounded part of the
        # rectangle is a curve rather than a strip, and cutting the two
        # directions in turn is the way to corner it.
        blind = (unbounded_1 == 4) & (unbounded_3 == 4)
        cut_energy1 = np.where(
            blind,
            (high1 - low1) >= (high3 - low3),
            (unbounded_1 < unbounded_3) |
            ((unbounded_1 == unbounded_3) & (mass_1 <= mass_3)))

        parts, child_ceilings = [], []
        for strip in range(4):
            parts.append(tuple(np.where(cut_energy1, candidates[strip][i],
                                        candidates[4 + strip][i])
                               for i in range(4)))
            child_ceilings.append(np.where(cut_energy1, strips[strip],
                                           strips[4 + strip]))
        children = tuple(np.concatenate([part[i] for part in parts])
                         for i in range(4))
        child_ceilings = np.concatenate(child_ceilings)

        child_estimates, child_largest = _interior_values(
            self.matrix_element, self.parent_mass, self.masses, children)
        if child_largest.size:
            self.attained = max(self.attained, float(child_largest.max()))
        alive = child_ceilings > 0.0
        children = tuple(part[alive] for part in children)
        child_ceilings = child_ceilings[alive]
        child_estimates = child_estimates[alive]
        self.boxes = tuple(np.concatenate([self.boxes[i][keep], children[i]])
                           for i in range(4))
        self.ceilings = np.concatenate([self.ceilings[keep], child_ceilings])
        self.estimates = np.concatenate([self.estimates[keep], child_estimates])
        if len(self.ceilings) == 0:
            raise ValueError(
                "no rectangle of the three-body energy plane survives the "
                "kinematic test; the channel has no allowed region")
