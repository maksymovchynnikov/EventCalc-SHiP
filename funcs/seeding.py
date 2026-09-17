"""One seed entry point for every random stream EventCalc drives.

Three independent generators produce a decay record and none of them is
reached by ``numpy.random.seed`` alone:

* the host NumPy stream (``numpy.random``) used by the pure-Python/NumPy
  samplers (four-body, N-body, channel distribution);
* Numba's *own* RNG, which ``@njit`` code compiles in.  ``np.random.seed``
  called from the interpreter does not touch it, so every module with njit
  sampling exposes a ``seed_random`` njit entry point and this module calls
  all of them;
* Pythia's internal Mersenne twister, which restarts from the same default
  stream in every freshly constructed ``Pythia()`` unless ``Random:setSeed``
  and ``Random:seed`` are set.  ``process_events_with_pythia`` draws that
  seed from the host NumPy stream, so seeding NumPy here makes the Pythia
  stage reproducible as well.

``seed_all(seed)`` seeds all three coherently.  Two runs of the same card
with the same seed then produce identical decay products.

exHad's own generators are seeded separately and explicitly: every
``Generator`` call takes ``seed=`` and reproduces its batch at fixed
model/mass/count/chunk_size, for every worker count.  ``derive(seed, *tag)``
produces the stable per-channel sub-seeds used for that.
"""
from __future__ import annotations

import hashlib

import numpy as np

# Numba's RNG state is per process and shared by every njit function, but it
# must be entered through njit code at least once per module that NumPy's host
# seeding cannot reach.  Keep this list in step with the sampler modules.
_NJIT_MODULES = ("TwoBodyDecay", "ThreeBodyDecay")

_LAST_SEED = None


def derive(seed, *tag):
    """A stable 63-bit sub-seed for one labelled stream of a run.

    Pure function of the run seed and the tag, so a channel's events do not
    move when an unrelated channel's event count changes.
    """
    payload = ':'.join(['eventcalc-seed-v1', str(int(seed))] + [str(t) for t in tag])
    return int.from_bytes(hashlib.sha256(payload.encode('ascii')).digest()[:8], 'big') >> 1


def seed_all(seed):
    """Seed the host NumPy stream and Numba's njit RNG from one run seed."""
    global _LAST_SEED
    seed = int(seed)
    if not 0 <= seed < 2 ** 32:
        # NumPy's legacy seeder takes a 32-bit value; fold anything wider.
        seed = derive(seed) % (2 ** 32)
    np.random.seed(seed)
    for name in _NJIT_MODULES:
        module = __import__(f'funcs.{name}', fromlist=[name])
        module.seed_random(seed)
    _LAST_SEED = seed
    return seed


def last_seed():
    """The seed most recently applied by :func:`seed_all`, or None."""
    return _LAST_SEED
