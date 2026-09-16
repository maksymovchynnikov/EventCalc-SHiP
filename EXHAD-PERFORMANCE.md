# Generation performance

## rc10 conditional ALP generation

The installed integration generated **100,000 full 3-GeV ALP events in
49.8 s on one worker**, with its normal 512-event chunks and unit
hadronization weights. Initialization plus a separate 50-event warmup took
7.24 s. All events pass the unchanged 2e-6 conservation tolerance.

A controlled single-request comparison took **440.1 s with rc9 and
40.9 s with the conditional batched sampler**, a 10.8-fold single-core
speedup. These are actual equal-size runs; the different chunked seed
stream and request overhead explain why the installed number is reported
separately. No event bank, reused momenta, flat phase space or importance
weights are introduced. The original Pythia momentum routines are retained
after conditioning on the scarce eta/eta-prime plus two-pion channels.

Enable the optional accelerator from the exHad checkout with
`python tools/build_conditional.py --pythia-source /absolute/path/to/pythia8317`
after ordinary configuration. It applies to central-generator ALP at
2.4 < m <= 3.5 GeV; other masses and portals retain ordinary generation.
See exHad's `docs/PERFORMANCE.md` and conditional-kernel derivation for
the exact scope, rate-draw distinction and independent physics checks.

## Matched ALP with persistent parallel workers (rc9)

Actual 100,000-event full EventCalc runs at 3 GeV, eight workers and
512-event chunks, took **112.27 s unweighted** or **29.46 s weighted** with
floor fraction 0.1 and hadronization effective sample size **53,615**.
These timings exclude startup; initialization plus a separate 4,096-event
warmup took 13.23 s and 10.51 s. All laboratory records passed the existing
relative four-momentum tolerance of 2e-6. These were local precursor builds
with rc9's scheduler, not a universal performance guarantee.

A paired 4,000-event test took 17.88 s with one worker and 4.94 s with eight,
returning identical complete records and labels. The pool generates fresh,
independently seeded chunks and restores request order; it does not reuse
events, alter channel probabilities or smooth momentum distributions.
Weighted samples require their weights and observable-specific precision
checks; 100,000 weighted events are not 100,000 unweighted events.

The default is up to eight available CPUs; `EXHAD_WORKERS=1` selects serial
chunked execution. This reduces elapsed time by using additional cores and
memory, not by reducing the roughly 49 outer proposals per retained event
at this mass. The original single-request API is still available in exHad.
See `EXHAD.md` for reproducibility and configuration details.

## Raw EventCalc pipeline profile (13 September 2026)

A controlled 3 GeV ALP benchmark of 100,000 full laboratory events, in
10,000-event batches, took 20.16 s before the event-record/boost changes.
Timed and untimed runs produced identical complete event records and labels.
Its measured phase costs were:

| Operation | Time for 100,000 events |
|---|---:|
| Parent inverse-CDF sampling | 0.033 s |
| All parent/vertex work, including that sampling | 0.049 s |
| Pythia construction, settings and initialization, ten batches | 0.529 s |
| Pythia `forceHadronLevel`, 100,000 calls | 0.485 s |
| Building primary Pythia records | 0.574 s |
| Final-particle extraction, validation and loop overhead | 16.648 s |
| Boosts | 1.690 s |

The inverse-CDF row is included in the parent/vertex row; do not add it twice.
The extraction entry is a residual timing, not time inside fragmentation.
Pythia's Python `Event` binding has no iterator: implicit iteration goes
through the sequence protocol, ending with an exception for each event.
Explicit indices bounded by `event.size()` reduced full generation to 4.84 s.
Vectorizing the same boost formulas reduced it further to 3.20 s at 3 GeV
and 2.63 s at 2 GeV. The 3 GeV records/labels are bit-identical across all
three implementations (SHA-256
`682542c05a77cf35bf7a6ed534faf4a9a0795c7783b83f5a8094b08f19678175`).
The final 100,000-event samples at both masses pass the existing relative
four-momentum tolerance of 2e-6. Timings exclude initial table loading and
first JIT compilation, but include Pythia setup inside each batch.

These are native raw-Pythia events, not exHad rejection-sampled events.
The optimized extraction/boost code changes no branching fractions,
matrix elements, Pythia settings, cuts or weights. Native Pythia seeding
and numerical checks were held fixed across this comparison.

## Earlier exHad rejection-sampler measurements

The optimized sampler is the default. Reuse one `Generator` context for
successive batches; its bounded Pythia workers are closed with the context.
EventCalc retains exHad as its default and `--rawPythia` as the explicit
alternative physics backend.

## ALP measurements

These local macOS measurements use Python 3.14.6, Pythia 8.317 and the
universal-fermion ALP at 2 GeV. They count complete rest-frame decays, not
detector-selected events. Do not extrapolate them to other models or masses.

| Measurement, 1,000 decays | Earlier implementation | Current implementation |
|---|---:|---:|
| Buffered EventCalc, first batch | 18.98 s | 5.57 s |
| Buffered EventCalc, subsequent batches | 18.34 / 17.33 s | 1.80 / 1.79 s |
| Hadronic kernel, after 1,000-event warmup; batch-owned workers | rc2: 6.39 s | 2.58 s |

The EventCalc comparison includes its native nonhadronic channels and uses
one persistent public generator. Its warmed throughput is about 550 decays/s
in this benchmark, roughly ten times the original measurement. The direct
kernel comparison starts and closes its workers for each batch; the public
interface avoids that repeated startup. Parent sampling is timed separately
and takes approximately 0.0005 s per 1,000 attempts.

First use also imports and authenticates inputs and may compile native
kinematic kernels. It is not equivalent to steady-state generation. These
are separate local runs, not a controlled hardware benchmark: ambient load,
release-bound seed streams and the mix of rare channels affect wall times.

## Execution changes

- The rejection bound is the maximum of actual family-times-charge weights,
  not the product of unrelated maxima. At 2 GeV it falls from 39.80 to 9.94.
  Both are valid common bounds; replacing the larger one preserves every
  normalized channel probability. It changes which candidate is accepted.
- Rejected proposals stay in C++. Bounded source-local batches preserve
  logical proposal ordering; only accepted complete graphs cross into Python.
- Pythia-owned exclusive rows no longer generate and discard an EventCalc
  phase-space event. Input checks and matrix-element parsing remain.
- Active and exclusive Pythia runtimes are reused across public batches.
  Caches have fixed bounds, are tied to authenticated configurations, and
  are cleared on failure or context exit. Every proposal is explicitly seeded.
- Graphs are parsed once, and frozen probabilities are evaluated once per
  mass/variation batch.
- Since rc4, the public B-L generator prepares its authenticated rate inputs
  once. Subsequent batches evaluate the requested mass and variation directly,
  without repeating the full-grid initialization audit. This does not cache or
  interpolate mass-dependent rates. Start a new generator when changing inputs.

The warmed kernel profiles generated 48,458 versus 13,135 outer proposals
for approximately 1,000 retained fragmentation events. The latter used
1,678 source-batch calls. There is still genuine rejection work: the frozen
proposal distribution differs from the target hadronic composition.

## Correctness checks

At 20 model/mass points, 2,200 complete events match a Python reference
using the same joint bound, including when the compiled workers are reused
across model and mass changes. A separate test reinstates the old bound
and reproduces all 2,200 earlier events exactly, isolating removal of
discarded work from the deliberate acceptance-sequence change. Analytic
normalization tests check the joint-bound identity for 1,000 varied
family/charge distributions. No physics probability, matching curve,
selection, source support or hard failure limit is relaxed.

Public smoke tests check all seven model choices, four-momentum closure
and seed replay after an intervening mass change. Seeds reproduce batches
at fixed release, runtime, model, mass and event count; identical streams
across different releases are not promised.

`EXHAD_PORTABLE_REFERENCE=1` selects the Python rejection implementation
with the same current matching probabilities and bound. It is a diagnostic,
not raw Pythia and not the pre-rc3 acceptance sequence.
