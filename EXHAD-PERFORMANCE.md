# Measured generation times

Every number here is a wall time from a separate local run on one macOS
machine with Pythia 8.317, under ambient load that was not controlled. They
are not a benchmark, they are not a promise for another model, mass or host,
and none of them is a reference a later change is required to beat. What they
are good for is the shape of the cost: where the time goes, and what a warm
run costs compared with a cold one.

The interpreter these runs used is not recorded with the timings. Repeat a
measurement on your own installation before planning a large scan around it.

## Fermion-coupled ALP, matched hadronic decays

exHad has two samplers of the same event distribution: the rejection sampler,
which fragments a complete string and accepts or rejects the trial event, and
the accelerated sampler, which decides acceptance while the string fragments
and generates the rarest channel groups directly. exHad uses the accelerated
one for unweighted ALP samples above 2.4 GeV and the rejection one for
weighted samples at every mass. At 3 GeV, with the default 512-event chunks:

| Run | Events | Time |
|---|---:|---:|
| Accelerated sampler, one worker | 100,000 | 49.8 s |
| Rejection sampler, eight workers | 100,000 | 112.27 s |
| Rejection sampler, eight workers, importance-weighted, floor fraction 0.1 | 100,000 | 29.46 s |

The timings exclude start-up. Initialization plus a separate warmup took
7.24 s for the first row and 13.23 s (10.51 s weighted) for the other two.
The rows come from separate runs with separate warmups.

A single-core comparison of the two samplers on one request of the same size
took 440.1 s with the rejection sampler and 40.9 s with the accelerated one.
The accelerated sampler introduces no event bank, no reused momenta, no flat
phase space and no importance weights: the momenta still come from Pythia's
own fragmentation, conditioned on the scarce $\eta$ and $\eta'$ two-pion
channels.

A paired test of the same 4,000 events took 17.88 s with one worker and
4.94 s with eight, and returned identical complete records and labels. Worker
processes reduce elapsed time by using more cores; they do not reduce the
number of proposals a retained event costs, which at 3 GeV is of order 50,
and they change no event.

A weighted sample of 100,000 events is not an unweighted sample of 100,000
events: at floor fraction 0.1 the effective sample size of the run above was
53,615, and the precision of a given observable has to be checked on that
observable.

Every laboratory record of these runs satisfies the relative four-momentum
closure of $2\times10^{-6}$ that the generator already enforced.

At 2 GeV, 1,000 complete rest-frame decays through the buffered EventCalc
interface took 5.57 s in the first batch and 1.80 s in each following batch,
about 550 decays per second warm. The first batch imports and authenticates
the model inputs and compiles the kinematic kernels; it is not steady-state
generation. Reuse one generator context across batches, and close it when the
context ends.

## Unmodified-Pythia baseline

The same ALP configuration on the Pythia baseline, 100,000 complete laboratory
events in batches of 10,000, took 3.20 s at 3 GeV and 2.63 s at 2 GeV. These
timings exclude the initial table loading and the first compilation of the
kernels, and include the Pythia setup inside each batch.

Two details of that path are worth knowing when reading a profile of it.
Pythia's Python `Event` binding has no iterator, so implicit iteration walks
the sequence protocol and ends with an exception for every event; EventCalc
therefore reads the record with explicit indices bounded by `event.size()`.
The boosts are evaluated as array operations over a whole batch. Neither
touches branching fractions, matrix elements, Pythia settings, cuts or
weights, and the event records they produce are bit-identical to the
element-by-element versions of the same formulas.

## What the matched generator spends its time on

The hadronic events of the ALP and the scalars are drawn by rejection: Pythia
fragments a string, the trial event is accepted with a probability set by its
channel group and charge combination, and the rejected trials stay in C++.
The bound of that acceptance is the maximum of the actual
group-times-charge weights rather than a product of unrelated maxima, which at
2 GeV is 9.94 instead of 39.80; both are valid common bounds and both leave
every normalized channel probability unchanged, so the smaller one only
discards less work. The remaining rejection is genuine: the distribution
Pythia proposes is not the hadronic composition the model requires.

The rest of the cost is set up once rather than per event: the decay graphs
are parsed once, the frozen probabilities are evaluated once per mass, and the
Pythia runtimes are reused across batches inside one generator context, with
bounded caches that are cleared when a context exits or a call fails. Each
proposal is explicitly seeded.

## Correctness alongside the speed

These checks are what make the timings above comparable with a slower path
rather than with a different physics.

- At 20 model and mass points, 2,200 complete events match a Python reference
  implementation that uses the same bound, including when the compiled workers
  are reused across changes of model and mass.
- A separate test reinstates the larger bound and reproduces all 2,200 events
  of that bound exactly, which separates the removal of discarded work from
  the deliberate change of which candidate is accepted.
- Analytic normalization tests check the bound identity for 1,000 varied
  group and charge distributions.
- The smoke tests cover every model choice, four-momentum closure, and seed
  replay after an intervening change of mass. A seed reproduces a batch at
  fixed installation, model, mass and event count.

`EXHAD_PORTABLE_REFERENCE=1` selects the Python implementation of the same
rejection sampling, with the same matching probabilities and the same bound.
It is a diagnostic: it is neither the Pythia baseline nor a different
acceptance rule.
