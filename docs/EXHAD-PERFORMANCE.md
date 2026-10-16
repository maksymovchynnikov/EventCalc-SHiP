# Measuring generation and acceptance time

Measure the complete command for the model, mass and selections of interest:

```sh
time python -m eventcalc.runtime_generator --card cards/exhad_alp_fermion.json \
  --mass 3 --ctau 10 --events 100000 --batch-size 10000 \
  --output-format npz --output-dir outputs/alp-timing
```

Report the processor, Python and Pythia8 versions, worker count, number of
requested parents and number of returned decays. Include whether timing covers
table loading, first-call compilation and writing event files. A second run in
the same Python context measures generation after initialization; a new process
includes setup again.

The runtime API can generate batches in memory when file output is unnecessary.
Reuse one scan context across lifetimes so production distributions and
rest-frame decays are prepared once per mass. Model and mass affect speed:
exclusive two-body decays need less work than complete hadronic decay chains.

The [weighted ALP option](EXHAD.md#weighted-alp-decays) can improve throughput
for hadronic samples. Compare precision through the effective sample size and
errors of the observables of interest. The number of weighted proposals alone
does not measure statistical precision.

The [acceptance integrator](SENSITIVITY.md#validation-and-performance) has separate
timings for transport and the lifetime scan. Those measurements exclude
production and decay generation.
