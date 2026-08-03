# ECAL-aware lifetime-profiled ALP discrimination

`analysis2` contains the reusable implementation of the ECAL-aware comparison
between the photophilic and $SU(2)_L$ ALP benchmarks. The four scripts in
`analysis/` are thin command-line wrappers. The pre-refactor implementations
are kept in `analysis/legacy_profiled_analysis/` for regression checks.

## Run the production analysis

From the repository root:

```bash
conda run -n eventcalc python -m analysis.scan_ctau_ranges
conda run -n eventcalc python -m analysis.lifetime_blind_discrimination
conda run -n eventcalc python -m analysis.lifetime_blind_distance_maps
conda run -n eventcalc python -m analysis.lifetime_blind_profiled_likelihood
```

The default profile is `production`. Faster diagnostic settings are selected
explicitly with `--profile quick` or `--profile validation`.

Generated files are separated by profile:

```text
analysis2/cache/<profile>/
analysis2/outputs/<profile>/
```

Both roots are ignored by Git. Only compact production summaries and numerical
comparison reports are tracked.

## Scientific definition

The production masses are

```text
0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.00, 1.05 GeV.
```

The comparison is shape-only and conditioned on the observed event count.
Absolute event rates are used only to determine the observable lifetime domain
of each model.

The photophilic prediction combines primary and cascade production with their
absolute normalisations. The $SU(2)_L$ prediction uses the inclusive source.

For the detector selection, the decay $a\to\gamma\gamma$ is generated
isotropically in the ALP rest frame and boosted to the laboratory frame. Both
photons must intersect the inclusive $4\,\mathrm{m}\times6\,\mathrm{m}$
rectangle in the ECAL plane at $z=95\,\mathrm{m}$.

For weighted events,

$$
N_{\mathrm{events}}=\sum_j w_j,
\qquad
N_{\mathrm{eff}}
=
\frac{\left(\sum_j w_j\right)^2}{\sum_j w_j^2}.
$$

At each mass, 20 photophilic and 20 $SU(2)_L$ lifetime templates share one
adaptive energy binning. The procedure starts from 50 logarithmic bins between
$m_a$ and $400\,\mathrm{GeV}$ and merges neighbouring bins until every
populated template bin satisfies $N_{\mathrm{eff}}\geq100$.

Jeffreys smoothing is applied as

$$
p_i
=
\frac{N_{\mathrm{eff}}p_{i,\mathrm{raw}}+\tfrac12}
     {N_{\mathrm{eff}}+\tfrac12 n_{\mathrm{bins}}}.
$$

For every pseudoexperiment, the likelihood is maximised independently over the
lifetime templates of the two models. The reported accuracy is the worst case
over truth model, truth lifetime and random seed.

The production configuration uses 100,000 pseudoexperiments per truth template
and seed, five seeds, and observed event counts from 1 to 12. The threshold is
the first event count for which the target accuracy remains satisfied at every
larger tested count.

The persistent worst-case 90% thresholds are

```text
2, 2, 2, 2, 3, 3, 4, 4 events
```

for the masses listed above.

## Lifetime endpoints

The template endpoint is obtained by log-log interpolation of the saved
event-rate scan at $N_{\mathrm{events}}=10$. An interior endpoint is shifted
inward by 0.2% of the logarithmic lifetime span before constructing the
template grid.

A separate 14-step logarithmic bisection is retained as a diagnostic. Its final
bracket midpoint is not used to construct the production templates. The two
endpoint definitions are tested independently.

## Validation

Run the test suite with

```bash
conda run --no-capture-output -n eventcalc pytest analysis2/tests -q
```

Compare the production outputs with the frozen reference calculation using

```bash
conda run --no-capture-output -n eventcalc \
  python -m analysis2.workflows.compare_frozen_results
```

The validated implementation gives `96 passed, 2 skipped`. The numerical
comparison exits with status zero and reproduces the frozen lifetime domains,
template banks, distance maps and profiled-likelihood results. The only nonzero
difference is a $3.55\times10^{-15}$ CSV round-trip effect.
