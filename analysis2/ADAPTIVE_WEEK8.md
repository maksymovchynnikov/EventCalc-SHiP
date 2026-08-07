# Adaptive Week-8 lifetime-blind scan

## Purpose

`analysis2.workflows.adaptive_week8_scan` produces the observed event count
needed for a persistent 90% model-classification accuracy for arbitrary ALP
mass points and for both detector selections:

- `diphoton_ecal`: both photons geometrically hit the ECAL;
- `diphoton_ecal_e1gev`: the same geometry plus the inclusive requirements
  `E_gamma1 >= 1 GeV` and `E_gamma2 >= 1 GeV`.

The statistical problem remains shape-only, conditioned on the observed event
count. Lifetime is profiled independently under the photon and SU(2)_L
hypotheses. The allowed lifetime domains remain the geometry-only SHiP
`N_events >= 2.3` domains minus existing exclusions; disconnected components
are never joined.

## Adaptive strategy

### Lifetime templates

The initial number of lifetime points in each connected component scales with
its logarithmic `c tau` width. The calibrated defaults are four points per
decade, at least five points per connected interval, and a maximum global
spacing of 0.25 decades. Endpoints are always retained.

Refinement uses these diagnostics:

1. logarithmic lifetime gap;
2. total-variation change between adjacent detector-level templates;
3. leave-one-out interpolation residual in log lifetime;
4. change of the competitive part of a distance-map row or column;
5. spacing and stability in the neighbourhood of the global minimum distance.

The nominal shape tolerances remain sensitive refinement priorities. They are
hard convergence requirements next to the current minimum and whenever their
combined priority is exceptionally large. Moderate curvature away from the
minimum guides refinement while a hard condition remains unresolved, but does
not force every harmless tail to the nominal tolerance. This stopping rule was
calibrated on the validated dense `m_a=0.3 GeV` bank: it reproduces the exact
minimum distance, minimizing lifetimes and connected components while avoiding
unnecessary global refinement.

Only sixteen new lifetimes per model are added in one round, with hard
requirements ranked before soft candidates. Up to eight rounds are allowed.
This best-first batching avoids overshooting a converged grid. The procedure is
axis-neutral: it allocates more points to the photophilic or SU(2)_L axis from
the actual detector-level distance surface rather than a fixed model
preference.

### Energy binning

The default bank starts from 200 energy bins and merges until every retained
bin meets the configured effective-statistics requirement. A cached finer
check is triggered only when the final bank has very few bins or the minimum
model distance is small. The finer stable representation is retained for the
likelihood calculation.

### Pseudoexperiments

1. A small distance-guided truth set brackets the event-count scale.
2. One final cache-stable event-count grid is fixed before the full-domain
   pilot. It contains a unit-spaced crossing window and a sparse persistence
   tail.
3. Every lifetime truth receives a 2,000-PE, five-seed pilot.
4. Truths close to the conservative envelope, distance minima and interval
   endpoints are promoted to high statistics.
5. The selected set is extended through 5k, 10k and, only when needed, 20k
   pseudoexperiments.
6. Omitted truths are certified with simultaneous one-sided bounds. Any truth
   whose bound overlaps the selected envelope is promoted automatically.
7. A result is final only after the threshold is stable across the required PE
   transitions and the omitted-truth audit passes.

Progressive truth caches store the NumPy random-generator state at every
checkpoint. A 5k -> 10k -> 20k extension therefore resumes directly at the
saved stream position instead of regenerating and discarding the earlier
pseudoexperiments.

## Resumption and safety

The scan never uses `--force` or overwrites an incomplete stage. Every
mass-selection point has an independent `state.json`, and incomplete stages are
continued in numbered retry directories. Re-running the same command resumes
completed work automatically. A settings fingerprint prevents incompatible
runs from sharing one output directory.

One failed point is recorded and the remaining overnight scan continues unless
`--fail-fast` is supplied.

## Initial check

```bash
python -m analysis2.workflows.adaptive_week8_scan \
  --masses 0.3 1.0 2.5 \
  --selections diphoton_ecal diphoton_ecal_e1gev \
  --profile validation \
  --workers 2 \
  --resume \
  --dry-run \
  --output-dir analysis2/outputs/validation/week8_adaptive_dry_run
```

The printed plan includes, for every mass and model, the connected-component
count, total logarithmic lifetime width and initial adaptive grid size.

## Dense-bank calibration (recommended before the overnight scan)

The existing validated `m_a=0.3 GeV` dense bank can test the lifetime-grid
planner without EventCalc generation or pseudoexperiments:

```bash
python -m analysis2.workflows.validate_adaptive_lifetime_grid \
  --banks \
  analysis2/outputs/validation/lifetime_blind_discrimination_week8_ma0p3_e1gev_adaptive185_400bin/template_banks/template_bank_ma_0p3.npz \
  --output-dir \
  analysis2/outputs/validation/week8_adaptive_dense_bank_calibration
```

The summary reports how many dense templates the adaptive planner retains, the
relative error in the minimum total-variation distance, and whether the
minimizing connected components are unchanged. This is a fast calibration of
the lifetime-grid approximation, not an independent `N90` calculation.

## Functional staged test

A first local interface test can stop after the range finder:

```bash
python -m analysis2.workflows.adaptive_week8_scan \
  --masses 1.0 \
  --selections diphoton_ecal diphoton_ecal_e1gev \
  --profile quick \
  --workers 2 \
  --resume \
  --initial-energy-bins 40 \
  --minimum-bin-n-eff 5 \
  --maximum-lifetime-rounds 2 \
  --maximum-lifetimes-per-model 35 \
  --lifetime-points-per-decade 3 \
  --rangefinder-pseudoexperiments 200 \
  --rangefinder-seeds 1 \
  --stop-after rangefinder \
  --output-dir analysis2/outputs/validation/week8_adaptive_smoke
```

This is only an interface/cache test and must not be used as a physics result.

## Importing already validated points

Previously finalized points can be inserted without rerunning them:

```bash
python -m analysis2.workflows.adaptive_week8_scan \
  --plot-only \
  --import-result-json path/to/final_result.json \
  --output-dir analysis2/outputs/production/week8_adaptive_scan
```

Imported or converged master-table points are skipped automatically in later
scans. Use `--rerun-final-points` only when an intentional independent rerun is
needed.

## Final scan

```bash
python -m analysis2.workflows.adaptive_week8_scan \
  --masses 0.3 0.5 0.75 1.0 1.5 2.0 2.5 \
  --selections diphoton_ecal diphoton_ecal_e1gev \
  --profile validation \
  --workers 2 \
  --resume \
  --output-dir analysis2/outputs/production/week8_adaptive_scan
```

Use exactly the desired mass list. Both selections are run consecutively for
each mass so the second selection reuses selection-independent EventCalc
proposals and paired blue/orange results are completed early.

## Outputs

- `adaptive_n90_results.csv`: one row per mass and selection;
- per-point convergence tables, audits, state files and manifests;
- `week8_n90_comparison.pdf` and `.png`;
- `per_mass/ma_*/geom|e1gev/final_result.json`.

The main plot includes only `converged` or explicitly
`imported_validated` points. Screening or grid-limited values remain in the CSV
but are not displayed as final physics results.

The plot convention is fixed:

- blue circles: ECAL geometry only;
- orange squares: ECAL geometry plus both-photon `E_gamma >= 1 GeV`.
