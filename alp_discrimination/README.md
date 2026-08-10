# ALP model discrimination at SHiP

This package extends EventCalc-SHiP with an $SU(2)_L$-coupled axion-like particle
(ALP) benchmark and a reproducible analysis for distinguishing it from the
photophilic ALP benchmark at SHiP.

The question is: if an ALP signal is observed at known mass $m_a$, how many
accepted $a\to\gamma\gamma$ decays are required to identify the coupling
scenario? The ALP lifetime is not assumed known and is profiled independently
under both hypotheses.

## Physics setup

The two benchmark hypotheses are:

- **Photophilic ALP**: primary plus electromagnetic-cascade production.
- **$SU(2)_L$ ALP**: dominated by the loop-induced $b\to s a$ transition
  and rare $B\to X_s a$ decays. The CKM-suppressed $B\to\pi a$ contribution
  is retained to extend the kinematic mass range.

Both are studied through $a\to\gamma\gamma$. The implemented $SU(2)_L$
benchmark is a simplified diphoton-only decay model, with
$\mathrm{Br}(a\to\gamma\gamma)=1$. EventCalc supplies production, decay
probability and the SHiP parent decay-volume geometry. The additional
daughter-level geometric selection requires both photons to intersect the
simplified ECAL plane; the second selection additionally requires
$E_\gamma\geq1$ GeV for each photon.

Absolute rates determine the allowed lifetime domains. They are not included in
the discrimination likelihood, which is shape-only and conditioned on the
observed number of accepted events $N$.

## Discrimination method

For each model and lifetime, accepted EventCalc events define an ALP-energy
distribution and conditional decay-position moments. The supported observable
sets are:

- `energy`
- `energy_mean_z`
- `energy_mean_r_perp`
- `energy_mean_z_r_perp`

with $r_\perp=\sqrt{x_d^2+y_d^2}$. At fixed $N$, the likelihood profiles the
lifetime independently under each model. The reported accuracy is conservative
over truth model, allowed truth lifetime and random seed. The threshold is

$$
N_{90}=\min\{N\mid A_{N'}\geq0.90\;\text{for every tested }N'\geq N\}.
$$

The implementation supports every physical event count $N\geq1$. In particular,
a crossing at $N=2$ is not accepted as a lower-grid boundary: the full-domain
scan is extended to $N=1$ before the threshold is finalized.

## Run the analysis

From the EventCalc-SHiP repository root:

```bash
python -m alp_discrimination.workflows.analysis --interactive
```

For a non-interactive production run:

```bash
python -m alp_discrimination.workflows.analysis \
  --masses 0.3 0.5 1.0 2.5 \
  --selections diphoton_ecal diphoton_ecal_e1gev \
  --observables energy_mean_z_r_perp \
  --profile production \
  --run-mode reuse_only \
  --workers 2 \
  --resume \
  --stop-after final
```

`reuse_only` is the safe default for expensive work: registered validated or
production banks are reused and unavailable points are skipped rather than built
implicitly.

## Runtime

Production runs can take several hours. At $m_a=0.3$ GeV, for example, the
full-domain screen contains 89 photophilic and 96 $SU(2)_L$ lifetime points.
The screening stage uses five seeds and 2000 pseudoexperiments per truth point;
selected points are then repeated with 5000 or 10000 pseudoexperiments.

With two workers, final $m_a=0.3$ ECAL runs took about 5.5--6.7 hours on the
2020 MacBook Air used for this project. The main settings controlling the
runtime are the analysis profile, number of pseudoexperiments, seeds and the
lifetime/truth grids. Use `--dry-run` before a long run and `--resume` to reuse
completed checkpoints.

## Validation chain

A project-final spatial result passes four stages:

1. **Threshold scan**: locate the relevant event-count region.
2. **Lifetime scan**: evaluate every allowed truth lifetime with 2000
   pseudoexperiments and five seeds.
3. **High-statistics validation**: repeat difficult truths with 5000
   pseudoexperiments and audit omitted lifetimes; 10k is used only when needed.
4. **Empirical validation**: resample complete weighted EventCalc events and
   verify that the Gaussian conditional approximation does not change $N_{90}$.

The distance measures used for lifetime maps are diagnostics; the project test
statistic is the profiled likelihood evaluated in pseudoexperiments.

## Package structure

```text
alp_discrimination/
├── physics/       models, spectra, lifetime and event-rate domains
├── eventcalc/     EventCalc adapters, proposals and diphoton selections
├── templates/     probability/lifetime banks and conditional feature moments
├── statistics/    likelihoods, reductions, distances and adaptive scan logic
├── plotting/      shared plotting and report-facing figures
├── constraints/   exclusion-curve conversion and plotting helpers
├── reference_data/ bundled photon-sensitivity reference curves
├── workflows/     command-line orchestration and production stages
└── tests/         unit, regression and integration tests
```

Package-level infrastructure (`config.py`, `cache.py`, `paths.py`, `planning.py`
and `progress.py`) remains at the top level.

## Outputs and checkpoints

Final products are written under

```text
analysis2/outputs/production/alp_su2l_analysis/final_results/
```

`analysis2/cache` and `analysis2/outputs` are kept as runtime directories so
existing banks and checkpoints can be reused. Old checkpoint names such as
`pilot` and `lifetime_blind` are kept for the same reason.

Report tables and plots are exported to
`final_results/report/{plots,tables,data}`. The main discrimination table is
`final_results/report/tables/discrimination_thresholds.csv`.

## Numerical bank quality

`production_noise_floor_limited` means that the global distance minimum is
stable, while additional local lifetime-grid refinement is limited by numerical
or template-statistical noise. It does not mean that the physical minimum is
unstable.

## Limitations

The discrimination study uses idealised truth-level decay coordinates. It does
not include realistic diphoton vertex resolution, photon separation, full
reconstruction efficiency, backgrounds or detector systematics. Spatial results
therefore quantify the information available in the accepted simulated
kinematics, not a complete experimental sensitivity.

## Next step

A useful first extension is to introduce a finite transverse vertex resolution
$\Delta r_\perp$ and repeat the discrimination study. The strong improvement
from $\langle r_\perp\rangle$ currently uses exact simulated decay positions,
so this directly tests how much of the separation survives reconstruction.

A more complete detector study can later include photon separation and
reconstruction, efficiencies, backgrounds and systematics.

## Tests

```bash
python -m pytest alp_discrimination/tests -q
```

Run the full suite and the saved golden-result regression after structural or
statistical changes and before starting new production points.

## EventCalc core

The underlying EventCalc-SHiP sampler and geometry implementation are inherited
from the main EventCalc-SHiP project. See the repository-level documentation for
the general LLP sampling, decay and event-rate implementation.
