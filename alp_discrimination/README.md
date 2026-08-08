# ALP model discrimination at SHiP

This branch extends EventCalc-SHiP with an $SU(2)_L$-coupled axion-like particle
(ALP) benchmark and a reusable analysis for distinguishing it from the existing
photophilic ALP benchmark at SHiP.

The physics question is simple:

> If an ALP signal is observed with known mass $m_a$, how many accepted
> $a\to\gamma\gamma$ decays are required to identify which coupling scenario
> produced it?

The ALP lifetime is not assumed to be known. It is profiled independently under
the two model hypotheses, so the result is conservative against lifetime-induced
changes of the accepted kinematics.

## Physics setup

Two benchmarks are compared:

- **Photophilic ALP** — primary and electromagnetic-cascade production.
- **$SU(2)_L$ ALP** — dominated by rare $B\to X_s a$ decays, with the
  $b\to d a$ contribution retained where relevant.

Both benchmarks are studied through $a\to\gamma\gamma$. EventCalc supplies the
production spectra, decay probability and SHiP decay-volume geometry. The
additional diphoton selection requires both photons to intersect the ECAL plane;
a second selection also requires $E_\gamma\geq1$ GeV for each photon.

Absolute event rates are used to determine the **allowed lifetime domains**.
They are not included in the discrimination likelihood, which is shape-only and
conditioned on the observed number of accepted events $N$.

## Discrimination method

For each model and lifetime the accepted EventCalc sample defines an energy
probability distribution. The analysis can additionally use the conditional
decay-position information

$$
\langle z_d\rangle,\qquad
\langle r_\perp\rangle,\qquad
r_\perp=\sqrt{x_d^2+y_d^2}.
$$

The standard observable sets are:

- `energy`
- `energy_mean_z`
- `energy_mean_r_perp`
- `energy_mean_z_r_perp`

The accuracy at fixed $N$ is the worst case over truth model, allowed lifetime
and random seed. The reported threshold is

$$
N_{90}=\min\{N\mid A_{N'}\geq0.90\;\text{for every tested }N'\geq N\}.
$$

## Run the analysis

Activate the EventCalc environment and run from the repository root.

### Interactive interface

```bash
python -m alp_discrimination.workflows.analysis --interactive
```

### Reproduce the headline analysis

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

## Validation

A final result passes the following sequence:

1. **Threshold scan** — locate the relevant event-count region.
2. **Lifetime scan** — evaluate every allowed truth lifetime with 2000
   pseudoexperiments and five seeds.
3. **High-statistics validation** — repeat the difficult truths with 5000
   pseudoexperiments and audit omitted lifetimes. A 10k extension is used only
   when the 5k crossing is statistically marginal.
4. **Empirical validation** — for spatial observables, resample complete weighted
   EventCalc events and verify that the Gaussian conditional approximation does
   not change $N_{90}$.

The fully validated $m_a=0.3$ GeV ECAL-only joint result is currently

$$
N_{90}=4,
$$

with the same threshold in the all-lifetime 2k scan, the selected 5k validation
and direct EventCalc-row resampling.

## Outputs

Primary results are written under

```text
analysis2/outputs/production/alp_su2l_analysis/final_results/
```

The `analysis2/outputs` and `analysis2/cache` directories are retained as the
historical runtime-data namespace so validated checkpoints and caches remain
reusable after the source-package rename.

Internal checkpoint directories are kept for resumability. The compact
report-facing products are exported separately to

```text
final_results/report/
├── plots/
├── tables/
└── data/
```

The report products include:

- correct-classification probability versus observed decays $N$;
- validation comparison between the all-lifetime and high-statistics scans;
- report-style lifetime-distance maps with excluded regions shown in grey;
- observable-comparison plots;
- $N_{90}$ versus ALP mass;
- compact CSV tables and the numerical arrays underlying the distance maps.

## Numerical bank quality

Lifetime-template banks are labelled explicitly in the final summaries.
`production_noise_floor_limited` means that the location and value of the global
distance minimum are stable, while further lifetime-grid refinement is limited
by numerical/template-statistical noise. It does **not** mean that the physical
minimum is unstable.

## Limitations

The discrimination study uses idealised simulated decay coordinates. It does
not yet include realistic diphoton vertex resolution, photon-separation or
reconstruction efficiencies, backgrounds, or detector systematic uncertainties.
The spatial result should therefore be interpreted as the information available
in the accepted truth-level kinematics.

## Code structure

The public analysis interface is organised as:

```text
alp_discrimination/
├── conditional_features.py     feature moments and likelihood ingredients
├── lifetime_template_banks.py  lifetime-dependent accepted templates
├── progress.py                 progress and ETA reporting
├── report_plots.py             report-facing figures
└── workflows/
    └── analysis.py             single user-facing analysis entry point
```

## Tests

```bash
python -m pytest alp_discrimination/tests -q
```

At the cleanup checkpoint used for the final production analysis, the suite gives
`201 passed, 2 skipped`.

## EventCalc core

The underlying EventCalc-SHiP sampler and detector geometry are inherited from
the main EventCalc-SHiP project. See `DETAILS.md` for the general LLP sampling,
decay and event-rate implementation.
