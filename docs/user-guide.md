# Running EventCalc-SHiP

EventCalc-SHiP generates decays of long-lived particles (LLPs) in the decay
volume of the [SHiP experiment](https://ship.web.cern.ch/) and calculates the
expected number of decays. The same calculation can be adapted to another
geometry through `funcs/ship_setup.py`.

The generated event record contains truth-level momenta and decay positions.
[FairShip](https://github.com/ShipSoft/FairShip) can use this record for
particle transport and detector reconstruction; EventCalc itself applies no
detector response.

## Physical calculation

For calibrated dark photons and inelastic DM, both launchers sample the compact
production tables described below. The correlated source is an explicit option. For the other tabulated sources, at each
selected mass and proper decay length $c\tau$, EventCalc

1. interpolates the production probability and the angular and energy density of
   the long-lived particle;
2. samples momenta directed toward the SHiP decay volume, using the tabulated
   dependence of the maximum LLP energy on mass and polar angle;
3. samples decay positions from the exponential decay law within the
   longitudinal extent of the volume;
4. applies the azimuthal geometry of the volume;
5. generates the rest-frame decay, using a channel-specific matrix element or phase space
   as described in the [sampling guide](DETAILS.md#propagation-and-decays);
6. hadronizes and decays what needs it, either with the exHad generator or
   with unmodified [Pythia8](https://pythia.org/) (see
   [Hadronic decays](#hadronic-decays));
7. boosts the final-state particles to the laboratory frame;
8. combines production, geometry, decay probability and the visible branching
   ratio into the number of decays.

The models are named by the launcher as `Scalar-mixing`, `Scalar-quartic`,
`ALP-photon`, `ALP-fermion`, `Dark-photons`, `Inelastic-DM`, `HNL`, `ALP-SU2L` and
`ALP-mixed`. The production and decay inputs of each one are described in
[`DETAILS.md`](DETAILS.md); the conventions a user has to know in order to
read the output are in the [model guide](models.md).

The [model guide](models.md) defines the interactions, couplings and scalar
width choices. [Production distributions](PRODUCTION_DATA.md) describes the
calibrated target calculation and compact sampling; [inelastic dark matter](INELASTIC_DM.md)
explains mediator-mass inputs and excited-state lifetimes.

## Installation

`Inelastic-DM` requires exHad with its inelastic decay module. Follow the
[exHad installation guide](EXHAD.md). Full production samples for correlated
positions and uncertainty scans have a separate [EOS installation](PRODUCTION_DATA.md#full-dataset-on-eos).

The code runs on Linux and macOS. Create an environment and install the
declared dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

This installs NumPy, pandas, SciPy, SymPy and Numba for event generation,
Matplotlib and Plotly for the plotting and event-display scripts, and the
Pythia8 Python binding. HepMC3 export from the runtime generator additionally
needs `pyhepmc`.

[Pythia8](https://pythia.org/) is used for decay channels that contain
partons or unstable particles. Channels whose products are all stable, such as
the diphoton decays of `ALP-photon`, `ALP-SU2L` and `ALP-mixed`, run without
it. EventCalc accepts an installed Pythia8 binding with matching XML data;
it rejects a binding paired with data from another version.
`requirements.txt` pins `pythia8mc==8.317.1`, which installs Pythia8.317 and
its matching XML data. Use 8.317 for comparisons with the current exHad
release, so that a version difference does not change the comparison. The CERN LCG
build exports the same API under the module name `pythia8`, which EventCalc
also accepts. For a custom build, point `PYTHIA8_LIB` at the directory
containing `pythia8mc` or `pythia8` and `PYTHIA8DATA` at that installation's
`share/Pythia8/xmldoc`. To check what was found:

```bash
python3 -c 'from funcs import decayProducts as d; p=d.load_pythia8().Pythia8(d._PYTHIA8_XML or "",False); print("Pythia8", p.settings.parm("Pythia:versionNumber"), "XML", d._PYTHIA8_XML)'
```

If Pythia8 rejects a numerically inconsistent hadronization record, EventCalc
discards that record and retries the identical partonic event, letting only
Pythia8's random state advance; it does not reselect the decay channel or
resample the phase space. Ten unsuccessful attempts are a fatal error.

The exHad generator is a separate package with its own compiled worker. Its
installation, and how EventCalc is pointed at it, are described in
[EXHAD.md](EXHAD.md).

## Running a simulation

EventCalc accepts interactive prompts, a JSON launch card and explicit
command-line arguments. The last two set the physical and numerical choices in advance and start
without prompts.

### Interactive

```bash
python3 simulate.py
```

prints the SHiP geometry and then asks for the number of events to sample in
the polar range of the experiment, the model, the model-specific parameters
(HNL mixing pattern, dark-photon uncertainty and production source,
inelastic-DM splitting, ALP production source, scalar prescription, ALP-mixed $\xi$ and interference),
the decay channels, the masses and the proper decay lengths. For models with compact phenomenology tables it writes the production-probability,
lifetime and branching-ratio figures before the scan starts. It does not run
a separate mass scan for inelastic-DM phenomenology plots.

### JSON launch card

The example launches the $SU(2)_L$ ALP for two masses and two lifetimes; each
lifetime is evaluated at each mass.

```json
{
  "model": "ALP-SU2L",
  "events": 200000,
  "masses": [0.3, 1.0],
  "c_taus": [0.01, 10000.0],
  "decay_channels": ["2gamma"],
  "seed": 12345,
  "plots": false,
  "export_events": true
}
```

```bash
python3 simulate.py --card cards/alp_su2l.json
```

Check the card, resolve its decay channels and print the normalized
configuration without running the simulation:

```bash
python3 simulate.py --card cards/alp_su2l.json --validate-only
```

The required fields are `model`, `events`, `masses` and one lifetime or coupling grid. A flat
`c_taus` list is used for every mass; a nested list gives one lifetime list per
mass. The optional common fields are

- `decay_channels` (see [Decay channels](#decay-channels); the default is
  `["all"]`);
- `seed`, an integer between 0 and $2^{32}-1$;
- `hadronization`, `"exhad"` or `"rawPythia"`, which sets the same switch as
  `--exhad` (see [Hadronic decays](#hadronic-decays));
- `plots`, default `false` in silent mode;
- `export_events`, default `true`;
- `n_pot`, default $6\times10^{20}$ protons on target;
- `min_events_threshold`, default $0.1$ expected decays;
- `scalar_prescription`, for the two scalar models.

Three models require one additional field: `HNL` requires `mixing_pattern`
with $\xi_e$, $\xi_\mu$ and $\xi_\tau$; `ALP-photon` requires
`alp_production_mode`. `ALP-mixed` requires `xi` and `interference`:

```json
{
  "model": "ALP-mixed",
  "xi": 0.35,
  "interference": "constructive",
  "events": 200000,
  "masses": [0.3, 0.5, 1.0],
  "c_taus": [0.01, 100.0, 10000.0],
  "decay_channels": ["2gamma"],
  "seed": 12345
}
```

`ALP-SU2L` already contains its $B$, charged-kaon, primary-photon and
cascade-photon contributions, so its card carries no production-source,
uncertainty or mixing field.

### Explicit command

```bash
python3 simulate.py \
  --model ALP-SU2L \
  --events 200000 \
  --masses 0.3 1.0 \
  --c-taus 0.01 10000 \
  --decay-channels 2gamma \
  --seed 12345
```

Command-line arguments override the fields of a card. `python3 simulate.py
--help` lists every option. The mixed example above is launched with
`--model ALP-mixed --xi 0.35 --interference constructive`.

### Scans without prompts

`run_batch.py` runs the same generation loop over a grid of masses and
lifetimes, taking every choice from the command line:

```bash
# HNL, pure electron mixing, five masses, one lifetime
python3 run_batch.py --llp HNL --mixing 1 0 0 \
    --mass-range 1.0 3.0 0.5 --ctaus 10 \
    --channels jets --nevents 100000 --exhad on

# dark photon, central flux, primary production, logarithmic lifetime grid
python3 run_batch.py --llp Dark-photons --masses 2.0 2.5 \
    --uncertainty central --dp-production primary \
    --ctau-logrange 0.1 100 5 --nevents 50000

# dark photon, primary and cascade production together
python3 run_batch.py --llp Dark-photons --masses 1.8 \
    --ctaus 10 --nevents 50000 --uncertainty central \
    --dp-production combined

# Higgs-like scalar on the unmodified Pythia8 baseline
python3 run_batch.py --llp Scalar-mixing --masses 2.2 --ctaus 5 \
    --nevents 50000 --exhad off
```

Masses and lifetimes are given as explicit lists (`--masses`, `--ctaus`) or as
ranges (`--mass-range`, `--ctau-range`, `--ctau-logrange`). `--dry-run` prints
the generator choice and the output tag of every requested mass without generating
anything; `--plots` also writes the phenomenology figures. `--batch-index-offset`
shifts the point index a scan starts from, so that several jobs of one grid
keep distinct random streams. `python3 run_batch.py --help` lists every option.
Outputs land in the same place as an interactive run.

### Specify a coupling

Use `--couplings` for coupling magnitudes $g$, or `--couplings-squared` for
values read from a plot of $g^2$. These replace, and cannot be combined with,
the lifetime options. EventCalc calculates $c\tau(m,g)=c\tau_1(m)/g^2$ at
**each mass**, using the selected model's installed lifetime table. The production rate uses the coupling implied by that lifetime.

```bash
# Photon ALP: g_agammagamma in GeV^-1
python3 run_batch.py --llp ALP-photon --alp-production primary \
    --masses 0.1 0.3 --couplings 1e-6 3e-6 --nevents 10000

# HNL: total U^2, with Ue^2:Umu^2:Utau^2 = 1:1:0
python3 run_batch.py --llp HNL --mixing 1 1 0 --masses 1 2 \
    --coupling-squared-logrange 1e-10 1e-7 15 --nevents 10000
```

`--coupling-range LO HI STEP` and `--coupling-logrange LO HI N` build grids
in $g$. The corresponding `--coupling-squared-range` and
`--coupling-squared-logrange` build grids in $g^2$ without squaring again.
`--dry-run` prints the coupling convention and the derived lifetimes before
any events are generated. Inputs must be finite and positive.

| Model | Quantity accepted by `--couplings` |
|---|---|
| `ALP-photon` | $|g_{a\gamma\gamma}|$ in GeV$^{-1}$, with $\mathcal L\supset(g_{a\gamma\gamma}/4)aF\widetilde F$ |
| `ALP-fermion` | $|g_Y|=2v_h/f_a$, dimensionless, at the fixed benchmark matching scale |
| `ALP-SU2L` | $|g_W|$ in GeV$^{-1}$, with $C_W=\alpha_2g_W/(4\pi)$ |
| `ALP-mixed` | $|g_{a\gamma\gamma}^{\rm tot}|$ in GeV$^{-1}$, at fixed `xi` and `interference` |
| `Dark-photons`, `Inelastic-DM` | $|\epsilon|$, dimensionless kinetic mixing |
| `Scalar-mixing`, `Scalar-quartic` | $|\sin\theta|$, dimensionless decay mixing |
| `HNL` | $U=\sqrt{U_e^2+U_\mu^2+U_\tau^2}$, with the flavour ratios fixed by `--mixing` |

For `Scalar-quartic`, this input sets the decay lifetime. The production
normalization remains fixed by its yield table.
For HNLs, `--couplings-squared` gives total $U^2$; the normalized mixing
fractions multiply that total to give each flavour's mixing squared.

`simulate.py` accepts the same `--couplings` and `--couplings-squared` lists.
A JSON launch card can replace `c_taus` with either `couplings` or
`couplings_squared`: a flat list applies to every mass; a nested list gives
one list per mass. This supports ranges that follow a sensitivity plot.
[examples/coupling_scan.py](../examples/coupling_scan.py) implements the example
photon-ALP scan, interpolating the chosen lower and upper edges in log-log
space. With `alpha < 1` it concentrates points toward the upper coupling
edge; `alpha > 1` concentrates them toward the lower edge. Requested masses must lie inside the defined interval.

From Python, after selecting the mass on an `LLP` object, use
`LLP.set_coupling(g)` or `LLP.set_coupling(g_squared, squared=True)`. The
method returns and sets $c\tau$ in metres, using the lifetime coefficient
evaluated at the selected mass.

A scan reuses one exHad event pool for every lifetime at one mass: the
rest-frame decay does not depend on $c\tau$. `simulate.py` does the same.

### Decay channels

A channel is selected by name, by `all`, or by `jets`. Channel names are case-sensitive. `jets` (with the alias
`all-jets`) is the single aggregate choice that expands to every `Jets-*` row
of the model's decay table; an individual `Jets-*` name cannot be selected,
because those rows are the partonic decomposition of one inclusive rate and
the branching vector is renormalized over the selected subset downstream.
`all`, the default, selects every row.

The interactive prompt prints the collapsed menu, in which the `Jets-*` rows
appear as one entry, and accepts the numbers of that menu. A number in a card
or on the command line is refused with a message naming both possible
meanings, so that a stored selection cannot silently change identity: write
channel names, `all` or `jets` there.

### Hadronic decays

The hadrons of a decay into quarks or gluons come either from unmodified
Pythia8 or from the exHad generator, which matches the composition of the final
states to exclusive calculations and to measured $e^+e^-$ cross sections.
`--exhad` selects between them in every entry point:

- `--exhad auto`, the default, follows the per-LLP card and the environment,
  and runs on the Pythia8 baseline when no usable exHad installation is
  configured;
- `--exhad on` (or a bare `--exhad`) requires a usable exHad installation and
  stops the run when the selection has a matched route and none is available;
- `--exhad off` (aliases `--rawPythia` and `--raw-pythia`) forces the Pythia8
  baseline for hadronic decays. Nonhadronic channels are generated directly
  and need no such flag.

The card key `hadronization` sets the same switch, and an explicit flag
overrides the card. The route that was actually used is recorded in the output
file name and in a metadata file beside it.

The [hadronic generation guide](EXHAD.md#model-coverage) gives the model coverage
and explains how decay rates and hadronic momenta are combined.

For large samples on the Pythia8 baseline, set `PYTHIA8_N_WORKERS` to an
integer or to `auto` to hadronize the chunks of one block in parallel; the
default is 1, which takes the sequential path and reproduces from the seed
alone. At a fixed seed and chunk size, the worker count leaves the events unchanged. A block is cut into chunks of `PYTHIA8_CHUNK_SIZE` events
(default 1000) and each chunk runs on the stream its index names, so one seed
gives one set of events whatever the worker count. Event files are written in
chunks of `EVENT_WRITE_CHUNK_SIZE` rows (default 10000).

### Python runtime generator

For `Dark-photons` and `Inelastic-DM`, `eventcalc.runtime_generator` samples
the compact tables by default. `production_backend: correlated` opts into the
full EOS source and preserves production origins in each batch. The interface
initializes the chosen model once, constructs a two-dimensional inverse CDF
for the tabulated $(\theta,E)$ density, and then offers `next()`,
`next_event()`, `generate()`, `yields()` and `stat()`. It works for every
model a card can select, including `ALP-mixed`.

The runtime generator samples the precomputed cumulative distribution
function (CDF) directly: the energy coordinate is mapped to a unit interval
between the lifetime-dependent lower bound and $E_{\max}(\theta)$, and within
each CDF cell the piecewise-bilinear density is inverted analytically. The
`simulate.py` workflow keeps its proposal-and-resampling algorithm.

The fiducial volume comes from a separate experiment card, by default
`cards/ship.json`:

```bash
python3 -m eventcalc.runtime_generator \
  --card cards/alp_su2l.json \
  --experiment-card cards/ship.json \
  --mass 0.3 \
  --ctau 100 \
  --events 1000000 \
  --batch-size 10000 \
  --mode fiducial \
  --output-format npz \
  --output-dir outputs/runtime-alp-su2l-ma0p3-ctau100
```

For table-based models other than calibrated DP/iDM, `fiducial` mode returns
`events` decays inside the volume. The calibrated DP/iDM source interface returns
`events` proposals; trajectories missing the volume carry zero decay weight. In `attempted` mode every sampled parent is returned, `inside_volume`
marks the trajectories inside the volume and `decay_weights` is zero outside
it. Each batch is a compressed NumPy archive holding these arrays together
with the standard event rows and the decay-channel labels, and `metadata.json`
records the resolved configuration and the running yield estimate.
`--output-format hepmc3` writes one HepMC3 ASCII stream instead, and `both`
writes the NumPy batches as well. HepMC events use GeV and mm, carry the
displaced decay vertex, the incoming LLP with its own PDG code and the
final-state particles, and carry `decay_weight` as the named event weight.
For full sources, HepMC also carries the correlated production coordinates
as `eventcalc_production_x_mm`, `eventcalc_production_y_mm` and
`eventcalc_production_z_mm` attributes. Its clock starts at that collision
because absolute shower times are not stored.

The API performs no per-event file input or output:

```python
from pathlib import Path
from eventcalc.runtime_generator import generator_from_card

generator = generator_from_card(
    Path("cards/alp_su2l.json"),
    experiment_card=Path("cards/ship.json"),
    mass=0.3,
    c_tau=100.0,
    mode="fiducial",
    seed=12345,
)
generator.init()
for _ in range(1_000_000):
    generator.next()
    analyze(generator.current_event)

summary = generator.yields()
```

For several masses or lifetimes, load one scan context and compile the points
before generating:

```python
from pathlib import Path
from eventcalc.runtime_generator import scan_from_card

points = [(0.3, 0.01), (0.3, 100.0), (0.5, 100.0)]
scan = scan_from_card(
    Path("cards/alp_su2l.json"),
    experiment_card=Path("cards/ship.json"),
    seed=12345,
)
scan.compile_points(points)

for mass, c_tau in points:
    generator = scan.generator(mass=mass, c_tau=c_tau, mode="fiducial")
    for batch in generator.generate(1_000_000):
        analyze(batch)
```

The model tables and the dense interpolation grid are built once per scan.
Each lifetime keeps its own cached CDF, because the survival-energy cutoff
depends on the lifetime; building that CDF takes a few milliseconds.
`yields()` combines the production normalization and the visible branching
ratio with the CDF integral and the running Monte Carlo estimates of
transverse acceptance and decay probability, and reports the statistical
errors of the sampled factors. Parent kinematics and decay vertices use
separate Philox streams and do not change when the batch partition changes.

## Event yields and output

For the other tabulated sources, EventCalc samples `events * 10` interpolation
candidates and resamples `events` momenta inside the polar range. The
azimuthal selection then keeps a fraction of order 0.6 to 1 of them for the
current geometry, which is why an output file holds fewer decays than the
number requested.

A mass and lifetime point whose production probability per proton on target is
below $10^{-21}$ is skipped. A point whose expected number of decays is below
`min_events_threshold` is written to the totals file, and its decay products
are not generated.

Results are written under `outputs/<model>/` in the repository, whatever
directory the run was started from:

- `eventData/` holds the event records when event export is enabled, as
  `<model>_<mass>_<lifetime>_<parameters>_<route>_<channels>_data.dat`;
- `total/` holds one row per mass and lifetime point with the mass, the squared
  coupling, the lifetime, the production yield, the polar and azimuthal
  acceptance, the mean decay probability, the visible branching ratio and the
  expected number of decays.

The file name carries the model parameters (the HNL mixing pattern, the
dark-photon uncertainty and production source, the ALP production source, the
scalar prescription, the ALP-mixed $\xi$ and interference sign), the route
(`eventcalc-direct-seed<N>` when every selected row with positive rate is
already stable, `raw-pythia-seed<N>` when at least one needs Pythia8, and
`exhad-<benchmark>-seed<N>` when a selected row with positive rate is matched)
and the channel selection (`channels-all`, a readable subset, or a digest of a
very long subset). Rerunning one coordinate replaces its row in the totals file. Each output file is accompanied by
`<file>.generator.json`, which records the route and its inputs,
for `ALP-fermion`, `<file>.coupling.json`, which records the coupling convention, and, for a
matched run, `<file>.hadronization.json`. Event files are written to a
temporary file in the same directory and published atomically; the totals file
is locked per file and replaced atomically, so concurrent jobs cannot lose
rows or expose a half-written file.

For `ALP-mixed` the `coupling_squared` column is
$|g_{a\gamma\gamma}^{\rm tot}|^2$ in $\mathrm{GeV}^{-2}$; for `ALP-fermion` it
is $g_Y^2$.

An event record begins with a header line giving the sample size, the squared
coupling, the total number of produced LLPs, the polar and azimuthal
acceptance, the mean decay probability, the visible branching ratio of the
selected channels, the expected number of decays and the channel selection.
The events then follow in blocks, each opened by
`#<process=...; sample_points=...>` naming the decay channel and the number of
events in the block. A pooled exHad block is labelled
`Jets-matched:<row>`, one label per table row it contains. Each event row is

```text
p_x,LLP p_y,LLP p_z,LLP E_LLP mass_LLP PDG_LLP P_decay,LLP x_decay,LLP y_decay,LLP z_decay,LLP p_x,prod1 p_y,prod1 p_z,prod1 E_prod1 mass_prod1 pdg_prod1 p_x,prod2 ...
```

The first ten numbers describe the LLP: its four-momentum, mass, identifier,
decay probability and decay position. The identifier column of this file holds
the physical mother PDG for the full collision source (4900022 for DP,
4900102 for chi-prime), and the placeholder 12345678 for the other tabulated sources; the HepMC3 output of the runtime generator carries
the LLP's own PDG code. Each further group of six describes one
final-state particle: four-momentum, mass and PDG code. Momenta, energies and
masses are in GeV and positions are in metres, with the origin at the front face
of the SHiP target. Rows of a channel with fewer products are padded with
`0. 0. 0. 0. 0. -999.` so that one channel's events form a rectangular array.
The weight of an event is its `P_decay,LLP`; the expected number of decays is
the product of the total number of produced LLPs, the polar and azimuthal
acceptance, the mean of `P_decay,LLP` and the visible branching ratio, all of
which stand in the header.

For full collision sources, column 6 is the **signed** decay probability.
For the compact source, `Polar acceptance` is the integral in the enclosing
angular cone; `Azimuthal acceptance` is the fraction of sampled rays intersecting
the volume. For the correlated source, `Polar acceptance` is the absolute weight
of intersecting rays divided by the signed production integral, and
`Azimuthal acceptance` is one in fiducial mode. The
header product above still gives the physical yield; do not drop negative rows.
Each text event file has a `.production.npz` companion, in the same row order,
with the collision origins, production momenta, categories, mechanisms and
normalized signed event weights per incident proton. Multiplying the sum of
`event_weights_per_pot` by `n_pot` reproduces the reported yield. The
`.production.json` companion records the target, source seed, integration
power and normalization. A separate JSON record accompanies each total-table
mass/lifetime coordinate, including points below the export threshold.

With `plots` in a card or `--plots` on the command line, EventCalc writes the
mass dependence of the production probability, of the proper decay length and
of the selected branching ratios under `plots/<model>/phenomenology`.

## Post-processing

These scripts read the files a completed run wrote. Run them as Python modules from the repository root; for example,
`python3 -m eventcalc.analysis.events_analysis`.

- `eventcalc.analysis.events_analysis` computes distributions of the decaying LLP and of its
  decay products; decay vertex, energies, multiplicities, invariant masses —
  and applies a truth-level detector requirement. It projects straight
  downstream trajectories onto a $4\,\mathrm{m}\times6\,\mathrm{m}$ plane at
  $z=95\,\mathrm{m}$, which lies downstream of the decay volume, and counts a
  particle as seen when its PDG code is one of $e^\pm$, $\mu^\pm$, $\pi^\pm$,
  $K^\pm$, $\gamma$ and $K_L$. The signatures are `two-photon` (exactly two
  visible particles, both photons, both crossing the plane), `neutral-pair`
  (at least one subset of two or more crossing particles with zero total
  charge) and `all-visible` (every visible particle crosses the plane), the
  default. `--detector-z`, `--detector-width` and `--detector-height` move the
  plane; `--min-energy` sets a common energy threshold, so that a counted
  particle satisfies $E\geq\max(m,E_{\min})$, with $E_{\min}=0$ by default. The fractions are weighted by the parent decay probability and
  are written to `detector_acceptance.txt`, beside the figures, under
  `plots/<model>/<file>/`. They are geometric: no efficiency, smearing,
  reconstruction or particle identification is applied.
- `eventcalc.analysis.events_pointing_analysis` applies the daughter-level pointing
  requirement; at least two of the same particles crossing the same plane —
  to a whole directory of event files and writes the per-file pointing
  efficiencies and corrected numbers of decays to
  `outputs/<model>/pointing_summary.csv`. It also reports the same efficiency
  with a 1 GeV momentum threshold per counted particle.
- `eventcalc.analysis.analyze_cascade_vs_primary` combines the totals files with that pointing
  summary and compares the dark-photon production sources: the ratio of the
  numbers of decays against mass and lifetime, and the corresponding figures.
- `eventcalc.analysis.plot_event_distributions` draws the polar-angle and energy distributions
  of the dark photons that decay inside the volume, for the primary and the
  cascade sources.
- `eventcalc.analysis.total_plots` draws the averaged quantities of a scan: the polar and
  total geometric acceptance, the mean decay probability, and the number of
  decays against coupling, mass and lifetime.
- `eventcalc.analysis.event_display` draws ten random events of a selected channel, as PDF and
  as an interactive HTML page, showing the decay point, the LLP direction and
  the directions of its decay products.

The two figure scripts write into `plots/` inside the repository; set
`EVENTCALC_FIGURE_DIR` to write them somewhere else.

## Checks against other calculations

EventCalc-SHiP has been compared with
[`SensCalc`](https://github.com/maksymovchynnikov/SensCalc), which was itself
tested against FairShip calculations and other tools. The number of decays,
the geometric acceptance, the mean decay probability and the kinematic
distributions agree at the 10 % level or better; see the
[comparison slides](https://indico.cern.ch/event/1481729/contributions/6256116/).

## Source layout

See [Repository layout and commands](repository-layout.md).

## Credits

This repository builds on [Josue Jaramillo's CERN student
project](https://github.com/josuejaramillo/summer_school_2024_SHiP).

## Perfect-reconstruction sensitivity scans

Use `python -m eventcalc.scan_sensitivity` to scan all lifetimes at one mass while generating
production and decays once. The model-independent API accepts weighted primary
and cascade decays, arbitrary lifetime/coupling laws and configurable energy,
aperture and impact-parameter cuts. See [SENSITIVITY.md](SENSITIVITY.md) for
normalization, model interfaces, compact outputs and examples.
