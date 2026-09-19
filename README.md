# EventCalc-SHiP

EventCalc-SHiP generates decays of long-lived particles (LLPs) in the decay
volume of the [SHiP experiment](https://ship.web.cern.ch/) and calculates the
expected number of decays. The same calculation can be adapted to another
geometry through `funcs/ship_setup.py`.

The generated event record contains truth-level momenta and decay positions.
[FairShip](https://github.com/ShipSoft/FairShip) can use this record for
particle transport and detector reconstruction; EventCalc itself applies no
detector response.

## Physical calculation

For each selected mass and proper decay length $c\tau$, EventCalc

1. interpolates the production probability and the angular–energy density of
   the long-lived particle;
2. samples momenta directed toward the SHiP decay volume, using the tabulated
   dependence of the maximum LLP energy on mass and polar angle;
3. samples decay positions from the exponential decay law within the
   longitudinal extent of the volume;
4. applies the azimuthal geometry of the volume;
5. generates the rest-frame decay: two-body, three-body and four-body decays
   from their matrix elements, and decays with five or more primary products
   from flat phase space;
6. hadronizes and decays what needs it, either with the exHad generator or
   with unmodified [Pythia 8](https://pythia.org/) (see
   [Hadronic decays](#hadronic-decays));
7. boosts the final-state particles to the laboratory frame;
8. combines production, geometry, decay probability and the visible branching
   ratio into the number of decays.

The models are named by the launcher as `Scalar-mixing`, `Scalar-quartic`,
`ALP-photon`, `ALP-fermion`, `Dark-photons`, `HNL`, `ALP-SU2L` and
`ALP-mixed`. The production and decay inputs of each one are described in
[`DETAILS.md`](DETAILS.md); the conventions a user has to know in order to
read the output are below.

## Couplings and production sources

### Heavy neutral lepton

`HNL` takes a mixing pattern $(\xi_e,\xi_\mu,\xi_\tau)$ summing to one, with
$U_\alpha^2=U^2\xi_\alpha$. Production yields, angular–energy densities,
widths, branching ratios and decay matrix elements are tabulated for the three
pure mixings and merged for the requested pattern, as described in
[`DETAILS.md`](DETAILS.md).

### Fermion-coupled ALP

For `ALP-fermion`, EventCalc uses the dimensionless coupling

$$
g_Y \equiv y \equiv \frac{2v_h}{f_a}.
$$

Every `coupling_squared` column, every `Squared coupling` field and every
phenomenology plot of this model refers to $g_Y^2$. The input tables use the
same normalization:
`Distributions/ALP-fermion/ctau-ALP-fermion.txt` stores
$(m_a, c\tau_1)$ with $c\tau_1(m_a)=c\tau(m_a,g_Y=1)$ in metres, and
`Distributions/ALP-fermion/Total-yield-ALP-fermion.txt` stores $(m_a, Y_1)$
with $Y_1=N_{\rm prod}/(N_{\rm POT}g_Y^2)$. Therefore

$$
g_Y^2=\frac{c\tau_1}{c\tau},\qquad
N_{\rm prod}=N_{\rm POT}\,Y_1\,g_Y^2 .
$$

Every partial width scales as $\Gamma_i(m_a,g_Y)=g_Y^2f_i(m_a)$, so branching
ratios do not depend on the coupling. The column-level data dictionary is in
[`Distributions/ALP-fermion/README.md`](Distributions/ALP-fermion/README.md).

### Higgs-like scalar

For `Scalar-mixing` the coupling is the mixing angle, $g_S=\sin\theta$.
Production rates and every partial width scale as $\sin^2\theta$, so

$$
c\tau(m_S,\sin\theta)=\frac{c\tau_1(m_S)}{\sin^2\theta},
\qquad
\sin^2\theta=\frac{c\tau_1(m_S)}{c\tau},
$$

where the tables store $c\tau_1(m_S)=c\tau(m_S,\sin\theta=1)$ in metres. Both
scalar models read one prescription-keyed pair of tables under
`Distributions/Scalar-mixing`: the hadronic widths of
[2407.13587](https://arxiv.org/abs/2407.13587) as the mean of its Monte Carlo
sample and one standard deviation below and above it
(`2407.13587-Central`, `2407.13587-Lower`, `2407.13587-Upper`), and the
calculation of [1809.01876](https://arxiv.org/abs/1809.01876) with the
corrections of [1904.10447](https://arxiv.org/abs/1904.10447)
(`1809.01876`). The default is `2407.13587-Central`; select another with
`--scalar-prescription` or the card key `scalar_prescription`. The data
dictionary is in
[`Distributions/Scalar-mixing/README.md`](Distributions/Scalar-mixing/README.md)
and in
[`Distributions/Scalar-mixing/coupling.json`](Distributions/Scalar-mixing/coupling.json).

### Dark-photon production sources

For `Dark-photons` the production source is chosen independently of the
flux-uncertainty variation (`lower`, `central`, `upper`):

- `primary`: the channels initiated by the first proton–target collision —
  proton bremsstrahlung, mixing with the light vector mesons of the
  fragmentation chain, decays of $\pi^0$, $\eta$ and $\eta'$, and the
  Drell–Yan process. The tables are
  `Distributions/Dark-photons/{DoubleDistr,Emax,Total-yield}-DP-<uncertainty>.txt`,
  generated with `SensCalc` following
  [2409.11096](https://arxiv.org/abs/2409.11096).
- `cascade`: the same channels initiated by the secondary particles of the
  hadronic cascade in the thick SHiP target, from a combined Geant4 and
  Pythia 8 simulation of the tungsten target (about 2.35 secondary inelastic
  proton interactions above a 5 GeV kinetic threshold, about 100 $\pi^0$ and
  6 $\eta$ per proton on target, with the primary interaction excluded so that
  nothing is counted twice). Its four contributions are bremsstrahlung off the
  secondary protons in the quasi-real approximation of
  [2108.05900](https://arxiv.org/abs/2108.05900) and
  [2409.11096](https://arxiv.org/abs/2409.11096); mixing with the
  $\rho^0/\omega/\phi$ mesons of the secondary collisions following
  [2504.06828](https://arxiv.org/abs/2504.06828); Drell–Yan
  $q\bar q\to V$ off the secondary protons; and the radiative decays
  $\pi^0,\eta\to\gamma V$, which dominate below $m_V\simeq0.6$ GeV. The merged
  tables carry the label `cascade-<uncertainty>`. Their angular–energy
  distribution is normalized over all directions, so the polar acceptance
  EventCalc computes accounts for the large fraction of cascade dark photons
  that miss the decay volume.
- `brem-cascade`: the bremsstrahlung contribution of the cascade on its own.
- `combined`: the sum of `primary` and `cascade`. EventCalc keeps both native
  kinematic grids, sums the absolute production yields, and samples the
  accepted events of each source with a weight proportional to
  yield $\times$ polar acceptance.

The `lower`/`central`/`upper` variations act on the bremsstrahlung and
Drell–Yan components (the proton virtuality scale and the timelike form
factors for the first, the factorization scale for the second). The
fragmentation and meson-decay components enter all three variations
identically.

`ALP-photon` takes the same three-way choice between `primary`, `cascades`
and `combined` for its Primakoff production.

### $SU(2)_L$ ALP

The `ALP-SU2L` model uses

$$
\mathcal L\supset C_WaW^I_{\mu\nu}\widetilde W^{I\mu\nu},
\qquad
C_W=\frac{\alpha_2}{4\pi}g_W,
\qquad
\frac{c_W}{f_a}=-C_W,
$$

with $c_B=c_{a\Phi}=0$. The last relation converts to the convention of
[1901.02031](https://arxiv.org/abs/1901.02031), where the operator is
$-c_WaW^I_{\mu\nu}\widetilde W^{I\mu\nu}/f_a$. Electroweak symmetry breaking
induces

$$
C_\gamma^{\rm ind}=\sin^2\theta_W C_W,
\qquad
g_{a\gamma\gamma}=\frac{\alpha_{\rm em}}{\pi}g_W.
$$

Four processes contribute to the ALP flux: rare $B\to X_{s,d}a$ decays,
$K^\pm\to\pi^\pm a$ below the charged-kaon threshold, Primakoff production by
photons from the proton–target interaction, and Primakoff production by
photons in the electromagnetic cascade. Let $\widehat Y_B$ and $\widehat Y_K$
denote the production coefficients per $g_W^2$, and $\widehat Y_{\rm primary}$
and $\widehat Y_{\rm cascade}$ the Primakoff coefficients per
$g_{a\gamma\gamma}^2$. Since $g_{a\gamma\gamma}=kg_W$ with
$k=\alpha_{\rm em}/\pi$, the combined coefficient per $g_W^2$ is

$$
\widehat Y_{\rm tot}=\widehat Y_B+\widehat Y_K
+k^2(\widehat Y_{\rm primary}+\widehat Y_{\rm cascade}),
$$

and the production probability is $P_{\rm prod}=g_W^2\widehat Y_{\rm tot}$.
The normalized angular–energy density is

$$
f_{\rm tot}=
\frac{\widehat Y_Bf_B+\widehat Y_Kf_K
+k^2\widehat Y_{\rm primary}f_{\rm primary}
+k^2\widehat Y_{\rm cascade}f_{\rm cascade}}
{\widehat Y_{\rm tot}},
$$

which EventCalc reads as one production model. The charged-kaon flux is
$0.36=0.29+0.07$ kaons per proton on target, summed over the two charges; its
branching coefficient is evaluated as a function of $m_a$ and vanishes exactly
at $m_a=m_{K^\pm}-m_{\pi^\pm}$.

For a requested $c\tau$, EventCalc obtains the coupling from

$$
g_W^2=(1\,\mathrm{GeV}^{-1})^2
\frac{(c\tau)_{g_W=1\,\mathrm{GeV}^{-1}}}{c\tau},
$$

and multiplies $\widehat Y_{\rm tot}$ by its numerical value in
$\mathrm{GeV}^{-2}$. The decay model of this benchmark is
$\mathrm{Br}(a\to\gamma\gamma)=1$; hadronic ALP widths need a different decay
description. Every production, lifetime and decay table this benchmark needs
is installed under `Distributions/ALP-SU2L/`.

### Mixed photon–$SU(2)_L$ ALP

The `ALP-mixed` model implements both operators coherently. Its two additional
inputs are the operator fraction $0\leq\xi\leq1$ and the interference sign,
`constructive` or `destructive`. The convention is

$$
C_W=\Lambda\xi,
\qquad
C_\gamma^{\rm dir}=s\Lambda(1-\xi),
\qquad s=\pm1,
$$

so that

$$
g_{a\gamma\gamma}^{\rm tot}
=4\Lambda\left[s(1-\xi)+\sin^2\!\theta_W\,\xi\right]
=g_{a\gamma\gamma}^{\rm dir}+\frac{\alpha_{\rm em}}{\pi}g_W .
$$

The amplitudes interfere before the width is calculated, so the lifetime is
not a sum of two coupling squares. For the diphoton benchmark,

$$
c\tau=\frac{64\pi\hbar c}
{m_a^3\left|g_{a\gamma\gamma}^{\rm tot}\right|^2},
$$

from which EventCalc infers $|g_{a\gamma\gamma}^{\rm tot}|^2$ for a requested
$c\tau$. At fixed $\xi$ and sign, the relative weight of the
flavour-changing $B$ and charged-kaon production against the primary- and
cascade-photon production is

$$
\frac{g_W^2}{|g_{a\gamma\gamma}^{\rm tot}|^2}
=\frac{(4\pi/\alpha_2)^2\xi^2}
{16\left[s(1-\xi)+\sin^2\!\theta_W\,\xi\right]^2}.
$$

No spectrum has to be generated at launch: the $B+K$ component is
reconstructed from the installed `ALP-SU2L` table and the installed
primary- and cascade-photon tables, so $\xi=0$ reproduces the combined
photophilic production and $\xi=1$ reproduces the pure `ALP-SU2L` physics. The
common tabulated mass range is 0.02–4 GeV. The exact destructive
cancellation point is refused, because a diphoton-only model has no finite
signal there.

## Installation

The code runs on Linux and macOS. Create an environment and install the
declared dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

This installs NumPy, pandas, SciPy, SymPy and Numba for event generation,
Matplotlib and Plotly for the plotting and event-display scripts, and the
Pythia 8 Python binding. HepMC3 export from the runtime generator additionally
needs `pyhepmc`.

[Pythia 8](https://pythia.org/) is used for decay channels that contain
partons or unstable particles. Channels whose products are all stable, such as
the diphoton decays of `ALP-photon`, `ALP-SU2L` and `ALP-mixed`, run without
it. EventCalc accepts an installed Pythia binding with matching XML data;
it rejects a binding paired with data from another version.
`requirements.txt` pins `pythia8mc==8.317.1`, which installs Pythia 8.317 and
its matching XML data. Use 8.317 for comparisons with the current exHad
release, so that a version difference does not change the comparison. The CERN LCG
build exports the same API under the module name `pythia8`, which EventCalc
also accepts. For a custom build, point `PYTHIA8_LIB` at the directory
containing `pythia8mc` or `pythia8` and `PYTHIA8DATA` at that installation's
`share/Pythia8/xmldoc`. To check what was found:

```bash
python3 -c 'from funcs import decayProducts as d; p=d.load_pythia8().Pythia(d._PYTHIA8_XML or "",False); print("Pythia", p.settings.parm("Pythia:versionNumber"), "XML", d._PYTHIA8_XML)'
```

If Pythia rejects a numerically inconsistent hadronization record, EventCalc
discards that record and retries the identical partonic event, letting only
Pythia's random state advance; it does not reselect the decay channel or
resample the phase space. Ten unsuccessful attempts are a fatal error.

The exHad generator is a separate package with its own compiled worker. Its
installation, and how EventCalc is pointed at it, are described in
[EXHAD.md](EXHAD.md).

## Running a simulation

EventCalc accepts interactive prompts, a JSON launch card and explicit
command-line arguments. The last two are silent: every physical and numerical
choice is supplied before the calculation starts, no prompt is called, and
Matplotlib runs on its non-interactive `Agg` backend.

### Interactive

```bash
python3 simulate.py
```

prints the SHiP geometry and then asks for the number of events to sample in
the polar range of the experiment, the model, the model-specific parameters
(HNL mixing pattern, dark-photon uncertainty and production source,
ALP production source, scalar prescription, ALP-mixed $\xi$ and interference),
the decay channels, the masses and the proper decay lengths. It writes the
production-probability, lifetime and branching-ratio figures before the scan
starts.

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

The required fields are `model`, `events`, `masses` and `c_taus`. A flat
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
with $\xi_e$, $\xi_\mu$ and $\xi_\tau$; `Dark-photons` requires `uncertainty`,
one of `lower`, `central` and `upper`; `ALP-photon` requires
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

# Higgs-like scalar on the unmodified-Pythia baseline
python3 run_batch.py --llp Scalar-mixing --masses 2.2 --ctaus 5 \
    --nevents 50000 --exhad off
```

Masses and lifetimes are given as explicit lists (`--masses`, `--ctaus`) or as
ranges (`--mass-range`, `--ctau-range`, `--ctau-logrange`). `--dry-run` prints
the route and the output tag of every requested mass without generating
anything; `--plots` also writes the phenomenology figures. `--batch-index-offset`
shifts the point index a scan starts from, so that several jobs of one grid
keep distinct random streams. `python3 run_batch.py --help` lists every option.
Outputs land in the same place as an interactive run.

A scan reuses one exHad event pool for every lifetime at one mass: the
rest-frame decay does not depend on $c\tau$. `simulate.py` does the same.

### Decay channels

A channel is selected by name, by `all`, or by `jets`. `jets` (with the alias
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
Pythia or from the exHad generator, which matches the composition of the final
states to exclusive calculations and to measured $e^+e^-$ cross sections.
`--exhad` selects between them in every entry point:

- `--exhad auto`, the default, follows the per-LLP card and the environment,
  and runs on the Pythia baseline when no usable exHad installation is
  configured;
- `--exhad on` (or a bare `--exhad`) requires a usable exHad installation and
  stops the run when the selection has a matched route and none is available;
- `--exhad off` (aliases `--rawPythia` and `--raw-pythia`) forces the Pythia
  baseline.

The card key `hadronization` sets the same switch, and an explicit flag
overrides the card. The route that was actually used is recorded in the output
file name and in a sidecar, so the two never have to be told apart by which
directory they landed in.

The installed exHad cards cover:

| Model | exHad mass support | What exHad replaces |
|---|---:|---|
| `Dark-photons` | 1.70–5.00 GeV | The selected $u\bar u$, $d\bar d$, $s\bar s$ and $c\bar c$ rows are one pool: their summed event count is generated once as complete hadronic final states |
| `ALP-fermion` | 1.911–5.00 GeV | All selected hadronic rows, including the nucleon-pair rows, are one pool. Leptonic and diphoton rows are unchanged |
| `Scalar-mixing`, `Scalar-quartic` | 2.00–63.0 GeV (the SHiP production tables end at 5.12 GeV and bound a run) | All selected hadronic rows are one pool; the prescription selected for the rates selects the matching exHad input |
| `HNL` | 0.02–40 GeV (the SHiP production tables end at 5.27 GeV) | The hadrons of a selected current, at the invariant mass $W$ of the quark–antiquark pair of each event. The charged lepton or neutrino, the sampled four-momentum and $W$ are kept |

Below those supports the hadronic rows are generated as exact exclusive final
states. The branching ratios, the lifetimes and the channel selection always
remain EventCalc's: exHad changes the composition and the kinematics of the
hadronic events, never their number. [EXHAD.md](EXHAD.md) describes the
installation, the per-LLP cards and the coverage in detail.

For large samples on the Pythia baseline, set `PYTHIA8_N_WORKERS` to an
integer or to `auto` to hadronize the chunks of one block in parallel; the
default is 1, which takes the sequential path and reproduces from the seed
alone. Above one worker the events of a block depend on the worker count. A block is cut into chunks of `PYTHIA8_CHUNK_SIZE` events
(default 1000) and each chunk runs on the stream its index names, so one seed
gives one set of events whatever the worker count. Event files are written in
chunks of `EVENT_WRITE_CHUNK_SIZE` rows (default 10000).

### Python runtime generator

`runtime_generator.py` is an event-by-event interface over the same tables. It
initializes the chosen model once, constructs a two-dimensional inverse CDF
for the tabulated $(\theta,E)$ density, and then offers `next()`,
`next_event()`, `generate()`, `yields()` and `stat()`. It works for every
model a card can select, including `ALP-mixed`.

The runtime generator samples the precomputed CDF directly instead of
proposing and resampling: the energy coordinate is mapped to a unit interval
between the lifetime-dependent lower bound and $E_{\max}(\theta)$, and within
each CDF cell the piecewise-bilinear density is inverted analytically. The
`simulate.py` workflow keeps its proposal-and-resampling algorithm.

The fiducial volume comes from a separate experiment card, by default
`cards/ship.json`:

```bash
python3 runtime_generator.py \
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

In `fiducial` mode, `events` is the number of returned decays inside the
volume. In `attempted` mode every sampled parent is returned, `inside_volume`
marks the trajectories inside the volume and `decay_weights` is zero outside
it. Each batch is a compressed NumPy archive holding these arrays together
with the standard event rows and the decay-channel labels, and `metadata.json`
records the resolved configuration and the running yield estimate.
`--output-format hepmc3` writes one HepMC3 ASCII stream instead, and `both`
writes the NumPy batches as well. HepMC events use GeV and mm, carry the
displaced decay vertex, the incoming LLP with its own PDG code and the
final-state particles, and carry `decay_weight` as the named event weight.

The API performs no per-event file input or output:

```python
from pathlib import Path
from runtime_generator import generator_from_card

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
from runtime_generator import scan_from_card

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

For each mass and lifetime, EventCalc samples `events * 10` interpolation
candidates and resamples `events` momenta inside the polar range. The
azimuthal selection then keeps a fraction of order 0.6–1 of them for the
current geometry, which is why an output file holds fewer decays than the
number requested.

A mass–lifetime point whose production probability per proton on target is
below $10^{-21}$ is skipped. A point whose expected number of decays is below
`min_events_threshold` is written to the totals file, and its decay products
are not generated.

Results are written under `outputs/<model>/` in the repository, whatever
directory the run was started from:

- `eventData/` holds the event records when event export is enabled, as
  `<model>_<mass>_<lifetime>_<parameters>_<route>_<channels>_data.dat`;
- `total/` holds one row per mass–lifetime point with the mass, the squared
  coupling, the lifetime, the production yield, the polar and azimuthal
  acceptance, the mean decay probability, the visible branching ratio and the
  expected number of decays.

The file name carries the model parameters (the HNL mixing pattern, the
dark-photon uncertainty and production source, the ALP production source, the
scalar prescription, the ALP-mixed $\xi$ and interference sign), the route
(`eventcalc-direct-seed<N>` when every selected row with positive rate is
already stable, `raw-pythia-seed<N>` when at least one needs Pythia, and
`exhad-<benchmark>-seed<N>` when a selected row with positive rate is matched)
and the channel selection (`channels-all`, a readable subset, or a digest of a
very long subset). Rerunning one coordinate replaces its row in the totals
file instead of appending a second one. Each output file is accompanied by
`<file>.generator.json`, which records the route and its inputs,
`<file>.coupling.json`, which records the coupling convention, and, for a
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
the placeholder 12345678; the HepMC3 output of the runtime generator carries
the LLP's own PDG code. Each further group of six describes one
final-state particle: four-momentum, mass and PDG code. Momenta, energies and
masses are in GeV and positions are in metres, with the origin at the centre
of the SHiP target. Rows of a channel with fewer products are padded with
`0. 0. 0. 0. 0. -999.` so that one channel's events form a rectangular array.
The weight of an event is its `P_decay,LLP`; the expected number of decays is
the product of the total number of produced LLPs, the polar and azimuthal
acceptance, the mean of `P_decay,LLP` and the visible branching ratio, all of
which stand in the header.

With `plots` in a card or `--plots` on the command line, EventCalc writes the
mass dependence of the production probability, of the proper decay length and
of the selected branching ratios under `plots/<model>/phenomenology`.

## Post-processing

These scripts read the files a completed run wrote. They are launched with
`python3 <script>.py`.

- `events_analysis.py` computes distributions of the decaying LLP and of its
  decay products — decay vertex, energies, multiplicities, invariant masses —
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
  particle satisfies $E\geq\max(m,E_{\min})$ instead of the default
  $E\geq m$. The fractions are weighted by the parent decay probability and
  are written to `detector_acceptance.txt`, beside the figures, under
  `plots/<model>/<file>/`. They are geometric: no efficiency, smearing,
  reconstruction or particle identification is applied.
- `events_pointing_analysis.py` applies the daughter-level pointing
  requirement — at least two of the same particles crossing the same plane —
  to a whole directory of event files and writes the per-file pointing
  efficiencies and corrected numbers of decays to
  `outputs/<model>/pointing_summary.csv`. It also reports the same efficiency
  with a 1 GeV momentum threshold per counted particle.
- `analyze_cascade_vs_primary.py` combines the totals files with that pointing
  summary and compares the dark-photon production sources: the ratio of the
  numbers of decays against mass and lifetime, and the corresponding figures.
- `plot_event_distributions.py` draws the polar-angle and energy distributions
  of the dark photons that decay inside the volume, for the primary and the
  cascade sources.
- `total-plots.py` draws the averaged quantities of a scan: the polar and
  total geometric acceptance, the mean decay probability, and the number of
  decays against coupling, mass and lifetime.
- `event_display.py` draws ten random events of a selected channel, as PDF and
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

- `simulate.py` — the launcher and the scan loop, interactive or from a card.
- `run_batch.py` — the same scan loop driven entirely from the command line.
- `runtime_generator.py` — the event-by-event and batched Python interface.
- `weighted_runtime.py` — the importance-weighted ALP sampler.
- `hepmc_export.py` — the streaming HepMC3 writer.
- `cards/` — launch cards and the experiment card `cards/ship.json`.
- `funcs/simulation_config.py` — validates cards and command-line arguments.
- `funcs/LLP_selection.py`, `funcs/channel_selection.py` — the interactive
  prompts and the one resolver of channel selections.
- `funcs/initLLP.py` — interpolates the production, lifetime and decay inputs.
- `funcs/ALPmerging.py`, `funcs/HNLmerging.py` — assemble the mixed photon–
  $SU(2)_L$ ALP and the arbitrary HNL mixing pattern from the installed pure
  tables.
- `funcs/kinematics.py` — samples parent momenta and decay positions.
- `funcs/TwoBodyDecay.py`, `ThreeBodyDecay.py`, `FourBodyDecay.py`,
  `NBodyDecay.py` — the rest-frame decay samplers.
- `funcs/thresholds.py` — the hadronic thresholds that close a decay row.
- `funcs/decayProducts.py` — decides what needs Pythia and routes the hadronic
  rows to Pythia or to exHad.
- `funcs/exhadDecays.py` — the exHad adapter.
- `funcs/boost.py` — the transformation to the laboratory frame.
- `funcs/mergeResults.py`, `funcs/output_provenance.py`,
  `funcs/fast_event_io.py` — write the event records, the totals and the
  sidecars.
- `funcs/ship_setup.py` — the SHiP geometry, and the one place to change it.
- `tests/` — the test modules; run them with
  `python3 -m pytest tests`.

Further details of the production and decay models are in
[`DETAILS.md`](DETAILS.md).

## Credits

This repository completes [Josue Jaramillo's CERN student
project](https://github.com/josuejaramillo/summer_school_2024_SHiP).
