# EventCalc-SHiP

EventCalc-SHiP generates decays of long-lived particles in the SHiP decay
volume and calculates their expected yields. The same calculation can be
adapted to another geometry through `funcs/ship_setup.py`.

## Physical calculation

For each selected mass and proper decay length, EventCalc

1. interpolates the production probability and the angular--energy density of
   the long-lived particle;
2. samples momenta directed toward the SHiP decay volume;
3. samples decay positions from the exponential decay law within the
   longitudinal extent of the volume;
4. applies its azimuthal geometry;
5. generates two-, three-, or four-body decays in the particle rest frame;
6. invokes Pythia 8 when partons or unstable decay products require
   showering, hadronization, or further decays;
7. boosts the final-state particles to the laboratory frame;
8. combines production, geometry, decay probability, and visible branching
   ratios to calculate the number of decays.

The generated event record contains truth-level momenta and decay positions.
[FairShip](https://github.com/ShipSoft/FairShip) can use this record for
particle transport and detector reconstruction.

The repository includes Higgs-like scalars, photophilic ALPs, dark photons,
heavy neutral leptons, the pure $SU(2)_L$ ALP, and a photon--$SU(2)_L$ ALP
mixture described below. The exact model
names accepted by the launcher are:

- `Scalar-mixing`;
- `Scalar-quartic`;
- `ALP-photon`;
- `Dark-photons`;
- `HNL`;
- `ALP-SU2L`;
- `ALP-mixed`.

## $SU(2)_L$ ALP

The `ALP-SU2L` model uses

$$
\mathcal L\supset C_WaW^I_{\mu\nu}\widetilde W^{I\mu\nu},
\qquad
C_W=\frac{\alpha_2}{4\pi}g_W,
\qquad
\frac{c_W}{f_a}=-C_W,
$$

with $c_B=c_{a\Phi}=0$. The last relation converts to the convention of
arXiv:1901.02031, where the operator is
$-c_WaW^I_{\mu\nu}\widetilde W^{I\mu\nu}/f_a$. Electroweak symmetry breaking
induces

$$
C_\gamma^{\rm ind}=\sin^2\theta_W C_W,
\qquad
g_{a\gamma\gamma}=\frac{\alpha_{\rm em}}{\pi}g_W.
$$

Four processes contribute to the ALP flux: rare $B\to X_{s,d}a$ decays,
$K^\pm\to\pi^\pm a$ below the charged-kaon threshold, Primakoff production
by photons from the proton target interaction, and Primakoff production by
photons in the electromagnetic cascade. Let $\widehat Y_B$ and
$\widehat Y_K$ denote the production coefficients per $g_W^2$. Let
$\widehat Y_{\rm primary}$ and $\widehat Y_{\rm cascade}$ denote the Primakoff
coefficients per $g_{a\gamma\gamma}^2$. Since
$g_{a\gamma\gamma}=kg_W$, the combined coefficient per $g_W^2$ is

$$
\widehat Y_{\rm tot}=\widehat Y_B+\widehat Y_K
+k^2(\widehat Y_{\rm primary}+\widehat Y_{\rm cascade}),
\qquad k=\frac{\alpha_{\rm em}}{\pi},
$$

and the production probability is
$P_{\rm prod}=g_W^2\widehat Y_{\rm tot}$. The normalized angular--energy
density is

$$
f_{\rm tot}=
\frac{\widehat Y_Bf_B+\widehat Y_Kf_K
+k^2\widehat Y_{\rm primary}f_{\rm primary}
+k^2\widehat Y_{\rm cascade}f_{\rm cascade}}
{\widehat Y_{\rm tot}}.
$$

EventCalc reads this combined density as one production model. Its forward
integral equals the polar acceptance; the angular restriction preserves the
normalization over the complete angular range. EventCalc multiplies
$\widehat Y_{\rm tot}$ by $g_W^2$ once when it calculates the number produced.

The charged-kaon flux is $0.36=0.29+0.07$ kaons per proton on target, summed
over the two charges. The branching coefficient is evaluated separately as a
function of $m_a$, and the contribution vanishes exactly at
$m_a=m_{K^\pm}-m_{\pi^\pm}$.

The present decay model assumes
$\operatorname{Br}(a\to\gamma\gamma)=1$. Its scope is the diphoton benchmark.
Calculations involving hadronic ALP widths or other decay modes require an
extended decay description. All numerical production, lifetime, and decay
tables needed to run this benchmark are supplied under
`Distributions/ALP-SU2L/`; users do not need table-generation code.

For a requested proper decay length $c\tau$, EventCalc obtains the coupling
from

$$
g_W^2=(1\,\mathrm{GeV}^{-1})^2
\frac{(c\tau)_{g_W=1\,\mathrm{GeV}^{-1}}}{c\tau}.
$$

It then multiplies $\widehat Y_{\rm tot}$ by the numerical value of $g_W^2$
expressed in $\mathrm{GeV}^{-2}$.

## Mixed photon--$SU(2)_L$ ALP

The `ALP-mixed` model implements both operators coherently. Its two additional
inputs are $0\leq\xi\leq1$ and `constructive` or `destructive` interference.
The convention is

$$
C_W=\Lambda\xi,
\qquad
C_\gamma^{\rm dir}=s\Lambda(1-\xi),
\qquad s=+1\ \hbox{or}\ -1,
$$

and therefore

$$
g_{a\gamma\gamma}^{\rm tot}
=4\Lambda\left[s(1-\xi)+\sin^2\!\theta_W\,\xi\right]
=g_{a\gamma\gamma}^{\rm dir}+\frac{\alpha_{\rm em}}{\pi}g_W.
$$

The amplitudes interfere before the decay width is calculated; the lifetime
is not obtained from a sum of the two coupling squares. For the diphoton-only
benchmark,

$$
c\tau=\frac{64\pi\hbar c}
{m_a^3\left|g_{a\gamma\gamma}^{\rm tot}\right|^2}.
$$

The `c_taus` values remain physical proper decay lengths. EventCalc infers
$|g_{a\gamma\gamma}^{\rm tot}|^2$ from this equation. At fixed $\xi$ and sign,
the relative coefficient of flavor-changing $B$- and charged-kaon production
to primary- and cascade-photon production is

$$
\frac{g_W^2}{|g_{a\gamma\gamma}^{\rm tot}|^2}
=\frac{(4\pi/\alpha_2)^2\xi^2}
{16\left[s(1-\xi)+\sin^2\!\theta_W\,\xi\right]^2}.
$$

No spectrum generation is needed at launch. The code reconstructs the $B+K$
component from the installed inclusive `ALP-SU2L` table and the installed
primary/cascade photon tables. Thus $\xi=0$ reproduces their combined
photophilic production, while $\xi=1$ reproduces the pure `ALP-SU2L` physics.
The common tabulated mass range is $0.02$--$4$ GeV. The exact destructive
diphoton-cancellation point is rejected because this diphoton-only model has no
finite signal there.

## Installation

The code has been tested on Linux. Install the Python dependencies with

```bash
pip3 install numpy sympy numba scipy pandas matplotlib
```

Plotly is optional and is used only by the interactive Plotly event display:

```bash
pip3 install plotly
```

[Pythia 8](https://pythia.org/) is needed for selected decay channels that
contain partons or unstable particles. Diphoton decays of `ALP-photon`,
`ALP-SU2L`, and `ALP-mixed` use the internal two-body generator and run
without Pythia 8.

To install the Python bindings for Pythia 8, configure and build it with

```bash
./configure --with-python-config=python3-config
make
```

Set `PYTHIA8_LIB` to the directory containing `pythia8.so`. The fallback path
is `/home/name/Downloads/pythia8312/lib`.

## Simulation interface

EventCalc supports a JSON launch card, explicit command-line arguments, and
the original interactive prompts. The first two interfaces are silent: every
physical and numerical choice is supplied before the calculation starts.
Card and command launches call no input prompts and use Matplotlib's
non-interactive `Agg` backend. They create phenomenology figures only when
`plots` or `--plots` is enabled, and write those figures to disk.

### JSON launch card

The example below launches the $SU(2)_L$ ALP for two masses and two lifetimes.
Each lifetime is evaluated at each mass.

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

Run it with

```bash
python3 simulate.py --card cards/alp_su2l.json
```

Check the card, resolve its decay channels, and print the normalized
configuration with

```bash
python3 simulate.py --card cards/alp_su2l.json --validate-only
```

This command exits before the numerical simulation is imported.

The required fields are `model`, `events`, `masses`, and `c_taus`.
`decay_channels` accepts names, one-based indices, or `"all"`. A flat
`c_taus` list is used for every mass. A nested list gives a separate lifetime
list for each mass.

The optional common fields are:

- `seed`, an integer between 0 and $2^{32}-1$;
- `plots`, with default `false` in silent mode;
- `export_events`, with default `true`;
- `n_pot`, with default $6\times10^{20}$ protons on target;
- `min_events_threshold`, with default $0.1$ expected decays.

Three models require an additional field:

- `HNL`: `mixing_pattern`, containing $\xi_e$, $\xi_\mu$, and $\xi_\tau$;
- `Dark-photons`: `uncertainty`, chosen from `lower`, `central`, and `upper`;
- `ALP-photon`: `alp_production_mode`, chosen from `primary` and `cascades`.

`ALP-mixed` requires `xi`, between 0 and 1, and `interference`, chosen from
`constructive` and `destructive`. For example:

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

Run the supplied example with

```bash
python3 simulate.py --card cards/alp_mixed.json
```

`ALP-SU2L` already contains its $B$, charged-kaon, primary-photon, and
cascade-photon contributions. Its card therefore has no production-source,
uncertainty, or mixing field.

### Explicit command

The same $SU(2)_L$ ALP calculation can be launched with

```bash
python3 simulate.py \
  --model ALP-SU2L \
  --events 200000 \
  --masses 0.3 1.0 \
  --c-taus 0.01 10000 \
  --decay-channels 2gamma \
  --seed 12345
```

Command-line arguments override fields from a card. `python3 simulate.py
--help` lists all options.

The mixed example can equivalently be launched with `--model ALP-mixed`,
`--xi 0.35`, and `--interference constructive`, together with the common
arguments above.

### Interactive prompts

Running

```bash
python3 simulate.py
```

starts the interactive interface. The model menu has a fixed order and uses
the names listed above. It asks for the event sample size, model-specific
parameters, decay channels, masses, and proper decay lengths. It generates the
production-probability, lifetime, and branching-fraction figures.

## Event yields and output

For each mass and lifetime, EventCalc samples `events * 10` interpolation
candidates and resamples `events` momenta within the polar range. The
azimuthal selection reduces the number of stored decays, typically by a factor
of order $0.6$--$1$ for the current geometry.

If the production probability per proton on target is below $10^{-21}$, the
mass--lifetime point is skipped. If the expected number of decays is below
`min_events_threshold`, EventCalc stores its total quantities and skips the
generation of decay products.

Results are written under `outputs/<model>/`:

- `eventData/` contains the generated event records when event export is
  enabled;
- `total/` contains mass, coupling, lifetime, production yield, polar and
  azimuthal acceptance, mean decay probability, visible branching ratio, and
  expected number of decays.

For `ALP-mixed`, the `coupling_squared` column is specifically
$|g_{a\gamma\gamma}^{\rm tot}|^2$ in $\mathrm{GeV}^{-2}$. Mixed output names
also contain $\xi$ and the interference sign, so different operator choices
cannot overwrite one another.

Each event record starts with the long-lived particle four-momentum, mass,
identifier, decay probability, and decay coordinates. Every final-state
particle contributes its four-momentum, mass, and identifier. Momenta,
energies, and masses are in GeV; positions are in metres. The event weight is
the decay probability stored with the parent particle.

Enable `plots` in a card or pass `--plots` to write the mass dependence of the
production probability, proper decay length, and selected branching ratios
under `plots/<model>/phenomenology`.

### Optional truth-level detector acceptance

The main simulation reports decays whose parent LLP trajectory lies inside the
fiducial decay volume. With event export enabled, `events_analysis.py` can
add a model-independent, truth-level requirement on the visible decay
products. By default it projects straight downstream trajectories onto a
$4\,\mathrm{m}\times6\,\mathrm{m}$ plane at $z=95\,\mathrm{m}$.

Run the analysis and select an existing event file interactively:

```bash
python3 events_analysis.py --signature two-photon
```

The available signatures are:

- `two-photon`: exactly two visible particles, both photons, must intersect the
  plane;
- `neutral-pair`: at least one subset of two or more particles that intersects
  the plane must have zero total electric charge;
- `all-visible`: every visible particle must intersect the plane. Neutrinos are
  ignored.

The default is `all-visible`. Change the plane with `--detector-z`,
`--detector-width`, and `--detector-height`. An optional common energy threshold
is set with `--min-energy E_MIN`, in GeV. Every counted particle must satisfy
$E\geq\max(m,E_{\min})$; without this option the default is the kinematic
condition $E\geq m$.

The script writes `detector_acceptance.txt` and
`channels_detector_acceptance.pdf` in the selected analysis directory. The
reported fractions are weighted by the parent decay probability. Multiplying
the generator's `Total number of events` by the overall fraction gives the
truth-level accepted yield. No detector efficiency, energy or position
smearing, reconstruction, or particle-identification inefficiency is applied.

## Checks against other calculations

EventCalc-SHiP has been compared with
[`SensCalc`](https://github.com/maksymovchynnikov/SensCalc), which was also
tested against earlier FairShip calculations and other tools. The event yield,
geometric acceptance, mean decay probability, and kinematic distributions
agree at the 10% level or better; see the
[comparison slides](https://indico.cern.ch/event/1481729/contributions/6256116/).

## Source layout

- `simulate.py` contains the launcher and simulation loop.
- `funcs/simulation_config.py` validates cards and command-line arguments.
- `funcs/initLLP.py` interpolates production, lifetime, and decay inputs.
- `funcs/ALPmerging.py` assembles a requested photon--$SU(2)_L$ mixture from
  the installed pure-source tables.
- `funcs/kinematics.py` samples parent momenta and decay positions.
- `funcs/TwoBodyDecay.py`, `ThreeBodyDecay.py`, and `FourBodyDecay.py`
  generate rest-frame decays.
- `funcs/decayProducts.py` decides whether Pythia 8 is required.
- `funcs/boost.py` transforms decay products to the laboratory frame.
- `funcs/mergeResults.py` writes event records and total yields.
- `funcs/ship_setup.py` defines the SHiP geometry.
- `events_analysis.py`, `total-plots.py`, and `event_display.py` provide
  additional analysis and visualization of completed simulations.

Further details of the implemented production and decay models are in
[`DETAILS.md`](DETAILS.md).

## Credits

This repository completes Josue Jaramillo's CERN student project.
