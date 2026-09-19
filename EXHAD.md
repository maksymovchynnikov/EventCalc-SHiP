# Hadronic decays with exHad

[exHad](https://arxiv.org/abs/2609.16104) generates the complete hadronic
final states of a GeV-scale long-lived particle: the composition of the
multihadron states follows exclusive calculations and measured $e^+e^-$
annihilation cross sections instead of unconstrained string fragmentation.
EventCalc can hand the hadronic decays of a scan to it. Production, lifetimes,
branching ratios, vertex sampling, boosts and yields stay with EventCalc, and
the particles exHad returns are not hadronized a second time.

This file covers the installation, how a run selects exHad, what each model
covers, and what the output records. The physical constructions behind the
models are summarized in [EXHAD-MODELS.md](EXHAD-MODELS.md).

## Installing exHad

exHad is a separate package with its own compiled worker, built against
Pythia **8.317**. EventCalc accepts other Pythia versions with matching XML
data, but use 8.317 when comparing the two hadronization models. Follow the
[exHad installation instructions](https://github.com/maksymovchynnikov/exHad):

```bash
cd /path/to/exhad
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python tools/setup.py --pythia8-dir /absolute/path/to/pythia8317
```

`tools/setup.py` downloads the external form-factor code, builds the C++
worker, checks that the worker and the Pythia XML payload report the same
version, and builds the runtime helpers exHad generates events with. Nothing
is generated if one of those steps fails. A quick standalone check that the
installation works:

```bash
python -m exhad --model alp-fermion --mass 2.0 --events 100 --seed 1 \
  --output /tmp/events.json
```

EventCalc and exHad link their own Pythia installations and never share a
library search path: the EventCalc process uses the XML payload installed with
its Python binding, and the exHad worker uses the one recorded when it was
built. Set `EXHAD_PYTHIA8DATA` only to relocate the latter.

## Pointing EventCalc at it

```bash
export EXHAD_ROOT=/absolute/path/to/exhad
python3 simulate.py --card cards/exhad_alp_fermion.json
```

`cards/exhad_dark_photon.json`, `cards/exhad_scalar.json` and
`cards/exhad_hnl.json` are the other prepared cards. A card may name the
installation itself with `"exhad_root"`, and `"exhad_python"` selects the
interpreter that runs the exHad workers when it lives in its own environment
(the environment variables `EXHAD_ROOT` and `EXHAD_PYTHON` do the same). The
workers are separate processes, so exHad's modules and its Pythia library
never replace EventCalc's.

`EXHAD_WORKERS` sets the number of worker processes, by default one per CPU up
to eight, and `EXHAD_CHUNK_SIZE` the number of events per chunk, by default
512. At a fixed card, seed and chunk size the worker count does not change the
events.

## Selecting exHad or the Pythia baseline

Every entry point takes `--exhad {auto,on,off}`:

- `auto`, the default, uses exHad where the selected model has a card and a
  usable installation, and runs on the unmodified-Pythia baseline otherwise;
- `on` requires a usable installation: a run whose selection has a matched
  route stops with an error instead of quietly changing the physics. For a
  selection with no hadronic route at all, such as the diphoton and leptonic
  final states of `ALP-photon`, `on` is a reported no-op;
- `off` (aliases `--rawPythia` and `--raw-pythia`) forces the baseline.

A bare `--exhad` means `on`. The card key `hadronization` takes `"exhad"` and
`"rawPythia"` and sets the same switch, and an explicit flag overrides the
card. An interactive run asks the same question after the channel menu, and
offers it when the selected particle has an available exHad model.

The selected particle and scalar prescription determine the exHad model in
the table below. Normally no internal card needs editing. Missing or
malformed model cards and errors loading a configured release stop the run;
the absence of an optional installation in `auto` mode uses raw Pythia.

The route is recorded in the name of every output file —
`eventcalc-direct-seed<N>` when every selected row with positive rate is
already stable, `raw-pythia-seed<N>` when at least one needs Pythia,
`exhad-<benchmark>-seed<N>` when a matched row carries positive rate — and in
the `<file>.generator.json` sidecar beside it. Matched and baseline runs land
in the same directory and are told apart by those names.

## What the installed cards cover

| EventCalc model | exHad model | Mass support | What exHad generates |
|---|---|---:|---|
| `Dark-photons` | `dark-photon` | 1.70–5.00 GeV | The selected $u\bar u$, $d\bar d$, $s\bar s$ and $c\bar c$ rows are one pool. Its light component follows the DeLiVeR exclusive decomposition from 1.70 to 2.00 GeV and the coherent fit to measured cross sections above 2.00 GeV; the measured open-charm channels are generated above the $D\bar D$ threshold; the unresolved light component changes continuously to Pythia fragmentation between 4 and 5 GeV |
| `ALP-fermion` | `alp-fermion` | 1.911–5.00 GeV | One pool of all selected hadronic rows, including the nucleon-pair rows |
| `Scalar-mixing`, `Scalar-quartic` | `scalar-central`, `scalar-lower`, `scalar-upper`, `scalar-1809` | 2.00–63.0 GeV, of which the SHiP production tables cover masses up to 5.12 GeV | One pool of all selected hadronic rows; the exHad model follows the prescription selected for the rates |
| `HNL` | `hnl` | 0.02–40 GeV, of which the SHiP production tables cover masses up to 5.27 GeV | The hadrons of a selected current at the exact invariant mass $W$ of the quark–antiquark pair of each event |

Below the support of a boson portal the hadronic rows are generated as exact
exclusive final states, row by row: EventCalc decides how many events each row
gets and exHad realizes them. This is what closes the gap just below the
matched support, where events used to fall through to unfiltered Pythia, and
it removes the sub-threshold neutral-kaon failures with it, because a row
exHad cannot open is never requested.

`ALP-photon`, `ALP-SU2L` and `ALP-mixed` are different benchmarks from the
fermion-coupled ALP and are not mapped to it; they keep their native decays.
exHad also carries a $B-L$ boson, for which this repository has no production
tables.

### Selecting channels for a matched run

Use `"decay_channels": ["all"]`, or `jets`, or a complete set of channel
names. The partonic row names are bookkeeping inputs, not the labels of
hadronic final states: their summed event count is handed to exHad once, and a
selection that takes some of the rows of one pool and leaves others is
refused. A mixed hadronic and leptonic selection keeps the nonhadronic
branching fractions untouched.

### Heavy neutral leptons

The HNL is the exception to pooling. Each signed `Jets-*` row keeps its own
event count and its own label. EventCalc's three-body matrix element generates
the charged lepton or neutrino together with the quark–antiquark pair, and
exHad replaces only that pair at its exact invariant mass $W$, preserving its
four-momentum and the spectator particle. The six currents with positive
installed production weight are:

- `CC_ud`, built from the ALEPH $\tau$ spectral functions up to $W=1.65$ GeV
  and continued above it, and `CC_us`, a data-tied strange-resonance
  composition with the explicit $K$ pole removed. The charged lepton and the
  sampled $W$ are kept, and the charge conjugation follows the electric
  charge.
- `CC_cd` and `CC_cs`, the open-charm continua at the same exact $W$, with the
  explicit single-$D$ and single-$D_s$ rows kept as EventCalc decays.
- `NC_ud`, the light $\rho/\omega$ and axial construction normalized to the
  OPAL and $\tau$ data up to 1.65 GeV, and `NC_s`, the strange $\phi$ and
  axial construction. The neutrino is kept.

`NC_c`, `NC_b`, `CC_ub` and `CC_cb` have no positive installed production
weight in this domain and a request for one fails explicitly. Every explicit
$N\to\ell P$ and $N\to\nu P$ row stays an EventCalc decay and is never sent to
exHad.

The hook is a decay-stage replacement of the final state: the tabulated
per-channel rates, the HNL lifetime, the channel counts and the lepton
kinematics are exactly what EventCalc computes, and only the hadrons of the
quark–antiquark system change. It does not recompute HNL widths and does not
patch the decay tables. The 88-channel HNL decay table and its total-width
table are checked against the installed exHad release before a matched run
starts, and a disagreement stops the run: a calculation whose rates come from
one table set and whose final states come from another is neither of the two.

The partonic phase space of an HNL decay uses the current-quark masses the
exported matrix elements are written with,
$\{d:0.0047,\ u:0.0022,\ s:0.104,\ c:1.27,\ b:4.18\}$ GeV, stated once in
`funcs/thresholds.py`. Using constituent masses instead would remove the
physical low-$W$ continuum. The hadronic threshold itself is enforced
separately, by the lightest-hadron floor in the three-body sampler and by the
charge-aware two-hadron rule in `funcs/thresholds.py`.

### Internal card names (for maintainers)

The installed `Distributions/<LLP>/exhad.json` files retain older internal
keys: `dv` maps to `dark-photon`, `alp` to `alp-fermion`, and
`hls`, `hls-lower`, `hls-upper`, `hls-1809` to the four `scalar-*` models.
These are compatibility keys, not additional physics models. They also
appear in existing output filenames. Use the public exHad names for its API
and CLI; use EventCalc's particle names and `--scalar-prescription` here.

For compatibility, `EXHAD_BENCH` overrides the internal card selection
(`raw` forces raw Pythia), while `EXHAD_LLP` is a fallback for callers that
provide no particle selection. An explicit raw-Pythia run takes precedence.
Leave these overrides unset in ordinary runs.

The cards also record matching and charm thresholds. These are model inputs,
not user-selected smoothing intervals; declared matching windows are checked
against the installed exHad model before generation.

## Output of a matched run

The events of one pool are written as one block per table row, each labelled
`Jets-matched:<row>`, in the order they were sampled. The counts per row and
the order of the events are exactly as drawn, so an event keeps its position
in the file and its correspondence with the per-row counts.

Beside each output file EventCalc writes `<file>.generator.json` with the
route and its inputs, `<file>.coupling.json` with the coupling convention,
and, for a matched run, `<file>.hadronization.json`, which records the exHad
release version and the mother PDG code. The default mother PDG code comes
from the release, `model_info()['mother_pdg']`, and `--llp-pdg CODE` overrides
it for one run. The code 25 that EventCalc uses internally when it hands a
partonic state to Pythia is an unrelated bookkeeping convention.

### Weighted ALP samples

Ordinary runs are unweighted. `weighted_runtime.py` is a separate launcher
that writes importance-weighted samples of the fermion-coupled ALP as bounded
NumPy archives, with one normalization for the whole generated sample:

```bash
python3 weighted_runtime.py --card cards/exhad_alp_fermion.json \
  --mass 3 --ctau 10 --events 100000 --batch-size 10000 --seed 12345 \
  --weight-floor-fraction 0.1 --output-dir outputs/alp_weighted
```

Fill weighted selections with `analysis_weights`. Each archive also keeps the
physical `decay_weights`, the normalized `hadronization_weights`, the raw
importance weights and the normalization-group labels, and `metadata.json`
records the full-sample normalization and the effective sample size. Do not
renormalize per file or after selecting events: hadronization weights and
decay probabilities are separate factors. A floor fraction of 0 keeps every
positive-weight proposal, 0.1 rejects low-weight proposals with exactly
compensating weights, and 1 recovers unweighted selection. Channel
probabilities, total widths, production and lifetime inputs do not depend on
this choice. Statistical precision depends on the observable and is worse for
the rare $\eta$ and $\eta'$ channels.

In Python, `WeightedALPGenerator` from `weighted_runtime` produces batches
carrying their raw weights and a normalization key; accumulate them with
`HadronizationNormalization` over the complete sample, or let
`export_weighted` write the normalized archives. Mixing masses or
installations in one normalization is refused. This launcher writes neither
HepMC3 nor the text event record.

## Checking an installation

```bash
python3 -m pytest tests

# Also exercise a separately installed, configured exHad release:
export EXHAD_ROOT=/absolute/path/to/exhad
python3 -m pytest tests
python3 tests/smoke_exhad.py --exhad-root "$EXHAD_ROOT" --events 100 \
  --models ALP-fermion Scalar-mixing Dark-photons HNL --mass 3
```

Without `EXHAD_ROOT`, release-dependent tests are skipped and the HNL sampler
is checked against EventCalc's own hadronic thresholds. With it, the tests
also check the release thresholds. The explicit smoke command above generates
four model samples, checks laboratory four-momentum conservation and event
labels, and prints yields. It is a correctness check, not a speed benchmark.

## The interface EventCalc uses

exHad has one `Generator`, which generates in parallel by default, and its
methods are `generate`, `generate_all`, `generate_rows`, `generate_weighted`
(fermion-coupled ALP only) and `hadronize_hnl`. EventCalc reaches all of it
through `funcs/exhad_release.py` and `funcs/exhadDecays.py`. exHad chooses its
own sampler: it generates the unweighted samples of the ALP above 2.4 GeV, of
the scalars from 3 to 5 GeV and of the $B-L$ boson from 2 to 5 GeV with an
accelerated sampler that decides acceptance while a string fragments, and uses
its rejection sampler elsewhere. Both give the same event distribution; no
EventCalc setting selects between them.

### Model names

`dark-photon`, `alp-fermion`, `scalar-central`, `scalar-lower`,
`scalar-upper`, `scalar-1809`, `b-l` and `hnl`. These are exHad's public names;
the internal EventCalc card keys are explained above. The scalar prescription
is chosen in EventCalc with `--scalar-prescription
central|lower|upper|1809`, default `central`.

### Mass windows and mother PDG codes

| Model | Mother PDG | Matched support (GeV) | Generation (GeV) | Exclusive rows below the support |
|---|---|---|---|---|
| `dark-photon` | 4900022 | 1.70–5.00 | 0.003–5.00 | 41 rows, 0.003–1.70 |
| `alp-fermion` | 36 | 1.911–5.00 | 0.01–5.00 | 25 rows, 0.01–1.911 |
| `scalar-central`, `-lower`, `-upper`, `-1809` | 35 | 2.00–63.0 | 0.01–63.0 | 9 rows, 0.01–2.00 |
| `b-l` | 32 | 2.00–5.00 | 0.001–5.00 | 32 rows, 0.001–2.00 |
| `hnl` | 9900012 | 0.02–5.27, a bound on $W$ | 0.02–40.0 | — |

A requested mass is checked against the generation interval. A boson portal
decays to hadrons alone, so its $W$ is its own mass and the two intervals end
together. An HNL shares its mass with a charged lepton or a neutrino, so its
$W$ stays below its own mass and the two intervals end apart.

### Matrix elements

The HNL tables take the lepton masses from the generator instead of freezing
them as decimals, so nine rows (`2ev`, `2muv`, `2tauv`, `emuv(bar)`,
`etauv(bar)`, `mutauv(bar)`) carry the symbols `m1` and `m3`. Every compiled
matrix element therefore takes six arguments, $(m_{\rm LLP}, E_1, E_3, m_1,
m_2, m_3)$, and the three-body sampler supplies the row's daughter masses. The
parser rejects any other free symbol. The dark-photon `Pip_Pim_Pi0` row is the
symmetric $\rho$-exchange form with the two physical pion masses. The HNL
tables carry no radiative $N\to\nu\gamma$ row, and none may be added: exHad
does not carry one, and the check of the EventCalc tables against the release
fails if the two sides differ.

### Sampling and seeding

Three-body and four-body decays use exact accept-reject sampling, so every
returned event is an independent draw and a channel can hold any number of
events. One seed reaches every stream: `funcs/seeding.py` seeds NumPy's host
generator and the generator Numba compiles into the `@njit` kernels, which
`np.random.seed` in the host does not reach. Pythia is seeded separately and
explicitly: each block of events derives a seed from the block's own seed and
its chunk index and writes it into `Random:seed`, so a block runs the stream
its index names. exHad's generators take their seed explicitly per call. Two
runs of the same card with the same seed produce the same decay products.

The base seed is `seed` in a card, `--seed` on the command line,
`--exhad-seed` in `run_batch.py`, or `EXHAD_SEED` in the environment for an
interactive session, which has no card and no command line to carry one; its
default is 1. Each scan point derives its own seed from the base seed and the
position of the point in the grid, so a scan is reproducible and
`--batch-index-offset` shifts a job to its own streams.

## Sources of the physics

- [Rethinking search signatures: Hadronic decays of GeV-scale feebly coupled
  particles, arXiv:2609.16104](https://arxiv.org/abs/2609.16104): the decay
  model exHad implements.
- [Blackstone et al., arXiv:2407.13587](https://arxiv.org/abs/2407.13587):
  Higgs-mixed scalar widths and their uncertainty;
  [hipsofcobra](https://github.com/blackstonep/hipsofcobra).
- [Boiarska et al., arXiv:1904.10447](https://arxiv.org/abs/1904.10447) and
  [arXiv:1809.01876](https://arxiv.org/abs/1809.01876): the other scalar
  prescription.
- [Ovchynnikov and Zaporozhchenko,
  arXiv:2501.04525](https://arxiv.org/abs/2501.04525): GeV-scale ALP
  phenomenology.
- [Foguel, Reimitz and Zukanovich Funchal,
  arXiv:2201.01788](https://arxiv.org/abs/2201.01788):
  [DeLiVeR](https://github.com/preimitz/DeLiVeR), the hadronic rates of vector
  currents.
- [Bondarenko et al., arXiv:1805.08567](https://arxiv.org/abs/1805.08567): HNL
  widths and matrix elements.

The exHad release documents which input each of its models uses, with pinned
revisions and source hashes; the list above does not mean that every model
uses every entry.
