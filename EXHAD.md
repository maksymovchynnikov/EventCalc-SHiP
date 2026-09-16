# Hadronic decays with exHad

exHad replaces the applicable complete hadronic decay pool with events
matched to exclusive calculations and electromagnetic data. EventCalc keeps
LLP production, lifetime, vertex sampling, boosts and yields. The returned
particles are not hadronized a second time.

## Setup

exHad rc10 optionally accelerates central-generator ALPs at 2.4 < m <=
3.5 GeV without importance weights. Build it with exHad's
`tools/build_conditional.py` after ordinary configuration; EventCalc then
uses it automatically. Other masses and portals retain their ordinary
samplers. See `EXHAD-PERFORMANCE.md` for the measured end-to-end timings.

Build and configure the exHad source release following its README. It uses
Pythia **8.317** and has its own compiled C++ worker. Then, in this repository:

```bash
export EXHAD_ROOT=/absolute/path/to/exhad-release
python runtime_generator.py --card cards/exhad_alp_fermion.json \
  --mass 2 --ctau 10 --events 10000 --batch-size 10000 \
  --seed 12345 --output-dir outputs/alp_exhad
```

For a dark photon use `cards/exhad_dark_photon.json`; for a Higgs-mixed
scalar use `cards/exhad_scalar.json`. The original launcher also accepts
these cards:

```bash
python simulate.py --card cards/exhad_alp_fermion.json
```

The card may contain `"exhad_root": "/absolute/path/to/exhad-release"`
instead of the environment variable. If exHad uses a separate virtual
environment, set `"exhad_python": "/absolute/path/to/exhad-release/.venv/bin/python"`.
This is a subprocess boundary: exHad's internal modules and Pythia library
do not replace EventCalc's imported modules.

**exHad is the default.** Add `--rawPythia` to either command to use
EventCalc's original decay path, without editing the card:

```bash
python simulate.py --card cards/exhad_alp_fermion.json --rawPythia
python runtime_generator.py --card cards/exhad_alp_fermion.json \
  --mass 2 --ctau 10 --events 10000 --seed 12345 \
  --output-dir outputs/alp_raw --rawPythia
```

`python simulate.py --rawPythia` also starts the interactive launcher in
raw mode. Explicit command-line flags override the card. Cards may still
set `"hadronization": "raw"`; `--exhad` overrides that setting. The Python
`generator_from_card` and `scan_from_card` helpers accept `hadronization="raw"`.
When neither the card nor the command selects a backend, exHad is used.
A failed exHad installation
or invalid input never silently falls back to raw Pythia. The raw path and
native low-mass exclusive decays can require Pythia's Python bindings;
both `pythia8` and the `pythia8mc` package name are supported. For matched
comparisons use the same 8.317 release, for example `pythia8mc==8.317.1`.

## Models and normalization

| EventCalc model | exHad choice | Matched mass range |
|---|---|---|
| `Dark-photons` | Data-constrained electromagnetic description | 1.70–5 GeV |
| `ALP-fermion` | Universal fermion couplings at Λ = 1 TeV | 1.911–3.5 GeV (rc5) |
| `Scalar-mixing`, `Scalar-quartic` | Blackstone-central scalar input | 2–63 GeV (showered above 5 GeV) |
| `HNL` | Signed current-specific exact-W replacement | Parent mass 0.02–5.27 GeV (rc6) |

Below these boundaries EventCalc retains the native exclusive decay path.
Above the selected deployment's upper limit a matched request is rejected;
EventCalc reads that limit from the release's `exhad.model_info` and records it in metadata.
The corrected rc5 ALP charge partition has validated support through
3.5 GeV; dark-photon support still ends at 5 GeV.
The photon/SU(2) ALP models are
different benchmarks and are not mapped to the fermion ALP. B−L light-current decays are available
in standalone exHad, but this repository has no B−L production model.

Use `"decay_channels": ["all"]`, or select the **complete** applicable
hadronic row pool. Partonic row names are bookkeeping inputs, not final
hadronic labels: their combined event count is handed to exHad once. Mixed
hadronic and leptonic selections retain the nonhadronic branching fractions.
Partial positive-weight hadronic selections are rejected.

HNL is the exception to pooling: each signed `Jets-*` row retains its own
event count and label. The original three-body matrix element generates the
lepton/neutrino and quark pair. exHad replaces only that pair at its exact
invariant mass W, preserving its four-momentum and the spectator particle.
CC ud/us/cd/cs and NC ud/s are supported; inactive heavier currents fail
explicitly if requested. Explicit meson and leptonic rows stay native.
The installed 88-channel decay table and its total-width table are validated
together and checked against the release. Both raw and matched HNL modes use
these updated inputs and the current-quark masses used by their matrix elements.
The fragmentation choice does not change HNL widths, lifetime or production.

The matched scalar and dark-photon modes read the decay/lifetime inputs that
the release's `exhad.model_info` lists. The four scalar input pairs and the
central raw-mode defaults were refreshed on 15 September 2026 and cover
0.01--63 GeV. Branching fractions and lifetime values are preserved, with
channel labels and PDG padding adapted to the existing interface.
Scalar decay generation covers 2--63 GeV (joined to showered Pythia over
4--5 GeV, except where the supplied b-bbar rate precedes the B+B- threshold,
10.542--10.5585 GeV). Scalar production distributions are unchanged and end
at 5.12 GeV, which bounds EventCalc scalar runs.
Consequently switching to exHad can also change a scalar lifetime input;
do not interpret that comparison as a fragmentation-only variation. For
that purpose hold the same decay/rate tables fixed in both calculations.
The new `ALP-fermion` directory includes its production, lifetime, branching
and coupling-convention metadata. It does not change `ALP-photon`,
`ALP-SU2L` or `ALP-mixed`.

The ALP hadronization model is a prototype supporting qualitative signature
conclusions. The dark-photon description is more directly constrained by
electromagnetic data. Scalar extrapolations and input alternatives are
documented in the exHad release.

## Output and testing

### Weighted ALP option (exHad rc8)

Ordinary `runtime_generator.py` and `simulate.py` remain unweighted. The
explicit weighted launcher writes bounded NumPy batches with **one global
normalization for the entire generated sample**:

```bash
python weighted_runtime.py --card cards/exhad_alp_fermion.json \
  --mass 3 --ctau 10 --events 100000 --batch-size 10000 --seed 12345 \
  --weight-floor-fraction 0.1 --output-dir outputs/alp_weighted
```

Use `analysis_weights` when filling weighted selections. Each NPZ also
retains the unchanged physical `decay_weights`, normalized
`hadronization_weights`, raw importance weights and normalization-group
labels. `metadata.json` records the full-sample normalization and effective
sample size. Do not renormalize weights per file or after selecting events.
Hadronization weights and physical decay probabilities are distinct factors.

The fraction 0 retains all positive-weight proposals; 0.1 additionally rejects
low-weight proposals with exact compensating weights; 1 recovers unweighted
selection. Channel probabilities, momentum models, total widths and parent
production/lifetime inputs are not changed by this choice. Statistical
precision depends on the observable and can be worse for rare eta/eta-prime
channels. Weighted mode supports the fermion-universal ALP only.

For Python, use `WeightedALPGenerator` from `weighted_runtime`. Its batches
carry explicit raw weights and a normalization key. Accumulate them with
`HadronizationNormalization` across the complete mass/configuration sample,
or use `export_weighted` to handle normalized NPZ output automatically. Mixing
masses/releases in one normalization is rejected. The old unweighted
`EventRecord` accessor deliberately rejects weighted batches. This launcher
does not export weighted HepMC3 or legacy text files; use its explicit NPZ
weights or the Python API.

Pythia's native conservation threshold is now set to the existing downstream
`2e-6` relative tolerance in both raw EventCalc and exHad. Failed numerical
events follow the existing retry path. Native EventCalc Pythia is explicitly
seeded from each decay block, avoiding a restart of its default random stream
in every batch. Seeds therefore need not reproduce pre-rc8 raw/native records.

### Ordinary unweighted output

The buffered runtime labels generated pooled decays `Hadronic-exHad` and
keeps native channel names for other decays. `metadata.json` records the
model, exHad release version and release directory. Momenta are
truth-level. Reconstruction efficiency and analysis selections are separate
from hadronization.

The older text-file launcher groups the same pooled events under
`Hadronic-exHad`, writes a `.hadronization.json` sidecar and places matched
files under `outputs/<model>/exhad/`, separate from raw outputs.

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python tests/smoke_exhad.py --exhad-root "$EXHAD_ROOT" --events 100
```

The smoke test generates all three matched benchmarks, checks laboratory
four-momentum conservation and event labels, and reports yield quantities.
Use large batches for throughput: importing and authenticating the
electromagnetic model has a noticeable one-time cost. Buffering does not
remove the cost of generating matched hadronic decays; consult measured
results rather than assuming the newer interface is faster everywhere.

exHad rc7 removes repeated random-state and isospin-coefficient computations
inside rejection. Its controlled 3-GeV ALP benchmark is 1.39–1.48 times faster
on one core, with exactly identical events at fixed sealed inputs and seeds.
The installed EventCalc adapter receives this improvement from exHad; it needs
no new flag or alternate decay prescription. This measured gain is specific
to that benchmark, not a promised factor for every model or HNL current.

## Parallel execution (exHad rc9 or later)

The adapter now uses independent, persistent event workers by default, up to
eight available CPUs. Set `EXHAD_WORKERS=1` for serial chunked execution, or
another positive count up to 64 to control CPU and memory use. Set
`EXHAD_CHUNK_SIZE` to change the default 512 events per chunk. Small requests
start only as many workers as they can use; keep the runtime alive between
batches to amortize initialization. Raw Pythia is unaffected.

At fixed card, seed, host batching and chunk size, worker count does not
change complete events or raw weights. Chunked execution uses a different
seed stream from rc8's single-request interface; changing chunk size changes
that stream. The pool never repeats a retained event to fill a sample. It
supports the boson and exact-W HNL adapters, including weighted ALP output;
weighted normalization still occurs once over the full sample, not per chunk.
Execution settings are recorded in metadata. The adapter requires an exHad
release that provides `exhad.model_info`.
See [timings and remaining costs](EXHAD-PERFORMANCE.md).

## Phenomenology sources

- [Blackstone et al., arXiv:2407.13587](https://arxiv.org/abs/2407.13587):
  Higgs-mixed scalar widths and uncertainties; [hipsofcobra](https://github.com/blackstonep/hipsofcobra).
- [Ovchynnikov and Zaporozhchenko, arXiv:2501.04525](https://arxiv.org/abs/2501.04525):
  GeV-scale ALP phenomenology (`Ovchynnikov:2025gpx`).
- [Foguel, Reimitz and Zukanovich Funchal, arXiv:2201.01788](https://arxiv.org/abs/2201.01788):
  [DeLiVeR](https://github.com/preimitz/DeLiVeR) vector hadronic rates.
- [Boiarska et al., arXiv:1904.10447](https://arxiv.org/abs/1904.10447):
  scalar-portal phenomenology (`Boiarska:2019jym`).

The exHad release supplies the precise model-to-input mapping, pinned code
revisions and source hashes. These links do not imply that every model uses
every cited input.
