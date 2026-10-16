# Hadronic event generation

[exHad](https://github.com/maksymovchynnikov/exHad) generates complete hadronic
final states using the model of
[arXiv:2609.16104](https://arxiv.org/abs/2609.16104). EventCalc fixes the
production rates, decay-channel branching fractions and lifetimes of its
standard tabulated models, then obtains hadronic momenta from exHad. For
`Inelastic-DM`, exHad also computes the excited-state widths and complete decay
chains. The [decay-model summary](EXHAD-MODELS.md) explains the physical inputs.

## Installation

Follow the current [exHad installation instructions](https://github.com/maksymovchynnikov/exHad).
A local source installation uses:

```sh
cd /path/to/exHad
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python tools/setup.py --pythia8-dir /absolute/path/to/pythia8317
```

The setup builds exHad's worker against Pythia8 8.317. The Python binding used
by EventCalc and the worker used by exHad each require matching Pythia8 XML
data. For relocated worker data, set `EXHAD_PYTHIA8DATA`.

Return to the EventCalc repository and select the installation:

```sh
export EXHAD_ROOT=/absolute/path/to/exHad
python simulate.py --card cards/exhad_alp_fermion.json
```

A card can set `exhad_root` and `exhad_python`; the latter selects exHad's
Python interpreter when it has a separate environment. `EXHAD_PYTHON` provides
the same interpreter setting. Prepared cards also exist for `Dark-photons`,
`Scalar-mixing` and `HNL` under `cards/`.

## Selecting hadronic generation

`--exhad auto` follows the selected model's configuration and uses exHad where
it is installed; otherwise it uses unmodified [Pythia8](https://pythia.org/).
`--exhad on` requires exHad for a selected hadronic channel.
`--exhad off` selects unmodified Pythia8. Channels containing only stable
particles are generated directly. `Inelastic-DM` always requires exHad.
The card field `hadronization` accepts `exhad` or `rawPythia`; explicit command
options take precedence.

Use `decay_channels: ["all"]` for inclusive decays or `jets` for the complete
partonic contribution. Exclusive channel names can be selected from the printed
menu. For a boson, partonic channels contributing to the same inclusive
hadronic rate must be selected together. EventCalc sums their branching
fractions and asks exHad for that number of complete hadronic decays.
Nonhadronic channels keep their own rates and kinematics.

## Model coverage

The ranges below describe hadronic generation through this integration.
Production tables can impose a smaller mass range for a simulation.

| EventCalc model | exHad model | Hadronic generation | SHiP production limit |
| --- | --- | --- | --- |
| `Dark-photons` | `dark-photon` | Exclusive channels below 1.70 GeV; fitted multihadron composition from 1.70 to 5 GeV | See [production coverage](PRODUCTION_DATA.md#compact-distributions) |
| `ALP-fermion` | `alp-fermion` | Exclusive channels below 1.910 GeV; combined hadronic generation from 1.910 to 5 GeV | Set by the common production and lifetime tables |
| `Scalar-mixing`, `Scalar-quartic` | `scalar-central`, `scalar-lower`, `scalar-upper`, `scalar-1809` | Exclusive channels below 2 GeV; inclusive generation from 2 to 63 GeV | 5.12 GeV |
| `HNL` | `hnl` | Hadronic weak currents for parent masses 0.02 to 40 GeV, evaluated at the quark-pair invariant mass | 5.27 GeV |
| `Inelastic-DM` | `exhad.inelastic_dark_matter` | Inclusive excited-state decays in the [stated energy-release range](INELASTIC_DM.md#inputs-and-generation) | See [production coverage](PRODUCTION_DATA.md#compact-distributions) |

`--scalar-prescription` selects the scalar width calculation and the matching
hadronic model together; the allowed values are defined in the
[model guide](models.md#higgs-like-scalar). Above the supported exHad mass range, select `--exhad off` explicitly
to use Pythia8 for open channels within the production tables. A configured
exHad run stops if its decay model cannot generate the requested mass. `ALP-photon`, `ALP-SU2L` and `ALP-mixed` use their own decay models.

In `HNL` decays such as $N\to\ell q\bar q$ and $N\to\nu q\bar q$, exHad
replaces the quark pair by hadrons at the pair invariant mass $W$. Its total
four-momentum and the accompanying lepton remain fixed. Exclusive one-meson
channels are generated separately. Hadronic thresholds constrain $W$ before
sampling, while the matrix elements use current-quark masses.

## Outputs and reproducibility

The [output guide](user-guide.md#event-yields-and-output) defines text events,
HepMC3 and metadata. Hadronic runs record the selected generator and its
configuration, and their filenames distinguish exHad from Pythia8.

`EXHAD_WORKERS` controls worker processes, with a default of one per CPU up to
eight. `EXHAD_CHUNK_SIZE` sets events per chunk, default 512. At fixed model,
seed, installation and chunk size, changing the worker count preserves the
events. A scan reuses rest-frame decays across lifetimes at a fixed mass.
See [measuring performance](EXHAD-PERFORMANCE.md) for timing conventions.

## Weighted ALP decays

The separate weighted launcher generates `ALP-fermion` samples with importance
weights:

```sh
python -m eventcalc.weighted_runtime --card cards/exhad_alp_fermion.json \
  --mass 3 --ctau 10 --events 100000 --batch-size 10000 --seed 12345 \
  --weight-floor-fraction 0.1 --output-dir outputs/alp_weighted
```

Use `analysis_weights` for histograms and selections. Output also includes the
physical decay probability and hadronization weights. Their normalization uses
the complete generated sample and must remain fixed when selecting events or
combining its files. Metadata gives the effective sample size.

A floor fraction of 0 keeps all positive-weight proposals; 0.1 removes some
low-weight proposals with compensating weights; 1 gives unweighted selection.
The physical channel probabilities and total rates are independent of this
choice. Precision must be checked for each observable, especially rare final
states. This command writes NumPy archives.

The Python interface is `WeightedALPGenerator` from `eventcalc.weighted_runtime`.
`HadronizationNormalization` accumulates a full sample; `export_weighted`
writes normalized files. Separate masses require separate normalizations.

## Check an installation

```sh
python -m unittest discover -s tests -p 'test_weighted_runtime.py' -v
python tests/smoke_exhad.py --exhad-root "$EXHAD_ROOT" --events 100 \
  --models ALP-fermion Scalar-mixing Dark-photons HNL --mass 3
```

The smoke command generates four samples and checks event labels and
four-momentum conservation. Production and decay uncertainties are described
with their respective physical calculations; a successful run alone does not
validate those approximations.
