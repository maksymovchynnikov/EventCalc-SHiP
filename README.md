# EventCalc-SHiP

EventCalc-SHiP generates decays of long-lived particles in the SHiP decay
volume and calculates accepted event yields. It supports dark photons,
inelastic dark matter, heavy neutral leptons, Higgs-like scalars and several
axion-like-particle couplings. Events can be passed to FairShip for detector
simulation.

## Run a simulation

Run commands from the repository root. Install the Python dependencies and
configure Pythia8 as described in the [installation guide](docs/user-guide.md#installation).

```sh
python -m pip install -r requirements.txt
python simulate.py
```

The same calculation can run without interactive prompts:

```sh
python run_batch.py --llp Dark-photons --masses 0.3 --ctaus 100 \
  --nevents 10000 --channels e-pair --exhad off
```

For inelastic DM, install/configure [exHad](docs/EXHAD.md), then run:

```sh
python simulate.py --model Inelastic-DM --masses 20 --idm-delta .05 \
  --ctaus 0.1 1000 --events 10000 --seed 1
```

The inelastic mass input is the **mediator mass**; its proper lifetime belongs
to the excited dark-matter state. Couplings may be supplied instead of lifetimes.
See the [user guide](docs/user-guide.md) for cards, model conventions and outputs.

## Production inputs

Ordinary dark-photon and inelastic-DM generation samples compact mass–angle–energy
tables directly. The calibrated primary and cascade inputs use about **25 MiB**
and live beside the corresponding models:

- `Distributions/Dark-photons/production/`
- `Distributions/Inelastic-DM/production/`

These distributions include the target shower and beam profile. Propagation
uses a point source at the target front-face centre; the checked effect on
sensitivity is small in the tested benchmarks. The [production-data guide](docs/PRODUCTION_DATA.md)
gives the approximation, its numerical check and the optional full calculation.

The physics implementation is in `eventcalc/production/dark_photon/`. It
calculates dark-photon production and the excited dark-matter particles produced
by mediator decay. Large collision samples, production positions and uncertainty
weights are stored in a versioned CERN EOS dataset and a shared local cache.
They are not repository inputs. Select `--production-backend correlated` to use
them after installation.

## Acceptance and sensitivities

```sh
python -m eventcalc.model_sensitivity --model dp --mass 1.8 --output results/dp-1.8
python -m eventcalc.model_sensitivity --model idm --mass 20 --delta .05 --output results/idm-20
```

These commands use the optional full dataset and automatically propagate
production uncertainty to **both sensitivity boundaries**, for both ECAL
selections. The threshold is 2.3 expected events. The model-independent
acceptance calculation includes magnetic deflection of charged particles.
See [SENSITIVITY.md](docs/SENSITIVITY.md) for installation, conventions and the
generic `python -m eventcalc.scan_sensitivity` command.

The Python event-generation API is `eventcalc.runtime_generator`; HepMC output
is implemented in `eventcalc.hepmc_export`. Analysis commands live under
`eventcalc.analysis` and run as Python modules.

## Documentation and layout

- [User guide and installation](docs/user-guide.md)
- [Model inputs and physics conventions](docs/DETAILS.md)
- [Inelastic dark matter](docs/INELASTIC_DM.md)
- [Compact tables and optional EOS data](docs/PRODUCTION_DATA.md)
- [Acceptance and sensitivity calculations](docs/SENSITIVITY.md)
- [exHad integration](docs/EXHAD.md), [supported models](docs/EXHAD-MODELS.md),
  [performance](docs/EXHAD-PERFORMANCE.md)
- [Repository layout and updated commands](docs/repository-layout.md)

The root contains the two simulation launchers. `eventcalc/` holds runtime,
acceptance, production and analysis code; `funcs/` holds the existing shared
physics, configuration and event-I/O helpers. `Distributions/`, `cards/`,
`examples/` and `tests/` contain model tables, input cards, examples and checks.
Generated events and plots belong in ignored output directories.

## Known bugs/problems

The corrected ALP-fermion inputs include the updated RG evolution, lifetimes,
branching ratios and squared matrix elements. Three source squared matrix
elements (`piPpiMpi0`, `3pi0`, `KPKMpi0`) can still be negative in physical phase
space. They use `abs(source_squared_matrix_element)` as a sampling workaround;
the underlying expressions need correction. See the [user guide](docs/user-guide.md#known-bugsproblems)
for the source and scope of this limitation.

## Credits

This repository builds on [Josue Jaramillo's CERN student project](https://github.com/josuejaramillo/summer_school_2024_SHiP).
