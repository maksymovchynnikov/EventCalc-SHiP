# EventCalc-SHiP

EventCalc-SHiP generates decays of long-lived particles in the
[SHiP experiment](https://ship.web.cern.ch/) and calculates accepted event yields.
It implements the production and decay calculations listed in the
[physics guide](docs/DETAILS.md), including the hadronic decay model of
[arXiv:2609.16104](https://arxiv.org/abs/2609.16104).
Generated momenta and vertices can be passed to
[FairShip](https://github.com/ShipSoft/FairShip) for detector simulation.

## Installation and first run

From the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python simulate.py
```

The [installation guide](docs/user-guide.md#installation) explains Python and
Pythia8 dependencies. For a run without prompts:

```sh
python run_batch.py --llp Dark-photons --masses 0.3 --ctaus 100 \
  --nevents 10000 --channels e-pair
```

This generates dark photon decays to an electron pair, with mass in GeV and
proper decay length in metres. Output files contain laboratory momenta, decay
positions and weights. The [user guide](docs/user-guide.md) describes their
normalization, input cards and coupling scans.

Hadronic decays use configured exHad by default. Nonhadronic channels are
generated directly. Use `--exhad off` to select baseline Pythia8 for hadronic
decays.

## Models and production

The [model guide](docs/models.md) defines the supported interactions and coupling
conventions. It covers `Dark-photons`, `Inelastic-DM`, `HNL`, `Scalar-mixing`,
`Scalar-quartic`, `ALP-photon`, `ALP-fermion`, `ALP-SU2L` and `ALP-mixed`.

Default `Dark-photons` and `Inelastic-DM` simulations sample compact distributions
in mass, angle and energy, including primary collisions and the target cascade.
Their propagation uses a point source at the target front face. The
[production guide](docs/PRODUCTION_DATA.md) gives the calibration, target geometry,
coverage and numerical check of that approximation. Full samples with correlated
collision positions and production variations are available separately on CERN
EOS. Installation is explicit; ordinary event generation uses the tables in Git.

After [configuring exHad](docs/EXHAD.md), generate excited dark matter decays with:

```sh
python simulate.py --model Inelastic-DM --masses 20 --idm-delta .05 \
  --ctaus 0.1 1000 --events 10000 --seed 1
```

For `Inelastic-DM`, mass inputs denote the mediator and proper decay lengths
refer to the excited state. See the [inelastic model guide](docs/INELASTIC_DM.md).

## Acceptance and sensitivity

With the optional EOS dataset installed:

```sh
python -m eventcalc.model_sensitivity --model dp --mass 1.8 --output results/dp-1.8
python -m eventcalc.model_sensitivity --model idm --mass 20 --delta .05 --output results/idm-20
```

Each command calculates the lower and upper mixing boundaries at 2.3 expected
events and propagates production variations to both. The selections require
at least two or all visible particles to reach the ECAL end, with the energy
and impact-parameter cuts defined in the [sensitivity guide](docs/SENSITIVITY.md).
Charged particles undergo magnetic deflection. Reconstruction efficiency is
assumed to be one. The same guide explains the model-independent interface for
calculations with other production and decay generators, directly in memory.

## Documentation

- [Running simulations and reading outputs](docs/user-guide.md)
- [Models and couplings](docs/models.md); [physics inputs and sampling](docs/DETAILS.md)
- [Production distributions and EOS data](docs/PRODUCTION_DATA.md)
- [Inelastic dark matter](docs/INELASTIC_DM.md)
- [Acceptance and sensitivity](docs/SENSITIVITY.md)
- [Hadronic event generation](docs/EXHAD.md); [decay models](docs/EXHAD-MODELS.md)
- [Measuring performance](docs/EXHAD-PERFORMANCE.md)
- [Repository layout and commands](docs/repository-layout.md)

## Known bugs/problems

Three `ALP-fermion` squared matrix elements from the input calculation of
[arXiv:2501.04525](https://arxiv.org/abs/2501.04525) become negative in physical
phase space: $\pi^+\pi^-\pi^0$, $3\pi^0$ and $K^+K^-\pi^0$.
As a sampling workaround, EventCalc uses their absolute values. The underlying
expressions require correction. Tabulated lifetimes and branching fractions
remain unchanged.

## Credits

EventCalc-SHiP builds on [Josue Jaramillo's CERN student project](https://github.com/josuejaramillo/summer_school_2024_SHiP).
Asher Goddart implemented the inelastic dark matter model.
