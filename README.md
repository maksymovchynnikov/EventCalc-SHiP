# EventCalc-SHiP

Sampler of decay events with hypothetical long-lived particles for the SHiP experiment. This is the finalization of Josue Jaramillo's CERN student project [josuejaramillo/summer_school_2024_SHiP](https://github.com/josuejaramillo/summer_school_2024_SHiP).

## Overview

The code
- Takes the tabulated distributions of LLPs in mass, polar angle, and energy (the distribution is averaged over all possible production channels), and the tabulated dependence of the maximum LLP energy on the mass and the polar angle, and samples the 3-momentum of the LLPs in the direction of the decay volume of SHiP.
- Having the generated 4-momenta, samples LLPs' decay positions based on the exponential distribution in the LLP decay length. The sampling is done within the z coverage of the SHiP decay volume.
- Simulates 2-, 3-, or 4-body decays of LLPs using an internal phase space simulator and then passes the events to pythia8 for showering and hadronization of jets. 
- Using the tabulated total yields, decay branching ratios, lifetime-mass dependence, and computed geometric acceptance and decay probability, calculates the total number of decay events.

The code does not simulate decay products acceptance. Instead, its output is provided to [FairShip](https://github.com/ShipSoft/FairShip) to simulate the propagation and reconstruction of decay products.

### Validation

`EventCalc-SHiP` has been intensively cross-checked against `SensCalc` (which, in its turn, has been tested against `FairShip` and other tools), see [slides](https://indico.cern.ch/event/1481729/contributions/6256116/). The agreement in all the quantities (overall number of events, geometric acceptance, averaged decay probability, spectra, etc.) is within 10%.


## Installation

Ensure you have the required packages installed. You can use the following command to install the dependencies:

```bash
pip3 install numpy sympy numba scipy plotly
```

Also, [pythia8](https://pythia.org/) must be installed with python config. It just means that, when configuring it, users have to type

`./configure --with-python-config=python3-config`

and then `make`. As a result, the pythia's `lib` folder has to contain the `pythia8.so` file.

Once this is done, the pythia's `lib` folder has to be specified in the script `funcs/decayProducts.py`. Currently, it is

`sys.path.insert(0, '/home/name/Downloads/pythia8312/lib')`

## Usage

Running the main file `simulate.py` will first display the SHiP setup and then ask users about the requested number of LLPs to be sampled in the polar range of the SHiP experiment. Then, users will be asked about: 

- Entering the number of decay events to simulate in the polar angle coverage of SHiP.
- Selecting the LLP.
- Setting up genuine LLP properties such as the mixing pattern, variation of the theoretical uncertainty, and others (optionally, depending on LLP).
- Selecting the LLP's decay modes for which the simulation will be launched (their names should be self-explanatory).
- Range of LLP masses for which the simulation will be launched.
- Range of LLP lifetimes.
 
After that, the simulation will start. It produces two outputs in the folder `outputs/<LLP>` (see description below):
- The information about the decay events of LLPs and decay products (the file `eventData/<LLP>_<mass>_<lifetime>_....txt`), with dots meaning the other parameters relevant for the simulation (such as the mixing pattern in the case of HNLs, etc.).
- The information about total quantities from the simulation: mass, coupling, lifetime, number of events, etc. (the file `eventData/<LLP>/total/<LLP>-...-total.txt`).

Note that the resulting number of the decay events in the output file will be lower than the initial one (by a factor `~0.6-1`). This is the effect of the azimuthal acceptance of the SHiP decay volume.


### Code structure

- `funcs/`:
  - `initLLP.py`: Contains the `LLP` class, which initializes the LLP object with attributes like mass, PDGs (Particle Data Group identifiers), and branching ratios.
  - `decayProducts.py`: Contains functions for simulating decays of various LLPs, as well as the routine to perform showering and hadronization of quark and gluon decays via interfacing with `pythia8`.
  - `HNLmerging.py`: Contains functions for handling HNLs with arbitrary mixing pattern given the tabulated distributions, branching ratios, lifetimes, total yields, and decay matrix elements for pure mixings.
  - `PDG.py`: Contains functions or data related to Particle Data Group identifiers.
  - `rotateVectors.py`: Contains functions for rotating vectors.
  - `FourBodyDecay.py`: Contains functions for simulating four-body decays.
  - `ThreeBodyDecay.py`: Contains functions for simulating three-body decays.
  - `TwoBodyDecay.py`: Contains functions for simulating two-body decays.
  - `boost.py`: Contains functions for boosting decay products to the lab frame.
  - `kinematics.py`: Contains functions for handling kinematic distributions and interpolations.
  - `selecting_processing.py`: code to be used by the post-processing scripts.
  - `ship_setup.py`: specifies the SHiP setup used in the simulation and when making plots.

- Main code `simulate.py`: the script to run the decay simulation. 
  
- Post-processing:
  - `events_analysis.py`: the script computing various distributions with the decaying LLP and its decay products: position of the decay vertices, energy distributions, multiplicity, etc. The output is saved in the folder `plots/<LLP>/<LLP>_<mass>_<lifetime>_<other parameters>_decay products`.
  - `total-plots.py`: the script making the plot of some averaged quantities, such as the polar acceptance, total geometric acceptance, mean decay probability, etc., and the plot with the dependence of the number of events as a function of the LLP's coupling and mass. The output is two plots saved in the folder `plots/<LLP>`.
  - `event-display.py`: the script making .pdf and interactive .html plots showing the decay point of the LLP, the direction of its momentum, and the directions of its decay products. The output is the event display of 10 random events for the selected decay mode saved in the folder `plots/<LLP>/eventDisplay/<LLP>_<mass>_<lifetime>_<other parameters>`.

### Output files of the simulation

- The detailed event record file (located in `outputs/<LLP>/eventData/<LLP>_<mass>_<lifetime>_<other parameters>_decayProducts.dat`): 
  - The first string is `Sampled ## events inside SHiP volume. Total number of produced LLPs: ##. Polar acceptance: ##. Azimuthal acceptance: ##. Averaged decay probability: ##. Visible Br Ratio: ##. Total number of events: ##`. The meanings of the numbers are: the total sample size; the total number of LLPs produced during 15 years of SHIP running; the amount of LLPs pointing to the polar range of the experiment; of those, the amount of LLPs that also pass the azimuthal acceptance cut; of those, the averaged probability to decay inside the SHiP volume; the visible branching ratio of selected decay channels; the total number of decay events inside the decay volume.
  - After the first string, the data is split into blocks. Each is started with `#<process=##; sample_points=##>`. The meanings of `##` are: the name of the LLP decay process; the number of samples per this process. After this string, there is the tabulated data with the decay information. The meaning of the elements is as follows: 
 
   `p_x,LLP p_y,LLP p_z,LLP E_LLP mass_LLP PDG_LLP P_decay,LLP x_decay,LLP y_decay,LLP z_decay,LLP p_x,prod1 p_y,prod1 p_z,prod1 E_prod1 mass_prod1 pdg_prod1 p_x,prod2 ...`
 
  - where `...` means the data for the other decay products. The units are GeV (for mass, momentum, and energy) or meters (for coordinates). The center of the coordinate system corresponds to the center of the SHiP target. The first 10 numbers correspond to the decaying LLP info: its 4-momentum, mass, pdg identifier, decay probability in SHiP, decay coordinates. Each next 6 numbers correspond to the individual metastable decay product (electrons, muons, neutrinos and their antiparticles, photons, charged kaons, `K_L`, charged pions): 4-momentum, mass, and pdg identifier. Some of the rows end with the strings `0. 0. 0. 0. 0. -999.`, to account for varying number of decay products in the same decay channel and maintain the flat array if merging all the datasets.
  
- The file with the total information (located in `outputs/<LLP>/Total`): contains the self-explanatory header describing the meaning of columns. Results of various simulations corresponding to the same LLP setup (such as the choice of the phenomenology within the theoretical uncertainty and the mixing pattern) are added to the corresponding files.

- Plots with the LLP mass dependence phenomenology used to produce the event rates: 
  - The overall LLP production probability per proton-on-target per coupling squared.
  - The LLP lifetime.
  - The branching ratios of the decay modes selected for the simulation.



## Implemented LLPs

Currently, the following LLPs are implemented:

- HNLs `N` with arbitrary mixing patterns (`HNL`). Corresponds to the so-called BC6 (if the pattern is 1:0:0), BC7 (if the pattern is 0:1:0), BC8 (0:0:1) models according to the [PBC definition](https://arxiv.org/abs/1901.09966). Of course, the pattern may be made arbitrary.
- Higgs-like scalars `S` that have mass mixing with the Higgs bosons (`Scalar-mixing`). This corresponds to the BC4 model.
- Higgs-like scalars `S` produced by the trilinear coupling (`Scalar-quartic`). The default value of the trilinear coupling is fixed in a way such that `Br(h->SS) = 0.01`, which corresponds to the so-called BC5 model. If one wants to compute the total number of events in the BC5 model, one needs just to sum the event rate from the mixing model and the quartic model for the given scalar mass and lifetime.
- ALPs `a` coupled to photons (`ALP-photon`). Corresponds to the BC9 model.
- Dark photons `V` (`Dark-photons`). They have a large theoretical uncertainty in the production. Because of this, the users are asked to select the flux within the range of this uncertainty - `lower`, `central`, or `upper` (see [2409.11096](https://arxiv.org/abs/2409.11096) for details). Corresponds to the BC1 model. 


Currently, the tabulated distributions are generated by [`SensCalc`](https://github.com/maksymovchynnikov/SensCalc). The matrix elements, branching ratios, lifetime dependence on mass, and production fluxes are also taken from it. 

## LLP phenomenology description

The phenomenology of various LLPs implemented in `EventCalc` follows the description used in `SensCalc`: 

- [1805.08567](https://arxiv.org/abs/1805.08567) for HNLs. Minor changes include improved matching between exclusive and perturbative QCD descriptions of decays into hadrons (separately for neutral and charged current processes and for three various mixing).
- [1904.10447](https://arxiv.org/abs/1904.10447) for Higgs-like scalars. Minor changes include the modification of the decay width into gluons (higher order QCD corrections are taken into account). 
- [1904.02091](https://arxiv.org/abs/1904.02091) for the ALPs coupled to photons (in addition, the decay modes `ALP -> gamma l l` and processes of the ALP production in the decays of `pi^0` and `eta` are added).
- [2409.11096](https://arxiv.org/abs/2409.11096) and references therein for the dark photons. 

The phenomenology is very different from the descriptions in the PBC report. This is because the latter is quite outdated. 

### How LLP decays are sampled

It is split into three steps:
- `EventCalc` simulates the phase space of decay products at the LLP rest frame. 
  - Two-body decays are sampled isotropically, with the two decay products having the same energy.
  - Three-body decays `LLP -> 1+2+3` are sampled using the squared matrix element of the process in terms of the LLP mass and energies `E_1`, `E_3` of the decay products.
  - Currently, four-body decays `LLP->1+2+3+4` are sampled assuming unit squared matrix element (to be improved in the future).
- The phase space is passed to `pythia8` for decaying unstable particles such as `pi^0`, showering and hadronization.
- The resulting phase space is boosted into the LLP's lab frame.

### How arbitrary mixing pattern of HNLs is implemented

To make it possible to consider various mixing pattern, `EventCalc` has the HNL production yield, HNL tabulated distribution, HNL decay width, branching ratios of all possible HNL decay modes and their matrix elements (to sample the kinematics of decay products) for the pure three mixings (1,0,0), (0,1,0), (0,0,1). Once the user selects the arbitrary mixing pattern `(Ue2, Umu2, Utau2)`, the code merges the mentioned quantities:
- The total yields `(yield_e, yield_mu, yield_tau)` for pure mixings are summed with weights `(Ue2, Umu2, Utau2)`, giving the total production yield `yield_total`.
- The tabulated distributions are summed with weights `(yield_e*Ue2, yield_e*Umu2, yield_tau*Utau2)/yield_total`.
- The decay widths `(Gamma_e, Gamma_mu, Gamma_tau)` are summed with weights `(Ue2, Umu2, Utau2)`, giving the total decay width `Gamma_total`.
- The branching ratios and decay matrix elements are summed with weights given by `(Gamma_e*Ue2, Gamma_mu*Umu2, Gamma_tau*Utau2)/Gamma_total`.

Once this is done, we have all the ingredients for calculating the event rate for the given mixing pattern.

### How hadronic decays of LLPs are handled

Masses of LLPs to be probed at SHiP are in the GeV range. Therefore, there is no unified description of their hadronic decays: at masses m_LLP <~ 1 GeV, one has to use exclusive description, whereas at larger masses, one may switch to perturbative QCD. In the domain of intermediate masses, matching between these two descriptions has to be done (see, e.g., [1801.04847](https://arxiv.org/abs/1801.04847) as an example of such a discussion). 

In `EventCalc`, this is performed in the following way:

- The exclusive widths are non-zero below the matching mass and zero above it, while perturbative widths are non-zero above it and zero below.  
- For HNLs, neutral current and charged current widths are matched independently from each other, and for each pure mixing, matching is performed separately.
- Consider perturbative QCD decays, `LLP -> quarks`, say `S->c cbar` for explicit example. Often, in the literature it is assumed that the threshold for this channel is just `2*m_c`, and the corresponding decay width is included to the total decay width of the LLP already above this di-quark threshold (see, e.g., [2201.065801](https://arxiv.org/pdf/2201.06580)). It is wrong, as the true threshold is `2*m_D`. The approach of `EventCalc` to deal with this and other similar decay modes is the following:
  - When calculating the decay width and the branching ratio, just replace the quark mass with the corresponding lightest meson mass. This way, the decay `S->cc` is replaced with `S->DD`.
  - However, to sample the proper kinematics of the decay, the actual quark masses have to be used. I.e., above the di-D threshold, `S` decays into two `c` quarks, each with mass `m_c = 1.27 GeV`. They are then properly hadronized in `pythia8`.
  - The described approach works trivially for 2-body decays. For 3-body decays like `N -> cc nu`, the situation is more subtle. If sampling the full Dalitz phase space of quarks and neutrinos (of course, reweighted by the squared matrix element of the decay), there may be situations when the invariant mass of the `cc` pair is still below the di-D threshold, which means that such event would be discarded. This would lead to the double counting of the decay suppression - the first comes from the replacement `m_c -> m_D` when calculating the width and branching ratio, while the second comes from the threshold. To avoid the double counting, the invariant mass of the `cc` pair in `EventCalc` is sampled in a way such that it is always above the di-D threshold. The same applies to the other hadronic decays.

  
## To be done

- Adding more LLPs (ALPs, B-L mediators, HNLs with dipole coupling, inelastic and elastic LDM, etc.).
- Adding theoretical uncertainty (decay widths for Higgs-like scalars, B-L production uncertainties, etc.).
- Improving the performance of the code (parallelization of pythia8 run, etc.).
- Adjusting the SHiP setup with the up-to-date setup if needed.
- Adding cascade production from kaons and light mesons for HNLs, Higgs-like scalars, dark photons.
- Adding more sophisticated simulation codes (such as the machinery to simulate HNL-anti-HNL oscillations).
- Introduce individual identifiers for various LLPs (currently, it is 12345678).
- Add the possibility to sample LLPs solely within the azimuthal acceptance (as it is done in the `EventCalc` module in the [SensCalc repository](https://github.com/maksymovchynnikov/SensCalc)).