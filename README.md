# EventCalc-SHiP

Sampler of decay events with hypothetical Long-Lived Particles (LLPs) for the [SHiP experiment](https://ship.web.cern.ch/). May be easily adapted for any other experiment.

## Overview

The code
- Takes the tabulated distributions of LLPs in mass, polar angle, and energy (the distribution is averaged over all possible production channels), and the tabulated dependence of the maximum LLP energy on the mass and the polar angle, and samples the 3-momentum of the LLPs in the direction of the decay volume of SHiP.
- Having the generated 4-momenta, samples LLPs' decay positions based on the exponential distribution in the LLP decay length. The sampling is done within the z coverage of the SHiP decay volume.
- Simulates 2-, 3-, or 4-body decays of LLPs using an internal phase space simulator and then passes the events to pythia8 for showering and hadronization of jets. 
- Using the tabulated total yields, decay branching ratios, lifetime-mass dependence, and computed geometric acceptance and decay probability, calculates the total number of decay events.

The code does not simulate decay products acceptance. Instead, its output is provided to [FairShip](https://github.com/ShipSoft/FairShip) to simulate the propagation and reconstruction of decay products.

### Validation

`EventCalc-SHiP` has been intensively cross-checked against [`SensCalc`](https://github.com/maksymovchynnikov/SensCalc) (which, in its turn, has been tested against [`FairShip`](https://github.com/ShipSoft/FairShip) and other tools), see [slides](https://indico.cern.ch/event/1481729/contributions/6256116/). The agreement in all the quantities (overall number of events, geometric acceptance, averaged decay probability, spectra, etc.) is within 10%.


## Installation

The code has been tested on Linux only.

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

Running the main file `simulate.py`

`python3 simulate.py`

will first display the SHiP setup and then ask users about the requested number of LLPs to be sampled in the polar range of the SHiP experiment. Then, users will be asked about: 

- Entering the number of decay events to simulate in the polar angle coverage of SHiP.
- Selecting the LLP.
- Setting up genuine LLP properties such as the mixing pattern, variation of the theoretical uncertainty, and others (optionally, depending on LLP).
- Selecting the LLP's decay modes for which the simulation will be launched (their names should be self-explanatory).
- Range of LLP masses for which the simulation will be launched.
- Range of LLP lifetimes.
 
After that, the simulation will start. If the given combination of the mass and lifetime has too small production yield (< 1) or decay probability (<10^-21), its simulation terminates. In the other case, it produces two outputs in the folder `outputs/<LLP>` (see description below):
- The information about the decay events of LLPs and decay products (the file `eventData/<LLP>_<mass>_<lifetime>_....txt`), with dots meaning the other parameters relevant for the simulation (such as the mixing pattern in the case of HNLs, etc.).
- The information about total quantities from the simulation: mass, coupling, lifetime, number of events, etc. (the file `eventData/<LLP>/total/<LLP>-...-total.txt`).

Note that the resulting number of the decay events in the output file will be lower than the initial one (by a factor `~0.6-1`). This is the effect of the azimuthal acceptance of the SHiP decay volume.

### Example

The followin setup of launching `simulate.py` will run the simulation of `200000` decay events with HNLs with the mixing pattern `(1,0,0)`, all possible decay modes, masses `0.5, 4.5` GeV, and lifetimes ctau `0.01, 10000` for each mass:

```bash

SHiP setup (modify ship_setup.py if needed):

z_min = 32 m, z_max = 82 m, Delta_x_in = 1 m, Delta_x_out = 4 m, Delta_y_in = 2.7 m, Delta_y_out = 6.2 m, theta_max = 0.044960 rad

Enter the number of events to simulate: 200000

Particle Selector

1. Scalar-mixing
2. ALP-photon
3. Scalar-quartic
4. Dark-photons
5. HNL
Select particle: 5

Enter xi_e, xi_mu, xi_tau: (Ue2, Umu2, Utau2) = U2(xi_e,xi_mu,xi_tau), summing to 1, separated by spaces: 1 0 0

Select the decay modes:
0. All
1. 2ev
2. 2muv
3. 2Pie
4. 2Piebar
5. 2Pimu
6. 2Pimubar
7. 2Pitau
8. 2Pitaubar
9. 2Piv
10. 2tauv
11. a1v
12. emuv
13. emuvbar
14. EtaPrv
15. etauv
16. etauvbar
17. Etav
18. Jets-ccv
19. Jets-cse
20. Jets-csebar
21. Jets-csmu
22. Jets-csmubar
23. Jets-cstau
24. Jets-cstaubar
25. Jets-ddv
26. Jets-ssv
27. Jets-ude
28. Jets-udebar
29. Jets-udmu
30. Jets-udmubar
31. Jets-udtau
32. Jets-udtaubar
33. Jets-uuv
34. Ke
35. Kebar
36. Kmu
37. Kmubar
38. Ktau
39. Ktaubar
40. mutauv
41. Omegav
42. Phiv
43. Pi0v
44. Pie
45. Piebar
46. Pimu
47. Pimubar
48. Pitau
49. Pitaubar
Enter the numbers of the decay channels to select (separated by spaces): 0

Generating LLP phenomenology plots...
Phenomenology plots generated.

Enter LLP masses in GeV (separated by spaces): 0.5 4.5
Enter lifetimes c*tau in m for all masses (separated by spaces): 0.01 10000
```




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
  
- Post-processing (launched as `python3 ...`):
  - `events_analysis.py`: the script computing various distributions with the decaying LLP and its decay products: position of the decay vertices, energy distributions, multiplicity, etc. The output is saved in the folder `plots/<LLP>/<LLP>_<mass>_<lifetime>_<other parameters>_decay products`.
  - `total-plots.py`: the script making the plot of some averaged quantities, such as the polar acceptance, total geometric acceptance, mean decay probability, etc., and the plot with the dependence of the number of events as a function of the LLP's coupling and mass. The output is two plots saved in the folder `plots/<LLP>`.
  - `event-display.py`: the script making .pdf and interactive .html plots showing the decay point of the LLP, the direction of its momentum, and the directions of its decay products. The output is the event display of 10 random events for the selected decay mode saved in the folder `plots/<LLP>/eventDisplay/<LLP>_<mass>_<lifetime>_<other parameters>`.

### Output files of the simulation

- The detailed event record file (located in `outputs/<LLP>/eventData/<LLP>_<mass>_<lifetime>_<other parameters>_decayProducts.dat`): 
  - The first string is `Sampled ## events inside SHiP volume. Total number of produced LLPs: ##. Polar acceptance: ##. Azimuthal acceptance: ##. Averaged decay probability: ##. Visible Br Ratio: ##. Total number of events: ##`. The meanings of the numbers are: the total sample size; the total number of LLPs produced during 15 years of SHIP running; the amount of LLPs pointing to the polar range of the experiment; of those, the amount of LLPs that also pass the azimuthal acceptance cut; of those, the averaged probability to decay inside the SHiP volume; the visible branching ratio of selected decay channels; the total number of decay events inside the decay volume.
  - After the first string, the data is split into blocks. Each is started with `#<process=##; sample_points=##>`. The meanings of `##` are: the name of the LLP decay process; the number of samples per this process. After this string, there is the tabulated data with the decay information. The meaning of the elements is as follows: 
 
   `p_x,LLP p_y,LLP p_z,LLP E_LLP mass_LLP PDG_LLP P_decay,LLP x_decay,LLP y_decay,LLP z_decay,LLP p_x,prod1 p_y,prod1 p_z,prod1 E_prod1 mass_prod1 pdg_prod1 p_x,prod2 ...`
 
  - where `...` means the data for the other decay products. The units are GeV (for mass, momentum, and energy) or meters (for coordinates). The center of the coordinate system corresponds to the center of the SHiP target. The first 10 numbers correspond to the decaying LLP info: its 4-momentum, mass, pdg identifier, decay probability in SHiP, decay coordinates. Each next 6 numbers correspond to the individual metastable decay product (electrons, muons, neutrinos and their antiparticles, photons, charged kaons, `K_L`, charged pions): 4-momentum, mass, and pdg identifier. Some of the rows end with the strings `0. 0. 0. 0. 0. -999.`, to account for varying number of decay products in the same decay channel and maintain the flat array if merging all the datasets. Each event has its own weight given by `P_decay,LLP`. The total number of events may be obtained by multiplying the total number of produced LLPs, polar acceptance, azimuthal acceptance, the sum of `P_decay,LLP` divided with the total number of stored events, and the visible branching ratio (all these numbers are actually provided in the first row). 
  
- The file with the total information about the simulation (located in `outputs/<LLP>/Total`): contains the self-explanatory header describing the meaning of columns. Results of various simulations corresponding to the same LLP setup (such as the choice of the phenomenology within the theoretical uncertainty and the mixing pattern) are added to the corresponding files.

- Plots with the LLP mass dependence phenomenology used to produce the event rates: 
  - The overall LLP production probability per proton-on-target per coupling squared.
  - The LLP lifetime.
  - The branching ratios of the decay modes selected for the simulation.
  
### More information

For more information (description of the phenomenology of LLPs, details of sampling, how fluxes of mother mesons have been generated, etc.), read [`DETAILS.md`](https://github.com/maksymovchynnikov/EventCalc-SHiP/blob/main/DETAILS.md).


## Credits

This is the finalization of [Josue Jaramillo's CERN student project](https://github.com/josuejaramillo/summer_school_2024_SHiP).