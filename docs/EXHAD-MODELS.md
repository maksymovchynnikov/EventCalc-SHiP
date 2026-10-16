# Hadronic decay models

EventCalc uses the hadronic model of
[arXiv:2609.16104](https://arxiv.org/abs/2609.16104), implemented in
[exHad](https://github.com/maksymovchynnikov/exHad). The [integration guide](EXHAD.md)
gives commands and mass coverage.

For electromagnetic decays, measured $e^+e^-$ annihilation cross sections
constrain the rates and composition of multihadron final states.
[DeLiVeR](https://github.com/preimitz/DeLiVeR), described in
[arXiv:2201.01788](https://arxiv.org/abs/2201.01788), gives the exclusive-current
input. For other bosons, Pythia8 generates fragmentation events that are
accepted with probabilities fixed by the calculated hadronic fractions.
Exclusive channels absent from fragmentation are generated separately.

| EventCalc model | Physical input for hadronic momenta |
| --- | --- |
| `Dark-photons` | Electromagnetic amplitudes fitted to measured cross sections, including open charm above the $D\bar D$ threshold |
| `Inelastic-DM` | Electromagnetic final states evaluated over the virtual mediator mass in the excited-state decay |
| `ALP-fermion` | Pseudoscalar quark and gluon sources, constrained by the ALP hadronic fractions and charge combinations |
| `Scalar-mixing`, `Scalar-quartic` | Scalar sources with hadronic fractions from the selected width calculation |
| `HNL` | Charged and neutral weak currents at the invariant mass of the quark pair, with the associated lepton momentum preserved |

The unresolved light-hadron contribution for `Dark-photons` approaches Pythia8
fragmentation continuously between 4 and 5 GeV. Other currents use their own
hadronic fractions and quantum numbers. The [physics guide](DETAILS.md#decay-inputs)
links the rate calculations for each model.
