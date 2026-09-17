# One hadronic production construction per portal

exHad describes the hadrons of a decay with the construction that suits the
current the decaying particle couples to. EventCalc uses exactly one of them
for each portal, and the choice is fixed by that portal's card in
`Distributions/<LLP>/exhad.json`. This file says which construction each
portal gets and why; [EXHAD.md](EXHAD.md) says how to run them.

Two constructions appear below.

- **Constrained string fragmentation.** The hadrons come from Pythia string
  fragmentation of a source fixed by the decaying particle's quantum numbers
  (a gluon pair or a quark pair), and each fragmentation event is accepted
  with a probability that makes the fraction of every group of final states
  agree with the rates calculated for that particle. Exclusive channels that
  string fragmentation cannot produce with the right rate are generated
  separately.
- **Coherent electromagnetic amplitudes.** The rates and compositions of the
  multihadron final states are fitted to measured $e^+e^-$ annihilation
  cross sections (BaBar and BESIII), so they describe a particle coupled to
  the electromagnetic current.

| Portal | Construction | What must be supplied with it |
|---|---|---|
| Dark photon | Coherent electromagnetic amplitudes | The fitted response of the seven multihadron blocks, `data/dark-photon/em_response.json` in the exHad release, together with the measured open-charm channels above the $D\bar D$ threshold |
| $B-L$ or baryon-current boson | Coherent amplitudes of that current | The electromagnetic spectrum cannot be reused unchanged, because the light isovector current cancels for $B-L$: only the separately identified nonstrange- and strange-isoscalar amplitudes are reweighted. Above open charm the charm current is supplied as well. EventCalc ships no $B-L$ production tables, so this portal is available in exHad alone |
| Fermion-coupled ALP | Constrained string fragmentation | The ALP decay rates, the quantum numbers of a pseudoscalar source, and the mass above which the construction starts. The electromagnetic fit is not imported |
| Higgs-like scalar | Constrained string fragmentation | The selected scalar width calculation, its source composition, and the mass above which the construction starts |
| Heavy neutral lepton | Weak-current construction | The hadrons are generated separately for each charged and neutral current at the invariant mass $W$ of the quark–antiquark pair, with no electromagnetic input |

The dark-photon and the fermion-coupled ALP are also compared with each other
in the electromagnetic study, where the same fragmentation construction is
applied to the electromagnetic current to see how far it reproduces the
measured spectra. That comparison is a diagnostic of the physics; it is not a
second production choice for the dark photon, and no EventCalc card selects
it.

Electromagnetic branching fractions use the full hadronic width, including
charm above open $D\bar D$.
