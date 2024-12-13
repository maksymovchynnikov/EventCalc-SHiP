## LLP phenomenology

The phenomenology of various LLPs implemented in `EventCalc` follows the description used in `SensCalc`: 

- [1805.08567](https://arxiv.org/abs/1805.08567) for HNLs. Minor changes include improved matching between exclusive and perturbative QCD descriptions of decays into hadrons (separately for neutral and charged current processes and for three various mixing).
- [1904.10447](https://arxiv.org/abs/1904.10447) for Higgs-like scalars. Minor changes include the modification of the decay width into gluons (higher order QCD corrections are taken into account). 
- [1904.02091](https://arxiv.org/abs/1904.02091) for the ALPs coupled to photons (in addition, the decay modes `ALP -> gamma l l` and processes of the ALP production in the decays of `pi^0` and `eta` are added).
- [2409.11096](https://arxiv.org/abs/2409.11096) and references therein for the dark photons. 

The phenomenology is very different from the descriptions in the PBC report. This is because the latter is quite outdated. 

### How arbitrary mixing pattern of HNLs is implemented

To make it possible to consider various mixing pattern, `EventCalc` has the HNL production yield, HNL tabulated distribution, HNL decay width, branching ratios of all possible HNL decay modes and their matrix elements (to sample the kinematics of decay products) for the pure three mixings (1,0,0), (0,1,0), (0,0,1). Once the user selects the arbitrary mixing pattern `(Ue2, Umu2, Utau2)`, the code merges the mentioned quantities:
- The total yields `(yield_e, yield_mu, yield_tau)` for pure mixings are summed with weights `(Ue2, Umu2, Utau2)`, giving the total production yield `yield_total`.
- The tabulated distributions are summed with weights `(yield_e*Ue2, yield_e*Umu2, yield_tau*Utau2)/yield_total`.
- The decay widths `(Gamma_e, Gamma_mu, Gamma_tau)` are summed with weights `(Ue2, Umu2, Utau2)`, giving the total decay width `Gamma_total`.
- The branching ratios and decay matrix elements are summed with weights given by `(Gamma_e*Ue2, Gamma_mu*Umu2, Gamma_tau*Utau2)/Gamma_total`.

Once this is done, we have all the ingredients for calculating the event rate for the given mixing pattern.

## Mother particles fluxes calculations

When producing LLP fluxes, the following setups have been used to get the fluxes of mesons (their mother particles):

- For light mesons (`Pi0`, `Eta`, `Eta'`, `Rho0`, `Omega`, etc.): `pythia8` setup from [1904.02091](https://arxiv.org/abs/1904.02091). This means that the cascade enhancement of the fluxes is not included. It would typically result in too soft particles that would be outside of the SHiP acceptance, so there should not be any significant impact on the event rate. Nevertheless, it will be implemented in the future.
- For charm and bottom: [SHiP study](https://cds.cern.ch/record/2115534).
- For kaons: [2004.07974](https://arxiv.org/abs/2004.07974) (will be regenerated as well).

## Sampling details

### How LLP's kinematics and decay vertices are sampled

- The code starts with sampling the LLP's kinematics within the polar angle coverage of the SHiP experiment. To improve the quality of sampling, it is done in the following way. The code first randomly samples polar angles `theta_random` within the SHiP coverage. It then samples energies within the range `[E_min,E_max(theta_random)]`. Here: 
  - `E_min = max(m_LLP, E_min(l_decay))`, where `E_min(m_LLP, c tau_LLP)` is the minimal energy for which the LLP decay probability is not too exponentially suppressed (it is very important for short lifetimes `ctau << z_{to SHiP}`); Exp[-15] is taken as this boundary.
  - `E_max(theta_random)` is the maximal LLP energy for the given polar angles. It is needed to take into account as for some LLP production channels (e.g., the proton bremsstrahlung, etc.), this dependence is very steep.
- Then, for each pair `(theta_random, E_random)`, the code calculates the weights `f_tabulated(theta_random,E_random)*(E_max(theta_random)-E_min)`, and selects a fraction of the events based on the weigths. The polar acceptance `epsilon_polar` is then defined as the sum of the selected weights.
- The code proceeds with sampling random azimuthal angles and generating the decay vertices within the longitudinal coverage of SHiP. This is done in the following way:
  - First, one defines the inverse CDF for the `z` displacement of the decay vertex based on the exponential distribution in the LLP decay width: `ctau_LLP*cos(theta_LLP)*p_LLP/m_LLP log(1/(1-u))`. If one wants to sample the decays everywhere, `u` randomly ranges from 0 to 1. However, for the SHiP case, `u` is fixed in a way such that `z` ranges from `z_min,SHiP` (32 m for the current setup) to `z_max,SHiP` (82 m).
  - Having sampled `z` and knowing `theta_LLP,phi_LLP`, the decay point is sampled.
  - Finally, the decay probability is also calculated: `P_decay = Exp[-z_min/(ctau_LLP cos(theta_LLP)p_LLP/m_LLP)]-Exp[-z_max/(ctau_LLP cos(theta_LLP)p_LLP/m_LLP)]`
- Knowing the decay point position, the code selects only those LLPs which are within the azimuthal acceptance of the decay volume. This is at most O(1) effect given the large azimuthal coverage of SHiP decay volume. Then, it simulates the phase space of LLP's decay products (described below).   

### How LLP decay phase space is sampled

It is split into three steps:
- `EventCalc` simulates the phase space of decay products at the LLP rest frame. 
  - Two-body decays are sampled isotropically, with the two decay products having the same energy.
  - Three-body decays `LLP -> 1+2+3` are sampled using the squared matrix element of the process in terms of the LLP mass and energies `E_1`, `E_3` of the decay products.
  - Currently, four-body decays `LLP->1+2+3+4` are sampled assuming unit squared matrix element (to be improved in the future).
- The phase space is passed to `pythia8` for decaying unstable particles such as `pi^0`, showering and hadronization.
- The resulting phase space is boosted into the LLP's lab frame.

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

- Adding more LLPs (ALPs, B-L mediators, HNLs with dipole coupling, inelastic and elastic LDM, etc.). Pending due to their phenomenology revision.
- Adding theoretical uncertainty (decay widths for Higgs-like scalars, B-L production uncertainties, etc.).
- Improving the performance of the code (parallelization of pythia8 run, etc.).
- Adjusting the SHiP setup with the up-to-date setup if needed.
- Adding cascade production from kaons and light mesons for HNLs, Higgs-like scalars, dark photons.
- Adding more sophisticated simulation codes (such as the machinery to simulate HNL-anti-HNL oscillations).
- Introduce individual identifiers for various LLPs (currently, it is 12345678).
- Add the possibility to sample LLPs solely within the azimuthal acceptance (as it is done in the `EventCalc` module in the [SensCalc repository](https://github.com/maksymovchynnikov/SensCalc)).
- Add the version adapted for HTCondor.