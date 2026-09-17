## LLP phenomenology

The phenomenology of various LLPs implemented in `EventCalc` follows the description used in `SensCalc`: 

- [1805.08567](https://arxiv.org/abs/1805.08567) for HNLs. Minor changes include improved matching between exclusive and perturbative QCD descriptions of decays into hadrons (separately for neutral and charged current processes and for three various mixing).
- [1904.10447](https://arxiv.org/abs/1904.10447),
  [1809.01876](https://arxiv.org/abs/1809.01876), and
  [2407.13587](https://arxiv.org/abs/2407.13587) for the installed
  Higgs-like-scalar prescriptions. The prescription-specific lifetime and
  branching-ratio tables are selected as one matched pair.
- [1904.02091](https://arxiv.org/abs/1904.02091) for the ALPs coupled to photons (in addition, the decay modes `ALP -> gamma l l` and processes of the ALP production in the decays of `pi^0` and `eta` are added).
- [1901.02031](https://arxiv.org/abs/1901.02031) for the flavor-changing and diphoton rates of the $SU(2)_L$ ALP. Its production includes rare $B$ decays, $K^\pm\to\pi^\pm a$ below threshold, and Primakoff production by primary and electromagnetic-cascade photons. The decay model of this benchmark uses $\operatorname{Br}(a\to\gamma\gamma)=1$.
- `ALP-mixed` coherently combines the direct photon and $SU(2)_L$ operators. The physical lifetime fixes the total diphoton coupling, while the chosen operator fraction and interference sign fix the accompanying $B+K$ production. It reuses the installed pure-model tables; see `README.md` for the convention and the launch-card example.
- [2501.04525](https://arxiv.org/abs/2501.04525) and [2504.06828](https://arxiv.org/abs/2504.06828) for the fermion-coupled ALP decay and production descriptions.
- [2409.11096](https://arxiv.org/abs/2409.11096) and references therein for the dark photons. 

The phenomenology is very different from the descriptions in the PBC report. This is because the latter is quite outdated. 

### Fermion-coupled ALP normalization

For `ALP-fermion`, the native SensCalc and EventCalc coupling is

$$
g_Y \equiv y \equiv \frac{2v_h}{f_a}.
$$

If $W_i(m_a)$ denotes the coupling-independent width function imported by
SensCalc, each partial width is evaluated as

$$
\Gamma_i(m_a,g_Y)
=\left(\frac{g_Y}{2v_h}\right)^2 W_i(m_a)
\equiv g_Y^2 f_i(m_a),
\qquad
f_i(m_a)=\frac{W_i(m_a)}{(2v_h)^2}.
$$

It follows that

$$
c\tau(m_a,g_Y)=\frac{\hbar c}{g_Y^2 f_{\mathrm{tot}}(m_a)}
=\frac{c\tau_1(m_a)}{g_Y^2},
$$

where `ctau-ALP-fermion.txt` tabulates $c\tau_1=c\tau(g_Y=1)$ in metres.
Similarly, `Total-yield-ALP-fermion.txt` tabulates the production probability
per proton on target per $g_Y^2$, so EventCalc evaluates
$N_{\mathrm{prod}}=N_{\mathrm{POT}}Y_1g_Y^2$. Its generic
`coupling_squared` output is therefore $g_Y^2$ for this model. Branching
ratios are independent of $g_Y$ because the same $g_Y^2$ multiplies all
partial widths.

The main SensCalc sensitivity notebook defines $y=2v_h/f_a$ in its
phenomenology chapter; this is the authoritative normalization. A later plot
annotation in that notebook reads $y=v_h/f_a$. That annotation is stale and
non-normative and must not be used to reinterpret the tables. See
[`Distributions/ALP-fermion/README.md`](Distributions/ALP-fermion/README.md)
for the column-level data dictionary. In source filenames,
`scale-1000.-GeV` denotes the BC10 EFT/RG matching scale
$\Lambda=1\,\mathrm{TeV}$ at which $c_f=1$; it does not set
$f_a=1\,\mathrm{TeV}$.

### How arbitrary mixing pattern of HNLs is implemented

To make it possible to consider various mixing pattern, `EventCalc` has the HNL production yield, HNL tabulated distribution, HNL decay width, branching ratios of all possible HNL decay modes and their matrix elements (to sample the kinematics of decay products) for the pure three mixings (1,0,0), (0,1,0), (0,0,1). Once the user selects the arbitrary mixing pattern `(Ue2, Umu2, Utau2)`, the code merges the mentioned quantities:
- The total yields `(yield_e, yield_mu, yield_tau)` for pure mixings are summed with weights `(Ue2, Umu2, Utau2)`, giving the total production yield `yield_total`.
- The tabulated distributions are summed with weights `(yield_e*Ue2, yield_mu*Umu2, yield_tau*Utau2)/yield_total`.
- The decay widths `(Gamma_e, Gamma_mu, Gamma_tau)` are summed with weights `(Ue2, Umu2, Utau2)`, giving the total decay width `Gamma_total`.
- The branching ratios and decay matrix elements are summed with weights given by `(Gamma_e*Ue2, Gamma_mu*Umu2, Gamma_tau*Utau2)/Gamma_total`.

Once this is done, we have all the ingredients for calculating the event rate for the given mixing pattern.

## Mother particles fluxes calculations

When producing LLP fluxes, the following setups have been used to get the fluxes of mesons (their mother particles):

- Light-meson fluxes (`Pi0`, `Eta`, `Eta'`, `Rho0`, `Omega`, etc.) use the Pythia 8 setup of [1904.02091](https://arxiv.org/abs/1904.02091). The present flux omits secondary production in the target cascade. These secondaries are expected to be soft and largely outside the SHiP acceptance; their numerical impact remains to be established. Dark-photon production is the exception: its cascade and combined sources described in `README.md` carry the secondary contribution explicitly.
- For Primakoff ALP production: the primary photons from the proton-target interaction are generated by decaying `Pi0`, `Eta`, and `Eta'` mesons into two photons. Secondary photons from electromagnetic cascades inside the thick target follow Ref. [2509.14310](https://arxiv.org/abs/2509.14310). These two photon populations produce photophilic ALPs and the photon component induced by the $SU(2)_L$ interaction.
- For charm and bottom: [SHiP study](https://cds.cern.ch/record/2115534).
- For the charged-kaon contribution to the $SU(2)_L$ ALP flux: the angular--energy distribution from [2004.07974](https://arxiv.org/abs/2004.07974) is normalized independently of the charge-summed flux $0.36=0.29+0.07$ kaons per proton on target.

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

### Optional detector-plane acceptance

The generator-level yield includes the production and the decay-volume
geometry. For exported events, `events_analysis.py` adds a truth-level
selection on the visible decay products. It propagates them along straight
trajectories to a rectangular plane, by default the
$4\,\mathrm{m}\times6\,\mathrm{m}$ plane at $z=95\,\mathrm{m}$, which lies
downstream of the decay volume. A particle counts as visible when its PDG code
is one of $e^\pm$, $\mu^\pm$, $\pi^\pm$, $K^\pm$, $\gamma$ and $K_L$: those
are the particles the SHiP spectrometer and its calorimeter reconstruct. The
user may require exactly two photons, a crossing subset of at least two
particles with zero total charge, or every visible particle to cross the
plane. A common particle-energy threshold is optional; without one the
condition is $E\geq m$. The resulting acceptance is weighted by the parent
decay probability and is a geometric quantity: it contains no reconstruction
or efficiency model.

### How LLP decay phase space is sampled

It is split into three steps:
- `EventCalc` simulates the phase space of decay products at the LLP rest frame. 
  - Two-body decays are sampled isotropically, with the two decay products having the same energy.
  - Three-body decays `LLP -> 1+2+3` are sampled using the squared matrix element of the process in terms of the LLP mass and energies `E_1`, `E_3` of the decay products.
  - Currently, four-body decays `LLP->1+2+3+4` are sampled assuming unit squared matrix element (to be improved in the future).
  - Decays with five or more primary products are sampled from flat phase space with the Raubold--Lynch algorithm. This route is the same whether the hadrons come from Pythia or from exHad; unstable products then enter the usual decay, boost and vertex chain.
- If the selected decay products contain partons or unstable particles, the phase space is passed to `pythia8` for showering, hadronization, or decays such as `pi^0 -> gamma gamma`. Already-stable final states are kept without calling `pythia8`.
- The resulting phase space is boosted into the LLP's lab frame.

### How hadronic decays of LLPs are handled

Masses of LLPs to be probed at SHiP are in the GeV range. Therefore, there is no unified description of their hadronic decays: at masses m_LLP <~ 1 GeV, one has to use exclusive description, whereas at larger masses, one may switch to perturbative QCD. In the domain of intermediate masses, matching between these two descriptions has to be done (see, e.g., [1801.04847](https://arxiv.org/abs/1801.04847) as an example of such a discussion).

The rates are matched as follows:

- The exclusive widths are non-zero below the matching mass and zero above it, while perturbative widths are non-zero above it and zero below.
- For HNLs, neutral current and charged current widths are matched independently from each other, and for each pure mixing, matching is performed separately.
- Consider the perturbative decay `S->c cbar`. The physical hadronic channel opens at the $D\bar D$ threshold, $2m_D$; the quark threshold $2m_c$ lies below it. Some calculations include the perturbative width above $2m_c$, as in [2201.06580](https://arxiv.org/abs/2201.06580). EventCalc applies the hadronic threshold as follows:
  - When calculating the decay width and the branching ratio, the quark mass is replaced with the corresponding lightest meson mass. This way, the decay `S->cc` is replaced with `S->DD`.
  - To sample the kinematics of the decay, the actual quark masses are used. Above the di-D threshold, `S` decays into two `c` quarks, each with mass `m_c = 1.27 GeV`, which are then hadronized.
  - For a three-body decay such as `N -> cc nu`, the invariant mass of the `cc` pair is sampled so that it always lies above the di-D threshold. Sampling the full Dalitz plot instead would discard events below that threshold and suppress the channel twice: once through the replacement `m_c -> m_D` in the width, and once through the threshold. The same applies to the other hadronic decays.
- A decay row whose daughters do not fit inside the parent has rate exactly zero at and below its threshold (`funcs/thresholds.py`). A rate interpolated between tabulated masses can otherwise open such a row slightly, which produces daughters with no phase space. Closed rows are zeroed before the event counts are drawn, and the rate of the selected rows is recomputed from what survives.

The hadrons themselves come either from unmodified `pythia8`, which fragments the selected partonic state, or from the exHad generator, which matches the composition of the final states to exclusive calculations and to measured cross sections. The installed coverage is:

- **Dark photon:** $1.70\leq m_{A'}\leq5.00$ GeV. The selected $u\bar u$, $d\bar d$, $s\bar s$ and $c\bar c$ rows form one pool. Its light-hadron composition follows the DeLiVeR exclusive decomposition from 1.70 to 2.00 GeV and the coherent fit to measured $e^+e^-$ cross sections above 2.00 GeV; the measured open-charm channels and the inclusive charm remainder are generated separately above the $D\bar D$ threshold. The unresolved light component changes continuously to Pythia fragmentation between 4 and 5 GeV.
- **Fermion-coupled ALP:** $1.911\leq m_a\leq5.00$ GeV. The selected hadronic rows, including `ppbar` and `nnbar`, form one pool. Leptonic and two-photon rows are unchanged.
- **Higgs-like scalar:** $2.00\leq m_S\leq63.0$ GeV, of which the SHiP production tables cover $m_S\leq5.12$ GeV, which bounds a run. All selected hadronic rows form one pool. The decay tables use the C-even neutral-kaon rows `KLKL` and `KSKS` with equal rates and carry `ppbar` and `nnbar` as separate rows. The four prescriptions are those listed in `README.md`; their lifetime files keep a `.txt` suffix and are JSON arrays of `[mass_GeV, c_tau_times_sin_theta_squared_m]` pairs.
- **HNL:** $0.02\leq m_N\leq40$ GeV, of which the SHiP production tables cover $m_N\leq5.27$ GeV. The hadrons of a selected current are generated at the exact invariant mass $W$ of the quark-antiquark pair of each event. The six currents are `NC_ud`, `NC_s`, `CC_ud`, `CC_us`, `CC_cd` and `CC_cs`. Every explicit pole and exclusive HNL row stays an EventCalc decay.

EventCalc's branching ratios and lifetimes fix the number of events of every row in both cases. For the ALP and the scalar, the sum of the selected hadronic rows is generated once at the mass of the point, rather than row by row; for the HNL, only the hadrons of the selected current are replaced, so the charged lepton or neutrino, the sampled four-momentum of the pair and $W$ are untouched. `--exhad off` runs the same points on the `pythia8` baseline.

Below 1.70 GeV the dark-photon exclusive rows carry the detailed DeLiVeR decomposition: aggregate entries such as `rho_other`, `omega_other` and `phi_other` are split into physical submodes rather than represented by surrogate daughters. The same input fixes the charge-mode fractions of the $K\bar K$ and $K\bar K\pi$ channels.

For an HNL three-body decay the sampled hadronic mass is restricted to the physical threshold of the selected current before the hadrons are generated, so the threshold suppression is not applied a second time. The partonic phase space uses the current-quark masses of the exported matrix elements, which is the convention those matrix elements were written in.

## To be done

- Adding further LLP models, including ALPs with independent hadronic operators, B-L mediators, HNLs with dipole couplings, and inelastic and elastic light dark matter.
- Adding theoretical uncertainty (decay widths for Higgs-like scalars, B-L production uncertainties, etc.).
- Improving the performance of the code.
- Adjusting the SHiP setup with the up-to-date setup if needed.
- Adding cascade production from kaons and light mesons for HNLs, Higgs-like scalars, dark photons.
- Adding more sophisticated simulation codes (such as the machinery to simulate HNL-anti-HNL oscillations).
- Writing the LLP's own PDG code into the event record, where the identifier column currently holds the placeholder 12345678. The HepMC3 output already carries the real code.
- Add the possibility to sample LLPs solely within the azimuthal acceptance (as it is done in the `EventCalc` module in the [SensCalc repository](https://github.com/maksymovchynnikov/SensCalc)).
- Add the version adapted for HTCondor.
