# BEBC: implemented geometry and cuts

Companion to [bebc.py](bebc.py). The default is a phenomenological lepton-pair
acceptance, using E > 1 GeV for each visible daughter. It is not the complete
historical HNL analysis. No production model, exposure, statistical event threshold
or measured reconstruction efficiency is installed implicitly.

## Papers used

| Paper | Role |
| --- | --- |
| [Kyselov and Ovchynnikov, Searches for long-lived dark photons at proton accelerator experiments, arXiv:2409.11096v3](https://arxiv.org/abs/2409.11096v3) | BEBC phenomenological lepton selection: individual daughter E > 1 GeV; decay chamber itself is the detector |
| [Barouki, Marocco and Sarkar, Blast from the past II, arXiv:2208.00416v4](https://arxiv.org/abs/2208.00416v4) | Section 3 and Table 1: cuboid approximation and dump-end distance; separate pair-energy, scanning/PID and transverse-mass conditions of the HNL reanalysis |
| [Cooper-Sarkar et al., Search for heavy neutrino decays in the BEBC beam dump experiment (1985)](https://doi.org/10.1016/0370-2693(85)91493-5) | Original historical analysis; full original selection has not been independently verified here |

## Default geometry and selections

Lengths are metres and momenta/energies are GeV (c = 1). The geometry frame has
the **dump end at z = 0**, beam along +z, and chamber centred on x = y = 0.
External origins must use this frame; target-front and dump-end coordinates must
not be interchanged without translating the source and volume together.

| Requirement | Default | Interpretation / source |
| --- | --- | --- |
| Decay volume | x = -1.785 to 1.785; y = -1.26 to 1.26; z = 404 to 405.85 m | 3.57 x 2.52 x 1.85 m cuboid from the reanalysis, Table 1; transverse sizes are more precise than the rounded 2024 recast description |
| Parent trajectory | Forward flight from its supplied origin, exact ray/cuboid intersection | General three-dimensional geometry; stationary nonzero-weight populations rejected |
| Pair topology | Exactly two visible charged daughters, zero total charge | Truth-level two-track proxy; additional visible neutral daughters also cause rejection |
| Individual energy | Each visible daughter E > 1 GeV | Default phenomenological selection, arXiv:2409.11096v3 |
| Individual momentum | p > 0 by default; configurable through `track_p_min_GeV` | No positive scanning/PID momentum threshold silently inferred from the energy cut |
| Pair energy | E_pair > 0 by default; configurable through `pair_energy_min_GeV` | Distinct from the per-daughter threshold |
| Transverse mass | Disabled by default | Optional `transverse_mass_max_GeV` requires an explicit nonzero `transverse_axis` |
| Downstream detector planes / ECAL | No requirement | The decay volume is itself the detector |
| Daughter forward direction | No requirement | Backward daughters are retained; no implicit downstream-plane projection |
| Reconstruction efficiency | 1 by default; explicit `efficiency` in [0,1] | Applied once to selected outputs, never to volume-only output |

The optional transverse mass is sqrt(M_pair^2 + pT_pair^2) + pT_pair, using the
configured reference axis. The reanalysis's Eq. (13) bounds this by m_D - m_mu.
Its Table 1 also uses E_pair > 1 GeV and an assumed efficiency of 0.96. These are
separate HNL-reanalysis inputs, not an automatic DP/iDM preset. The scanning and
lepton-identification momentum qualifications in Section 3 are likewise not
represented by a universal constant efficiency or by the default energy threshold.

| Named output | Included requirements |
| --- | --- |
| `decay_in_volume` | Parent decay inside chamber only; no daughter selection or efficiency factor |
| `selected_pair` | Pair topology and configured energy/momentum/transverse-mass cuts; diagnostic includes hadron pairs |
| `ee` | Selected pair consists of two electrons/positrons |
| `mumu` | Selected pair consists of two muons |
| `emu` | Selected pair consists of one electron/positron and one muon |
| `selected_leptons` | Disjoint sum of ee, mumu and emu; use this for the implemented lepton recast |

`selected_pair` is not a claim that all charged hadronic channels were searched
with the leptonic statistics. Invisible daughters remain in the complete decay
record and conservation check but do not enter the visible pair.

## Missing response and normalization choices

| Feature | Status |
| --- | --- |
| Scanning/PID response, neutral-hadron veto and secondary decays/interactions | Not simulated; exact two-visible-particle topology is only a proxy |
| Chamber magnetic tracking, minimum measured track length, material losses and resolution | Not simulated |
| Exposure and target/material correction | Explicit source/scan inputs; no SHiP shower is converted automatically |
| Statistics | Explicit scan threshold; no universal observed-limit construction. Historical leptonic and muon-pion channels have different candidate/background information |
| Lifetime probability | Exact exponential integral over accepted proper-flight-distance intervals |

All fields of `BEBCConfig` are recorded in scan metadata. Update this table with
changes to the companion module; a saved run's resolved configuration determines
the cuts and assumptions used in that run.
