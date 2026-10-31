# CHARM: implemented geometry and cuts

Companion to [charm.py](charm.py). Defaults implement a phenomenological
off-axis lepton-pair acceptance with individual E > 1 GeV and a nominal
downstream aperture. They do not reproduce the complete historical reconstruction.

## Papers used

| Paper / evidence | Role and limitation |
| --- | --- |
| [Kyselov and Ovchynnikov, Searches for long-lived dark photons at proton accelerator experiments, arXiv:2409.11096v3](https://arxiv.org/abs/2409.11096v3) | Off-axis 3 x 3 x 35 m volume beginning at 480 m, 5 m transverse offset; lepton daughters within calorimeter acceptance with individual E > 1 GeV |
| [Not-so-inelastic Dark Matter, arXiv:2405.08081v2](https://arxiv.org/abs/2405.08081v2) | Appendix C.2: phenomenological CHARM recast; its normalization choices should not be confused with those of the 2024 dark-photon description |
| [CHARM Collaboration, A search for decays of heavy neutrinos in the mass range 0.5–2.8 GeV (1986)](https://doi.org/10.1016/0370-2693(86)91601-1) | Historical dilepton search; full historical numerical selection has not been independently verified here |
| [CHARM Collaboration, Search for axion like particle production in 400 GeV proton-copper interactions (1985)](https://doi.org/10.1016/0370-2693(85)90400-9) | Separate historical visible-decay search; its response is not assumed identical to the HNL dilepton search |

The nominal final-plane position below follows the audited current SensCalc
geometry, not a recovered original-analysis survey. A rectangular full-plane mask
does not implement segmentation, dead areas or the original detector response.

## Default geometry and cuts

Lengths are metres and momenta/energies are GeV (c = 1). The beam axis is x = y = 0,
with the reference production origin at z = 0. Source origins, cuboid and planes
must be expressed in one common coordinate frame. The adopted offset is +x.

| Requirement / setting | Default | Interpretation |
| --- | --- | --- |
| Decay volume | x = 3.5 to 6.5; y = -1.5 to 1.5; z = 480 to 515 m | Cuboid with centre x = 5 m; exact ray/cuboid intersection |
| Detector plane | z = 521.4 m; x = 3.5 to 6.5; y = -1.5 to 1.5 m | Inherited nominal final plane; approximate rectangular aperture |
| Multiple planes | Supported through `planes`, each with `z`, `lower` and `upper` transverse bounds | Every required visible daughter must reach every configured plane |
| Pair topology | Exactly two visible charged daughters, zero total charge | Explicit truth-level proxy; extra visible daughters are rejected |
| Individual energy | Each visible daughter E > 1 GeV | Phenomenological recast cut |
| Daughter direction / transport | Forward reach to every supplied plane, with straight trajectories | No field, scattering, energy loss or displaced-secondary transport |
| Species for the lepton recast | ee, mumu or emu | Hadron pairs remain a separate diagnostic |
| Reconstruction efficiency | 1 by default; explicit `efficiency` in [0,1] | Applied once to selected outputs, not to volume-only output |
| Momentum, pair energy, impact parameter, separation, transverse mass | No additional cuts | Not inferred from other experiments |

Each plane clips the accepted parent flight interval analytically. The default
has one plane. Supplying an empty `planes` list explicitly disables daughter-plane
requirements while keeping the selected-pair kinematic cuts. Nonzero-weight
stationary parent populations are rejected. Invisible daughters remain in the
complete event record, but they are not required to hit a plane.

| Named output | Included requirements |
| --- | --- |
| `decay_in_volume` | Parent vertex in off-axis cuboid only; no daughter or efficiency requirement |
| `selected_pair` | Pair topology, energy threshold and configured plane intersections; diagnostic includes hadron pairs |
| `ee` | Selected pair consists of two electrons/positrons |
| `mumu` | Selected pair consists of two muons |
| `emu` | Selected pair consists of one electron/positron and one muon |
| `selected_leptons` | Disjoint sum of ee, mumu and emu |

All selected outputs use the same explicitly configured efficiency. No historical
channel efficiency is inferred from a daughter species or applied twice.

## Unresolved historical inputs and absent response

| Item | Treatment in this implementation |
| --- | --- |
| Exposure | Required in the scan card. The source audit found 1.7e18 POT in the 2024 description versus 2.4e18 in the niDM description/current preset; neither is a hidden default here |
| Reconstruction efficiency | Explicit input. The audit found a 0.65–0.85 description versus a 0.6 current-code multiplier; neither is installed as a universal response |
| Original analysis masks, detector-plane survey and species-dependent efficiencies | Not established by the nominal preset |
| Trigger, PID, shower separation/merging, veto and resolution | Not simulated |
| Upstream/shower production and material transfer | External source responsibility; geometry selection does not modify production |
| Observed statistics / event threshold | Explicit scan choice; do not mix the separate historical analyses |
| Lifetime probability | Exact exponential integral over accepted proper-flight-distance intervals |

All fields of `CHARMConfig`, including all detector planes, are stored in the scan
metadata. Update this table whenever the companion module changes.
