# NA62: implemented geometry and cuts

Companion to [na62.py](na62.py). The response combines frozen analytic ee/mumu
cuts with the hadronic **ideal-response projection at charged p > 1 GeV**.
It is a truth-level acceptance model, not a full reconstructed analysis or an
observed exclusion. The 1 GeV hadron threshold is a projection assumption.

Distances are in metres, energies and momenta in GeV (c = 1). Coordinates use
the nominal NA62 target at z = 0. The reference production point is
(x, y, z) = (0, -0.022, 23.07) m; external production can supply its own origins
in the same frame. Pointing uses a beam parallel to +z through
(x, y) = (0, -0.022) m. The radial fiducial and aperture cuts are centred on
the nominal detector axis (x, y) = (0, 0), not on that offset beam line.

## Geometry and transport

| Setting | Implemented value / condition | Status |
| --- | --- | --- |
| Parent direction | p_z > 0 | Required by both acceptance preparations |
| Base decay cylinder | Radius r < 1 m | All lepton volume outputs and hadronic stages |
| Muon and hadronic longitudinal volume | z = 105 to 180 m | Frozen baseline |
| Electron longitudinal volume | z = 104.987277... to 180 m | Digitized FV endpoints |
| Restricted electron/hadron FV | r below the piecewise linear contour in the table below | Raster approximation, not an exact experimental card |
| Dipole | z = 196.35 to 197.65 m; B_y = 0.6928 T, integral = 0.90064 T m | Uniform-field transport approximation |
| Final charged-track aperture plane | z = 243.85 m | Inherited aperture proxy |
| STRAW separation plane | z = 183 m | Hadronic selection |
| LKr plane | z = 241.495 m | Hadronic separation and photon reconstruction; approximate nominal alignment |
| Track transport | Forward daughter p_z > 0 and valid transport; charged tracks use dipole displacement, neutral particles follow straight lines | No material effects |

The digitized contour is linearly interpolated between each pair of endpoints:

| z range (m; rounded here) | Maximum radius at left endpoint (m) | Maximum radius at right endpoint (m) |
| --- | --- | --- |
| 104.987277 to 156.910214 | 0.317440 | 0.519913 |
| 156.910214 to 180 | 0.650837 | 0.763495 |

There is a radius step at z = 156.910214... m. The accepted side is strictly
below the contour and between its outer z endpoints. Full-precision coordinates,
digitization provenance and estimated raster tolerances are in
[the FV data](data/na62_ee_fv.json); see also
[the checked overlay](data/na62_ee_fv_overlay.png).

## Lepton selections

Here z_CA is the z coordinate of closest approach of the visible-pair line to
the offset beam line, and CDA is the distance between those lines. The pair
momentum is evaluated at the vertex, before the magnetic kick. Exactly parallel
pair/beam lines have no unique z_CA and fail the pointing requirement.

| Requirement | Electron pair (ee) | Muon pair (mumu) |
| --- | --- | --- |
| Visible topology | Exactly two visible, opposite-sign electrons | Exactly two visible, opposite-sign muons |
| Per-track momentum | p > 10 GeV | p > 10 GeV |
| Pair energy | Positive; no additional numerical pair-energy threshold | Positive; no additional numerical pair-energy threshold |
| Base vertex volume | r < 1 m; electron z range above | r < 1 m; 105 to 180 m |
| Restricted FV contour | Applied | Not applied |
| Pointing signal region | ((z_CA - 23 m) / 12 m)^2 + (CDA / 0.03 m)^2 < 1 | 6 m < z_CA < 40 m and CDA < 0.020 m |
| Detector aperture | Each track: 0.08 m < r < 1.15 m at z = 243.85 m | Same |
| STRAW/LKr pair separation | Not imposed | Not imposed |

| Named output | Requirements included |
| --- | --- |
| `volume_ee`, `volume_mumu` | Parent direction and corresponding base cylinder/z range only; **no daughter-species or multiplicity filter**, despite the output labels |
| `selected_ee`, `selected_mumu` | Corresponding volume plus lepton topology, momentum, pointing and, for ee, restricted FV |
| `detector_ee`, `detector_mumu` | Corresponding selected output plus transported-track aperture |

The analytic lepton preparation supports fully visible DP pairs. If a selected
pair has more than 1e-7 GeV of invisible daughter energy, it raises an error
because the required general pointing-boundary roots are not implemented.
Invisible daughters remain in the complete-event conservation check.

## Hadronic cuts, in cumulative order

The response accepts exactly two opposite-charge terminal pion/kaon tracks,
with optional photons and supported invisible daughters. Other terminal visible
species are rejected. In the fixed PDG filtering, neutrinos (absolute IDs
12, 14, 16, 18), IDs 4900101/4900102, and padding ID 0 are ignored for topology.

| Stage | Requirement | Value / interpretation |
| --- | --- | --- |
| `cylinder_volume` | Decay vertex in the base cylinder | r < 1 m; z = 105 to 180 m |
| `restricted_fv` | Vertex inside digitized FV | Same radial contour as ee |
| `two_opposite_hadron_tracks` | Exactly two charged terminal daughters with opposite charges | Supported visible species: charged pions, charged kaons and photons; no additional visible species |
| `track_p_threshold` | Momentum of each charged track | p > 1 GeV in `NA62Response`; forward transport also required |
| `track_aperture_proxy` | Each charged track at final aperture plane | 0.08 m < r < 1.15 m at z = 243.85 m |
| `separation_straw` | Charged-track separation at STRAW plane | At least 0.020 m at z = 183 m |
| `separation_lkr` | Charged-track separation at LKr plane | At least 0.200 m at z = 241.495 m |
| `photons_reconstructed` | Identify photons to add to reconstructed momentum | E_gamma > 5 GeV, forward/valid transport, and 0.08 m < r < 1.15 m at LKr; **missing photons do not veto the event**, so this stage adds no rejection |
| `hadronic_pointing` | Pointing of the two charged tracks plus reconstructed photons | ((z_CA - 23 m) / 23 m)^2 + (CDA / 0.04 m)^2 < 1 |

The final `hadronic_group_0` through `hadronic_group_5` outputs all use the
complete sequence above, separated by terminal-state diagnostics:

| Group | Diagnostic label | Effect of the supported-topology restriction |
| --- | --- | --- |
| 0 | Charged pions only | Can pass |
| 1 | Pions and photons | Can pass |
| 2 | Contains charged kaons | Can pass if every other visible species is supported |
| 3 | Contains terminal neutral kaons | Rejected by the current supported-topology requirement |
| 4 | Contains terminal baryons | Rejected by the current supported-topology requirement |
| 5 | Other | Only supported terminal combinations can pass; this is not an inclusive acceptance for arbitrary decays |

Groups are assigned in the listed priority order, so an event containing charged
kaons is classified as group 2 before neutral-kaon or baryon checks. Classification
does not override the topology rejection.

## Numerical settings and absent response

| Feature | Implementation / status |
| --- | --- |
| Lepton vertex integration | Analytic accepted intervals |
| Hadronic vertex integration | 128 midpoint-tested z cells by default; adjacent accepted cells merged; exponential probability integrated analytically over those cells |
| Numerical refinement | `NA62Config.vertex_cells` can be increased, e.g. to 256 |
| Threshold alternatives | The public frozen configuration requires hadron p > 1 GeV and lepton p > 10 GeV; other values are rejected. The lower-level diagnostic cutflow can evaluate explicit alternative hadron thresholds |
| Tracking, PID, trigger, veto efficiencies | Not implemented; ideal response |
| Full detector masks, photon association/merging, mass-window reconstruction | Not implemented |
| Vertex/momentum resolution, residual FV field, scattering, energy loss | Not implemented |
| Transverse-mass cut | Helper exists, but no such cut is enabled in these selections |
| Exposure, production and branching normalization, statistical event threshold | Supplied outside experiment cuts through event weights and scan configuration |

Update this table together with any change to the companion module's cuts,
defaults or interpretation. A saved run's resolved configuration remains the
authority for that run.
