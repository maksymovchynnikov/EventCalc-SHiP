# SHiP: implemented geometry and cuts

Companion to [ship.py](ship.py). This table describes the default
`ShipResponse`, including the current 0.77 T m field integral. It documents the
implemented acceptance model; it is not a full detector or reconstruction card.
Custom geometry and selection settings supplied by a run override these defaults.

Distances are in metres, energies and momenta in GeV (c = 1). The target-front
centre is (0, 0, 0), with the nominal beam along +z. A **visible daughter** is one
marked visible by the supplied event mask; neutral particles can be visible too.

## Geometry and transport

| Requirement / setting | Default implementation | Applies to |
| --- | --- | --- |
| Decay volume, longitudinal | z = 32 to 82 m | All selections |
| Decay volume, horizontal | Full width increases linearly from 1 m at entrance to 4 m at exit; centred on x = 0 | All selections |
| Decay volume, vertical | Full height increases linearly from 2.7 m at entrance to 6.2 m at exit; centred on y = 0 | All selections |
| Parent propagation | Forward flight from the supplied production origin; exact ray/volume intersection | All selections |
| Daughter detector plane | z = 97.32 m | All daughter selections |
| Detector aperture | Rectangle centred on the nominal axis: full width 4.32 m, full height 6.48 m (half-widths 2.16 and 3.24 m) | Each counted daughter |
| Dipole region | z = 86 to 91 m; uniform transverse field with integral B_y dz = 0.77 T m | Daughter transport |
| Charge-dependent transport | Shared exact uniform-dipole displacement; neutral daughters have no magnetic deflection | Daughter transport |
| Transport direction | Daughter p_z > 0 and successful forward transport to the detector plane | Each counted daughter |

## Named output selections

Every row also requires the parent decay vertex to lie inside the decay volume.
These are separate outputs, not a single cumulative cutflow.

| Output selection | Daughter multiplicity at detector aperture | Per-daughter energy | Pointing / impact parameter |
| --- | --- | --- | --- |
| `decay_in_volume` | No daughter requirement | None | None |
| `at_least_two_energy` | At least two accepted visible daughters | E > 1 GeV for each counted daughter | None |
| `at_least_two_energy_ip` | At least two accepted visible daughters | E > 1 GeV for each counted daughter | IP <= 2.5 m |
| `all_visible_energy` | At least two visible daughters; every visible daughter must be accepted | E > 1 GeV for every visible daughter | None |
| `all_visible_energy_ip` | At least two visible daughters; every visible daughter must be accepted | E > 1 GeV for every visible daughter | IP <= 2.5 m |

IP is the transverse distance at the target plane z = 0 of the line through the
decay vertex along the **sum of momenta of the accepted visible daughters**.
It is not computed from the parent momentum or from rejected daughters. The
summed momentum must have positive p_z for the IP selections. Energy thresholds
are strict; aperture boundaries have no effect on a continuous integral.

## Present and absent requirements

| Feature | Status in this response |
| --- | --- |
| Minimum accepted multiplicity, energy, geometry and optional pointing | Implemented as above |
| Exactly two tracks, opposite charge, particle identification | Not imposed; the selection counts visible daughters, not only charged tracks |
| Separate track momentum threshold | Not imposed; the default threshold is on energy |
| Track/cluster separation, central detector holes, multi-plane tracking masks | Not implemented |
| Trigger, tracking/PID/veto/reconstruction efficiencies and resolution | Not implemented |
| Material interactions, scattering and energy loss | Not implemented |
| Mass window or background-dependent statistical selection | Not implemented by the experiment response |
| Decay probability | Integrated analytically over accepted proper-flight-distance intervals by the shared engine |
| Production normalization, branching fraction and exposure | Supplied outside the experiment cuts; applied through event weights and scan configuration |

`Geometry.ship()` obtains the decay-volume dimensions from the legacy
`funcs.ship_setup` settings. The other geometry defaults and the five selections
are defined in the companion module. Update this table whenever those defaults
or their interpretation change.
