# Perfect-reconstruction sensitivities

`eventcalc.sensitivity` folds complete LLP decays with the SHiP decay volume and ECAL
aperture. It computes acceptance with magnetic transport once and integrates the
exponential decay law over the allowed flight intervals for every lifetime.
For fixed sampled decays, the direct API evaluates the exponential integral
analytically; the command-line scan uses a interpolated lifetime
response by default (8192 nodes). Increase this resolution for a tighter
interpolation tolerance and check it against exact endpoint evaluation.
Set `response_nodes` to `null` for exact endpoint evaluation. There is no decay-vertex Monte Carlo and no event regeneration per
coupling value. Reconstruction efficiency is one. Detector response and backgrounds require
separate calculations.

Run the **toy** example from the repository root:

```sh
python -m eventcalc.scan_sensitivity cards/sensitivity_example.json --output /tmp/toy-reach
python -m unittest discover -s tests -p test_sensitivity.py -v
```

Each job handles one mass and an entire parameter grid. Split a mass scan into
cards and run the same command locally or under HTCondor. Outputs are a compact
`events.npz` grid and `metadata.json`, with all connected threshold intervals,
cuts, geometry, timing and checksums. The command refuses to overwrite an
existing output directory. It does not save event records.

## Physics inputs and normalization

The integration API is independent of the model, its coupling convention and
its production generator. Supply `DecayBatch` objects containing:

| Field | Convention |
|---|---|
| `mother` | `[N,4]` lab four-momenta `(px,py,pz,E)` in GeV |
| `daughters` | `[N,K,4]` complete final state, including invisible particles; zero padding |
| `visible` | Explicit boolean `[N,K]` mask of particles counted by the selection |
| `weights` | One integral contribution per event, per incident particle, including production and selected decay branching fractions |
| `mass` | Mass in GeV of the particle whose decay is propagated |
| `origins` | `[N,3]` production positions in metres; zero by default |
| `frame` | `lab` or `rest` for the daughters |
| `charges` | `[N,K]` charges in units of the positron charge; required when the field is nonzero |

For N unweighted generated decays, a usual event weight is
`production_per_incident * selected_branching_fraction / N`. The denominator is the total sample size across all chunks. Importance weights must include
their sampling density. Signed weights are supported. Include invisible
daughters in four-momentum conservation even though their visibility is false.
The input must describe the full production spectrum, without lifetime-dependent
survival or fiducial-decay bias. Missing support cannot be recovered by reweighting.
Stationary mothers are rejected; they require a source/time treatment.

```python
from eventcalc.sensitivity import integrate, scan_replicas

events = integrate(batches, ctau_m,
                   exposure=6e20, production_scales=production_scales)
mean, integration_error = scan_replicas(
    lambda replica: make_independent_batches(replica), ctau_m,
    replicas=4, exposure=6e20, production_scales=production_scales)
```

Use batches of about 4096 events to bound memory. In callback-based generators,
`prepare(batch).evaluate(ctau_m)` returns the contribution of one batch; sum
those contributions before applying exposure and production scaling.

`ctau_m` and `production_scales` are separate arrays. No power of epsilon, mass
ratio, branching fraction or width law is inferred. For a model whose yield
is proportional to epsilon squared and width is proportional to epsilon
squared, one can supply `production_scales=epsilon2` and
`ctau_m=ctau_at_unit_epsilon2/epsilon2`. This shortcut is valid only when the
normalized production and decay distributions stay fixed throughout the scan.
Regenerate inputs when couplings change branching fractions, mediator widths,
polarization or spectral shapes. Evaluate each physical variation consistently through production, decay and
acceptance before comparing its reach with the central prediction.

Primary and cascade batches use the same API; concatenate their correctly
normalized streams to obtain the total. For inelastic DM the propagated mother
is the excited state, with its mass, momentum, production position and lifetime.
Generate the polarized mediator decay in the source provider and include the
stable dark particle among the invisible daughters of the excited-state decay.
The acceptance integrator does not change the input angular correlations.

## Selections

The default outputs, in order, are:

1. Decay inside the fiducial volume.
2. At least two visible daughters reaching ECAL, each with E > 1 GeV.
3. Selection 2 with transverse IP <= 2.5 m at the target plane.
4. All visible daughters reaching ECAL with E > 1 GeV, with at least two.
5. Selection 4 with IP <= 2.5 m.

IP uses the summed momentum of all daughters satisfying the aperture and energy
requirements. A missing particle can therefore affect IP. `Selection` permits
other multiplicity, energy and IP requirements. `Geometry.ship()` reads the
decay-volume dimensions from `funcs/ship_setup.py`; the downstream aperture is
at z=97.32 m with full widths 4.32 by 6.48 m. Geometry is explicit in the output
and can be overridden. All z coordinates are measured from the target front face.
Charged particles traverse a uniform dipole from z=86 to 91 m with integral
B_y dz = 0.5 T m, following the SensCalc ECN3 setup. The implementation solves
the uniform-field trajectory, including the subsequent drift to ECAL; a track
that turns before exiting the magnet is rejected. Neutral particles travel
straight. Set `field_integral=0` explicitly for a zero-field comparison.
IP uses the momentum at the decay vertex, before magnetic bending. Material
interactions and reconstruction losses are not included.

## Sources for the command-line interface

Each card gives ordered positive `parameters`, matching `ctau_m` and
`production_scales`, `exposure`, and optional `replicas` (default 2),
`event_threshold` (default 2.3, the conventional background-free 90% exclusion threshold), `geometry` and `selections`.

Three source types are available:

- `provider`: `callable` is an importable `module:function`; optional `kwargs`
  are passed along with `replica`. It returns a stream of `DecayBatch` objects.
  This supports any production/decay model, polarized sources and target cascades.
- `runtime`: `card` names an existing EventCalc simulation card, relative to the
  sensitivity card. Set `mass_GeV` at the top level. Optional `events`, `seed`,
  `chunk_size` and `invisible_pdgs` configure the source. The adapter shares
  `RuntimeModelScan`, samples its full tabulated energy range, and uses its
  selected decay channels. Default invisible PDGs are 12, 14, 16 and 4900101 (stable chi); add any other
  stable dark particles explicitly. `llp.Yield` is the default production normalization
  before `production_scales`; `yield_per_incident` overrides it. This adapter
  uses compact tables and a point source for calibrated dark photons and
  inelastic DM by default. Select `production_backend: correlated` in the model
  card to preserve collision positions and signed weights from the EOS dataset. For inelastic DM, `mass_GeV` is the
  mediator mass; the adapter uses the actual chi-prime mass in transport. There, `events` selects integration points per production
  contribution, rounded up to a power of two. Other installed models use the
  existing unweighted, point-source runtime. Use providers for their displaced
  production or weighted hadronization.
- `npz`: `replicas` is a list of independently sampled files, each containing
  the `DecayBatch` fields above. `origins` and `frame` are optional. Each array
  is decompressed once, then processed in chunks. Use providers for streaming
  datasets too large to keep one replica in memory.

The standard error is estimated across independently generated replicas of the
whole normalized integral. It excludes uncertainty in a reused source pool and
in the production model. Two replicas give only a rough error diagnostic.
`threshold_intervals` retains disconnected regions; `null` at an endpoint marks
an open grid boundary. It never extrapolates a closed contour beyond the scan.

## Validation and performance

Tests cover the analytic exponential probability, tiny and large lifetimes,
independent dense vertex quadrature, displaced and backward production,
three-body decays with invisible particles, energy/IP cuts, lab/rest boosts,
streaming normalization, replica errors and disconnected contours.

With warm compiled kernels, 100,000 complete decays, both ECAL selections and
241 lifetimes took about 0.21 s (two particles), 0.25 s (four), and 0.32 to 0.41 s
(eight) on the development machine. These timings include array validation,
magnetic transport, interval construction and the cached lifetime response.
They exclude source generation and first-call compilation. Independent
Lorentz-equation integration agrees with magnetic transport within 5.3e-12 m;
dense vertex quadrature agrees within 6e-7 relative. The two interval
implementations agree for displaced two-, four- and eight-particle decays.


For many correlated production variations, retain the event identity and build
one reusable lifetime response:

```python
prepared = prepare(batch, geometry=geometry, selections=selections)
response = prepared.response(event_reweights, nodes=8192)
# event_reweights: [input event, variation]; output: [variation, lifetime, selection]
yields = response.evaluate(proper_decay_lengths)
```

No model, coupling, PDF family, branching-fraction law, or uncertainty-combination
rule is assumed. Apply the same variation consistently to all batches before
forming an envelope. The response linearly interpolates endpoint exponentials
in log proper distance; compare its nominal column with
`prepared.evaluate(proper_decay_lengths)` and increase `nodes` if needed.
Acceptance is computed once. Subsequent evaluations contract the cached
response with lifetime weights. Empty selections and origins inside the decay
volume are supported. This interpolation error is distinct from finite event
statistics and physics uncertainties.

## Dark photon and inelastic dark matter sensitivity

The [production guide](PRODUCTION_DATA.md) defines the calibrated target model,
compact inputs and full dataset. One-mass scans use the full calculation with
correlated positions. Their conservative baseline keeps the small, nonzero
bremsstrahlung contribution fitted within the single-scale virtuality model.

Run a complete DP or inelastic-DM mass point without HepMC:

```sh
python -m eventcalc.model_sensitivity --model dp --mass 1.1 --exhad-root /path/to/exHad --output /tmp/dp-m1p1
python -m eventcalc.model_sensitivity --model idm --mass 20 --delta 0.1 --exhad-root /path/to/exHad --output /tmp/idm-m20
```

The inelastic parameters follow the [model definition](INELASTIC_DM.md). The generic
`DecayBatch` interface supports other parameters and models. Each command scans
the full epsilon-squared grid for the inclusive energy/IP selection and the
fully contained energy selection and reports
the conventional background-free threshold of 2.3 events. **Production uncertainty
bands are automatic for both models**, including both epsilon-squared boundaries.
Use `--central-only` only when a scan without production variations is intended.

The default varies three timelike proton form-factor pairs at the fixed fitted
Foroughi-Abari virtuality scale, all CT18/MSHT20/NNPDF members and seven scale
pairs per family, and the Drell-Yan continuum endpoint between 1.75 and 2.5 GeV.
The same choice applies to every primary and cascade collision and throughout
the inelastic virtual-mass integral. For inelastic DM, form-factor variations
also update the mediator polarization weight. Decay kinematics and accepted
flight intervals are shared across variations, so the bands include changes to
the accepted energy and angular spectra on both lifetime branches.

The calculation first sums accepted yields for each variation. It then forms
the PDF intervals: asymmetric Hessian errors for CT18 (converted from 90% to
68%) and MSHT20, and 16th/84th replica percentiles for NNPDF. It envelopes these
with the scale, family, form-factor and transition choices. The result spans these specified production choices; the individual PDF error
conventions apply within each family.
Alternative virtuality models, heavier fragmentation excitations, nuclear and
shower modeling, finite production-sample statistics, and reconstruction remain outside it.

`events.npz` contains:

- `events`, `lower_events`, `upper_events`: `[epsilon2, selection]` central and
  production-envelope yields, after summing primary and cascade production;
- `fixed_events`, `brem_variation_events`, `dy_variation_events`: the unvaried
  contribution, three form-factor choices, and `[transition, PDF/scale, epsilon2,
  selection]` Drell-Yan contributions, respectively;
- `integration_error_events` and component variation integration errors:
  numerical errors across independent replicas, separate from production bands.

`metadata.json` records the labels and input checksums. `boundary_bands` gives
central, lower-yield and upper-yield threshold intervals for each selection.
For matched components it gives `lower_epsilon2_boundary_range` and
`upper_epsilon2_boundary_range`. `variation_intervals` records both crossings
for every coherent form-factor/PDF/scale/transition combination. Disconnected
components stay separate. Null interval endpoints indicate open scan boundaries;
null boundary ranges indicate missing reach or ambiguous component matching.
The code never closes a contour by extrapolating outside the scan.

The one-mass uncertainty command uses the full production dataset on EOS.
Install it once with the command in [PRODUCTION_DATA.md](PRODUCTION_DATA.md).
PDF ratios and exact hard-event identities reside in its `uncertainties/`
subdirectory; no LHAPDF installation or manuscript directory is needed at runtime. Memory-mapped weight chunks are below 32 MB
each. The loader checks their hashes and their source-pool identities. Missing,
corrupt or mismatched data raise an error; there is no silent central-only
fallback. `--pdf-weights` selects a separately packaged, matching dataset.
Changing the coupling grid reuses the production distributions and response;
it never regenerates the hadronic shower.

The nominal response is checked against the exact decay integral near both
boundaries. A relative difference above 0.3% stops the scan; increase
`--response-nodes` if this check fails. Above the continuum transition, identical
endpoint variations are evaluated once. `--power`, `--decay-count`, `--replicas`
and `--seed` control the numerical integration independently of the physics band.

```sh
python -m unittest discover -s tests -p test_production_bands.py -v
python tests/validate_production_variations.py
```

Compact production tables contain the central prediction. A generic scan using
those tables must receive additional physical variations to produce uncertainty
bands. The complete one-mass commands above perform this propagation automatically.
