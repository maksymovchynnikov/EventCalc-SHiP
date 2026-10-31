# Sensitivity from external production

Production, conditional decays, experiment response, and lifetime integration are
separate stages. Replacing the production model requires a new input manifest,
not edits to acceptance code. Existing SHiP scripts and scan cards retain their
defaults. BEBC and CHARM responses can be added using the same interface.

## Run the examples

From the repository root, using the Python environment described in the user guide:

```sh
python -m examples.external_inputs --output outputs/external-input-example
```

This creates deterministic inputs and runs four small scans: spectrum tables and
sampled parents, each with SHiP and NA62. The examples use the existing dark-photon
electron-pair decay generator with exHad disabled; they need the ordinary numerical
dependencies but no hadronization installation. They are software demonstrations,
not experimental projections. Both input forms represent exactly the same parent
draws at each replica, so their numerical results should agree.

To create inputs only, add `--inputs-only`. Run an individual generated card with:

```sh
python -m eventcalc.scan_sensitivity outputs/external-input-example/na62_spectrum_scan.json --output outputs/my-na62-scan
```

The output directory must not exist. Generated NPZ files contain expected events
and replica integration errors; JSON records the resolved card, input/code hashes,
replica seeds, selections, and connected threshold intervals.

## External production contract

Each JSON manifest references one NPZ file, relative to the manifest. One manifest
supplies one exact propagated parent mass; no mass interpolation or extrapolation
is performed. Required common fields are:

```json
{
  "schema_version": 1,
  "kind": "spectrum",
  "mass_GeV": 0.3,
  "arrays": "spectrum.npz",
  "frame": "lab",
  "momentum_order": ["px", "py", "pz", "E"],
  "energy_unit": "GeV",
  "length_unit": "m",
  "angle_unit": "rad",
  "coordinate_frame": "nominal NA62 target at z=0; beam along +z",
  "normalization": "produced_parents_per_incident",
  "origin_m": [0, -0.022, 23.07],
  "production_probability": 1e-7
}
```

For `kind: spectrum`, NPZ contains `theta_edges[T+1]`, `energy_edges[E+1]`,
and `bin_probabilities[T,E]`. Probabilities are nonnegative bin integrals summing
to one. Sampling is uniform in theta and energy within each bin, with independent
uniform azimuth about +z and the explicitly supplied point origin. Energy support
must start at or above the parent mass. Each of N draws has weight P/N, where P
is `production_probability`. Production spectra with azimuthal asymmetry or
position/momentum correlations should use parent samples.

`density_to_probabilities` explicitly converts a histogram of dP/(dtheta dE) by
bin area. Other measures, such as dP/dOmega, require the provider to perform their
own Jacobian conversion. The loader never infers a density measure.

For `kind: parents`, NPZ contains `momenta[N,4]`, `origins[N,3]`, and
`weights[N]`; omit `origin_m` and `production_probability` from the manifest.
Weights are already integral contributions per incident particle for the entire
file. They may be signed. Splitting a file into chunks must not renormalize it.
Momenta must be finite, future on-shell four-vectors with the declared mass;
stationary populations are rejected because the flight-distance engine does not
represent them. Zero-weight and empty parent batches are allowed.

Coordinates must already match the experiment frame. No automatic target shift,
material correction, beam smearing, or shower reweighting is applied. Put such
corrections into the supplied production probability or sample weights and record
their assumptions in the manifest. Extra manifest fields are retained in output
metadata. These inputs contain no spin density matrix: the runtime adapter retains
the angular and polarization assumptions of the selected existing decay model.

## Scan and decay configuration

Use `experiment: {"name": "ship"}` or `experiment: {"name": "na62"}`. SHiP accepts
its existing `geometry` and `selections`, either inside the experiment object or
as legacy top-level fields. NA62 uses its named frozen response; set
`experiment.config.vertex_cells` to change its hadronic vertex grid (default 128).
The exposed momentum settings are fixed at hadron p>1 GeV and lepton p>10 GeV;
other values are rejected to avoid silently inventing a different baseline.

External source configuration is:

```json
{
  "kind": "external_spectrum",
  "manifest": "spectrum.json",
  "decay_card": "decay.json",
  "events": 4096,
  "chunk_size": 512,
  "seed": 17
}
```

`decay_card` is an ordinary model launch card, used for decay model/channel
selection. Production-file checks are bypassed for this path; the decay loader
reads and validates the existing decay resources directly. For sampled parents,
use `kind: external_parents` and `replicas: ["parents0.json", "parents1.json"]`.
Each parent replica must have distinct, independently generated arrays. A distinct
hash cannot prove independence; independence remains the provider's responsibility.
Spectrum replicas can share one manifest and use independent seeds, or specify
one manifest per replica. Replicas are never made by relabeling chunks of a pool.

`mass_GeV` in the scan is the propagated parent mass. For the existing inelastic-DM
model, `source.decay_mass_GeV` supplies the mediator mass used by that model; the
adapter checks that its resulting chi-prime mass agrees with the external parent.
The invisible PDG default includes neutrinos and 4900101; override
`source.invisible_pdgs` when needed. Complete invisible daughters remain in the
conservation check. Existing `runtime`, Python `provider`, and complete-decay
`npz` sources remain available. Complete-decay NPZ inputs now also accept
`pdg_ids[N,K]`, required for NA62, in addition to charges and the visibility mask.

For parameter point a, the expected yield is

`exposure * production_scale[a] * sum_i(production_weight[i] * decay_weight[i] * accepted_lifetime_probability[i, a])`.

The runtime decay weight is the selected branching fraction applied once to a
conditional decay sample. Neither source adapter includes branching fractions,
lifetime probabilities, or exposure. Complete-decay providers instead supply
`DecayBatch.weights` with production and branching/conditional-decay weights
already included, as before. Additional weighted hadronization must be normalized
by its established full-sample contract before supplying complete-decay batches.

`parameters`, `ctau_m`, and `production_scales` are explicit matching scan arrays.
No coupling law is inferred. They may rescale rates and lifetimes while holding
normalized production and conditional decay distributions fixed. Regenerate inputs
when shapes or branching fractions change. A yield threshold is a separately
chosen statistical prescription; the scan preserves disconnected threshold
intervals rather than joining unrelated branches.

Acceptance is the probability of passing geometry/selection conditional on the
chosen generated particles and lifetime. Expected yield additionally includes
production, branching fractions, and exposure. Sensitivity additionally requires
a specified statistical threshold. Signed integration contributions are preserved
throughout and are not themselves probabilities.

## Experiment response and validation

An `ExperimentResponse` exposes `selections`, opaque `geometry` metadata, and
`prepare(batch) -> PreparedAcceptance`. Each accepted interval is expressed in
proper flight distance in metres, with a weight, selection index, and input event
index. Several disjoint intervals can belong to one event. Experiments own
geometry, cuts, and transport; the shared engine evaluates the exponential decay
probability on their intervals, exactly or with its optional response interpolation.

SHiP's existing geometry and defaults live in `eventcalc.experiments.ship`.
NA62 geometry, cuts, transport helpers, cutflow, and response preparation live in
`eventcalc.experiments.na62`. Its frozen lepton cuts use analytic intervals. The
hadronic selection uses midpoint decisions on vertex cells, merges consecutive accepted cells, and then
integrates the exponential exactly over those cells. The exposed cutflow retains
the original cumulative stages and terminal-state groups. Digitized electron-FV
vertices, source metadata, and a checked overlay are bundled with the package.

NA62 is a truth-level response with ideal tracking/PID, approximate apertures,
and no full trigger, veto, clustering, resolution, or mass-window response.
Missing photons do not automatically veto events. The analytic lepton response
supports the frozen fully visible DP-pair topology and explicitly rejects selected
pairs with invisible energy; it is not a general iDM lepton response. Unsupported
hadronic topologies retain their original rejection. The hadronic selection is an
ideal-response projection, not a validated detector simulation or observed limit.

Portable regression fixtures cover the original SHiP response, six NA62 mass
points, and doubled NA62 vertex grids. The focused tests also cover analytic
lifetime probabilities, disjoint intervals, signed cancellation, scaling, source
validation, and production-independent decay initialization. Run:

```sh
python -m unittest discover -s tests -p 'test_sensitivity.py'
python -m unittest discover -s tests -p 'test_experiment_response.py'
python -m unittest discover -s tests -p 'test_external_production.py'
python -m unittest discover -s tests -p 'test_decay_runtime.py'
python -m unittest discover -s tests -p 'test_na62*.py'
python -m unittest discover -s tests -p 'test_response_baselines.py'
```
