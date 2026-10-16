# Compact tables and optional full production data

Ordinary dark-photon and inelastic-DM generation uses the compact
mass-angle-energy distributions in `Distributions/Dark-photons/production/` and `Distributions/Inelastic-DM/production/`. These
include primary and cascade production with the calibrated model and updated
target/beam profile. A lifetime or coupling change never regenerates production.
The tables occupy about 25 MB for both models and both inelastic splittings.
They store normalized angular-energy densities and yields per proton and per
epsilon squared. NumPy compression keeps the same three-dimensional tabulation
small; these files are not individual production-event samples.

The default `production_backend: compact` integrates out collision positions
and assumes a point source at the target front-face centre and uniform azimuth
when propagating the LLP. DP tables cover 0.01–20 GeV. Inelastic tables cover
mediator masses 0.04–40 GeV for Delta=0.05 and 0.1, with mA/mChi=3 and alphaD=0.1;
the launcher also enforces the physical chi-prime decay threshold. The
inelastic distributions describe chi-prime, including finite mediator width
and polarization in production. The generated chi-prime mass is not the
mediator mass supplied to the command.

A paired check moved only the production positions to the target front-face
centre, keeping momenta and decay samples fixed. Accepted yields changed by
less than 0.4% at a proper decay length of 10,000 m. Both coupling-squared
sensitivity boundaries shifted by less than about 1% for dark photons and 2%
for inelastic DM in the tested benchmarks. These were DP masses 0.3, 1.1 and
5 GeV; and inelastic mediator masses 5 and 20 GeV at Delta=0.1, and 20 and
28 GeV at Delta=0.05. The largest shift was near the inelastic mass-reach
endpoint. This checks the position approximation, not table discretization or
production-model uncertainty. The manuscript contours retain full positions.

## Full dataset on EOS

The source samples, correlated collision positions, form factors and PDF/scale
weights are archived at:

```
/eos/home-o/ovchynni/EventCalc-SHiP/production/20261016-v1/eventcalc-production-20261016-v1.tar.gz
```

The immutable archive and every extracted file are bound to SHA-256 checksums
in `eventcalc/production/dark_photon/dataset.json`. The archive is about 1.24 GB;
extracted inputs occupy about 1.41 GB. It is stored separately from Git.
CERN access to this EOS path is required for retrieval. From an authenticated
CERN account:

```sh
python -m eventcalc.production.dark_photon.datasets fetch --ssh-user YOUR_CERN_USERNAME
```

A shared SSH connection can be selected with `--ssh-control-path PATH`.
Alternatively, copy the archive separately and install it offline:

```sh
python -m eventcalc.production.dark_photon.datasets install --archive /path/to/eventcalc-production-20261016-v1.tar.gz
python -m eventcalc.production.dark_photon.datasets verify
```

The default destination is `$XDG_CACHE_HOME/eventcalc-ship/production/20261016-v1`,
or `~/.cache/eventcalc-ship/production/20261016-v1` when XDG_CACHE_HOME is unset.
All checkouts share this cache. `EVENTCALC_PRODUCTION_DATA` can instead name a
verified extracted dataset. The fetch command checks existing data instead of
overwriting them, verifies archive contents before installation and never
writes large inputs into a source checkout. Ordinary simulations never fetch
data automatically.

Use `--production-backend correlated` in `simulate.py` or `run_batch.py`, or
`production_backend: correlated` in a card, to retain full production-position
correlations. `--production-power` controls this optional source integration.
For other inelastic splittings, use this representation rather than interpolate
between the two benchmark splittings.

`model_sensitivity.py` continues to calculate production-uncertainty bands
on both coupling boundaries using the full dataset. It requires the EOS data
once, then works offline. Its automatic bands are unchanged. The generic
`scan_sensitivity.py` runtime provider follows the production backend in its
model card; compact-table scans are central-only unless the caller provides
additional coherent production variations. See `SENSITIVITY.md`.

Source sample arrays, per-mass export caches, intermediate fit audits and PDF
weight chunks do not belong in Git. Historical copies already committed remain
in Git history; removing files in this update does not rewrite published history.
