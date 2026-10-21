# Dark photon and excited dark matter production

The older dark-photon Lower/Central/Upper fluxes follow
[Kyselov and Ovchynnikov](https://arxiv.org/abs/2409.11096) and remain available
through the `legacy-*` options described below. The newer default dark-photon
production calculation is work in preparation; please contact the authors for
details.

The default `Dark-photons` and `Inelastic-DM` distributions combine radiative
meson decays, fragmentation, proton bremsstrahlung and Drell-Yan production.
Meson and dimuon measurements from NA27, NA60 and NA61/SHINE jointly constrain
the fragmentation normalization and the proton virtuality cutoff. The fitted
single-scale bremsstrahlung model has $\Lambda_p=0.4083$ GeV; its off-shell
suppression follows [arXiv:2108.05900](https://arxiv.org/abs/2108.05900) and
[arXiv:2409.09123](https://arxiv.org/abs/2409.09123). This is a conservative
choice with a small, nonzero fitted bremsstrahlung contribution. Alternative
off-shell descriptions and heavier vector mesons in fragmentation can increase
the total forward yield and are outside the tabulated uncertainty variations.

NA50 Drell-Yan measurements at [400 GeV/c](https://arxiv.org/abs/nucl-ex/0612012)
and [450 GeV/c](https://arxiv.org/abs/hep-ex/0603049) calibrate the proton-initiated
continuum. Predictions use the published mass, rapidity and Collins–Soper
angular cuts. The fit profiles the nuclear dependence and shared experimental
errors. Because the 450 GeV publication does not give a complete covariance,
the fit spans independent, partly shared and fully shared systematic errors.
For central CT18, the tungsten normalization is multiplied by 1.2347. The
correction applies to primary and secondary proton collisions, including the
inelastic virtual-mass integral; pion and other projectiles are unchanged.
Continuing it to other pair masses and secondary-proton energies is an
assumption. NA50 does not constrain the far-forward high-mass shape.

The compact distributions include this correction. The small `production/na50/`
calibration package stores the fitted coefficients, source identities and
checksums. The full source-event archive remains unchanged and is supplied
separately. [Sensitivity scans](SENSITIVITY.md#dark-photon-and-inelastic-dark-matter-sensitivity)
apply the PDF-specific fit and its profile variations before calculating both
sensitivity boundaries.

## Target and normalization

The target contains 33 tungsten disks, with 1.33 m of tungsten over 1.46 m
including helium gaps, and radius 0.25 m. The incident beam follows a ring of
radius 5 cm with independent Gaussian smearing of 16 mm in each transverse
direction. All longitudinal coordinates start at the target front face.
Primary collisions involve the incident beam; the cascade contains secondary
hadron collisions throughout the shower.

At fixed mediator mass $m_{A'}$, each source has production coefficient
$Y(m_{A'})$ and normalized density $f(\theta,E\mid m_{A'})$:

$$
\frac{d^2N}{d\theta\,dE}
=N_{\rm POT}\epsilon^2Y(m_{A'})f(\theta,E\mid m_{A'}),
\qquad \int d\theta\,dE\,f=1.
$$

Here $N_{\rm POT}$ is the number of incident protons and $\epsilon$ is kinetic
mixing. The angle $\theta$ is measured in radians relative to the beam; $E$ is
laboratory energy in GeV. For `Dark-photons` these are mediator kinematics;
for `Inelastic-DM` they describe the excited state $\chi'$. The latter includes
finite mediator width and polarization in $A'^{(*)}\to\chi\chi'$.

## Compact distributions

The compact tables occupy about 25 MiB. They contain yields and distributions
in mass, angle and energy, compressed with NumPy:

| Model | Directory | Mediator mass range | Other parameters |
| --- | --- | --- | --- |
| `Dark-photons` | `Distributions/Dark-photons/production/` | 0.01 to 20 GeV | Visible mediator decays |
| `Inelastic-DM` | `Distributions/Inelastic-DM/production/` | 0.04 to 40 GeV | $m_{A'}/m_\chi=3$, $\alpha_D=0.1$, $\Delta=0.05$ or $0.1$ |

The [inelastic guide](INELASTIC_DM.md) defines these parameters and the physical
decay threshold. Mediator pole masses above the collision energy describe
production through a virtual mediator; the produced pair still obeys the
collision kinematics.

The launchers use `production_backend: compact` by default. They interpolate
the differential yield linearly in mass and the density bilinearly in angle
and energy, then normalize above the propagated particle's mass. Changing the
lifetime reuses the same production distribution. Propagation assumes a point
source at the target front-face centre and uniform azimuth.

A check held momenta and decay samples fixed while moving production positions
to that point. At $c\tau=10^4$ m, accepted yields changed by less than 0.4%.
Both $\epsilon^2$ sensitivity boundaries shifted by less than about 1% for
`Dark-photons` and 2% for `Inelastic-DM`. Tested mediator masses were 0.3, 1.1
and 5 GeV for `Dark-photons`; 5 and 20 GeV at $\Delta=0.1$, and 20 and 28 GeV
at $\Delta=0.05$, for `Inelastic-DM`. This comparison isolates the production
position approximation. Table discretization and production uncertainties
require separate checks.

## Full production dataset

The full samples include correlated collision positions, momenta and the
weights needed to vary proton form factors, parton distribution functions and
QCD scales. They are supplied separately and are not needed for ordinary event
generation. Ask the maintainers for a compatible archive if you need correlated
production or production-uncertainty scans. The archive is about 1.24 GB, or
1.41 GB after extraction. Install it outside the source directory:

```sh
python -m eventcalc.production.dark_photon.datasets install --archive /path/to/production-data.tar.gz --destination /tmp/eventcalc-production-data
python -m eventcalc.production.dark_photon.datasets verify --destination /tmp/eventcalc-production-data
export EVENTCALC_PRODUCTION_DATA=/tmp/eventcalc-production-data
```

Installation checks the archive and every extracted file against the checksums
in `eventcalc/production/dark_photon/dataset.json`. A mismatched archive is
rejected. If the provider gives a remote archive path, the same checked
installation is available through:

```sh
python -m eventcalc.production.dark_photon.datasets fetch --source USER@HOST:/path/to/production-data.tar.gz --destination /tmp/eventcalc-production-data
```

Use `--ssh-control-path PATH` to reuse an authenticated SSH connection. No server
or private account is assumed. `EVENTCALC_PRODUCTION_DATA` selects the extracted
dataset; all checkouts can share that location while it is in use. Ordinary
generation does not download additional data automatically.

After the calculation, remove the temporary extracted dataset and any downloaded
archive, and unset `EVENTCALC_PRODUCTION_DATA`. Do not put full production samples,
per-event uncertainties or generated event outputs into Git, release archives or
copies of the source tree. Keep useful large results in separate storage and
delete disposable test outputs; retain the test code and small fixtures.

Select `--production-backend correlated` in either launcher, or
`production_backend: correlated` in a card, to preserve momentum-position
correlations and allow other physical inelastic splittings. The numerical
integration size is controlled by `--production-power`.

`python -m eventcalc.model_sensitivity` uses the full dataset to calculate
automatic production bands on both mixing boundaries. The generic
`python -m eventcalc.scan_sensitivity` follows its model card; compact scans
contain the central production prediction unless the caller supplies additional
variations. See the [sensitivity guide](SENSITIVITY.md).

## Selecting production contributions

`--dp-production primary`, `cascade` or `combined` selects contributions from
the calibrated calculation; `combined` is the default. For `Dark-photons`,
explicit `legacy-primary`, `legacy-cascade`, `legacy-combined` and
`legacy-brem-cascade` select the alternative flux tables based on
[arXiv:2409.11096](https://arxiv.org/abs/2409.11096), with `--uncertainty`
`lower`, `central` or `upper`. The interactive menu exposes these choices
when `EVENTCALC_DP_PRODUCTION=legacy` is set. Their rate variations are separate
from the accepted-event production bands of the calibrated calculation.
