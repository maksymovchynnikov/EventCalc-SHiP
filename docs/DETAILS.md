# Physics inputs and sampling

EventCalc-SHiP combines production, propagation and decays of long-lived
particles. The [model guide](models.md) defines each interaction and coupling;
[production distributions](PRODUCTION_DATA.md) describes the target calibration
for `Dark-photons` and `Inelastic-DM`.

## Decay inputs

| Model | Calculation |
| --- | --- |
| `HNL` | Widths and matrix elements of [arXiv:1805.08567](https://arxiv.org/abs/1805.08567); charged and neutral weak currents are treated separately |
| `Scalar-mixing`, `Scalar-quartic` | Higgs-mixed scalar widths of [arXiv:2407.13587](https://arxiv.org/abs/2407.13587), or [arXiv:1809.01876](https://arxiv.org/abs/1809.01876) with the corrections of [arXiv:1904.10447](https://arxiv.org/abs/1904.10447) |
| `ALP-photon` | Photon-coupled ALP calculation of [arXiv:1904.02091](https://arxiv.org/abs/1904.02091), including radiative lepton pairs |
| `ALP-SU2L`, `ALP-mixed` | Electroweak operators of [arXiv:1901.02031](https://arxiv.org/abs/1901.02031); the implemented decay approximation has $\mathrm{Br}(a\to\gamma\gamma)=1$ |
| `ALP-fermion` | Fermion-universal ALP rates and renormalization group evolution of [arXiv:2501.04525](https://arxiv.org/abs/2501.04525) and [arXiv:2504.06828](https://arxiv.org/abs/2504.06828) |
| `Dark-photons` | Electromagnetic-current widths of [arXiv:2201.01788](https://arxiv.org/abs/2201.01788), used in [arXiv:2409.11096](https://arxiv.org/abs/2409.11096) |
| `Inelastic-DM` | Off-shell electromagnetic-current integral and complete excited-state decay chains, as defined in the [inelastic guide](INELASTIC_DM.md) |

The hadronic final states follow the model of
[arXiv:2609.16104](https://arxiv.org/abs/2609.16104) when exHad is enabled.
The [hadronic generation guide](EXHAD.md) gives coverage and installation.
The `ALP-fermion` [known limitation](../README.md#known-bugsproblems) applies to
three squared matrix elements. Input lifetimes and branching fractions include
the renormalization group evolution of the stated fermion-universal model.

## Heavy neutral lepton mixing

For `HNL`, let $\xi_\alpha=|U_\alpha|^2/U^2$ be the flavour fractions, with
$U^2=\sum_\alpha|U_\alpha|^2$. Production coefficients $Y_\alpha$ and widths
$\Gamma_\alpha$ for pure electron, muon and tau mixing combine as

$$
Y=\sum_\alpha\xi_\alpha Y_\alpha,
\qquad \Gamma=U^2\sum_\alpha\xi_\alpha\Gamma_\alpha.
$$

Each normalized production spectrum has weight $\xi_\alpha Y_\alpha/Y$.
Decay-channel fractions and normalized matrix elements use
$\xi_\alpha\Gamma_\alpha/\sum_\beta\xi_\beta\Gamma_\beta$.
Thus production and decay can have different flavour weights.

## Other production inputs

Light mesons used by the other tabulated models follow the Pythia8 setup of
[arXiv:1904.02091](https://arxiv.org/abs/1904.02091). These tables contain
primary hadronic production. Cascade production is included explicitly in the
calibrated dark photon calculation and in the electromagnetic photon sources.
The latter follow [arXiv:2509.14310](https://arxiv.org/abs/2509.14310).
Charm and bottom production use the [SHiP calculation](https://cds.cern.ch/record/2115534).
The charged-kaon spectrum for `ALP-SU2L` follows
[arXiv:2004.07974](https://arxiv.org/abs/2004.07974), with a charge-summed yield
of $0.36$ kaons per incident proton.

## Propagation and decays

Calibrated `Dark-photons` and `Inelastic-DM` use compact production tables by
default. Their optional correlated source keeps each momentum with its
collision position. Both intersect the trajectory with all faces of the
truncated-pyramid decay volume. For a particle of mass $m$, momentum magnitude
$p$ and proper decay length $c\tau$, the probability between path lengths
$s_{\rm in}$ and $s_{\rm out}$ from its production point is

$$
P_{\rm decay}=e^{-m s_{\rm in}/(p c\tau)}-e^{-m s_{\rm out}/(p c\tau)}.
$$

For other production tables, the launchers first sample the polar-angle and
energy density, then the exponential longitudinal decay position and transverse
acceptance. Their lower energy boundary removes probabilities below
$e^{-15}$ to concentrate sampling on particles that survive to the volume.
The runtime API samples an interpolated cumulative density; the standard
launcher uses weighted proposals. The compact calibrated sources use their
full physical energy range.

Two-body decays are isotropic in the parent rest frame. Three-body channels
use their squared matrix elements; four-body and higher multiplicities use
phase space unless a dedicated decay model supplies their correlations.
Kinematically closed channels have zero branching fraction before events are
drawn. Hadronic thresholds are imposed when sampling the invariant mass of a
quark pair, while the partonic matrix elements use their current-quark masses.
Unstable particles decay before the final-state momenta are boosted to the
laboratory frame.

## Detector acceptance

The generator returns momenta, vertices and decay weights. The
[sensitivity module](SENSITIVITY.md#selections) integrates the magnetic
transport and visible-particle selections at the ECAL end. Separately,
`eventcalc.analysis.events_analysis` can apply a configurable straight-line
projection to exported events. Its default plane is at 95 m, with full widths
4 by 6 m; it is a distinct diagnostic selection. Both calculate geometric
acceptance with assumed unit reconstruction efficiency.
