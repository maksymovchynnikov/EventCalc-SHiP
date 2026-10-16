# Models and coupling conventions

These names are the values accepted by `--model` and `--llp`. The standard
simulation ranges are the intersection of the selected production and decay
inputs; channel thresholds can further restrict a requested point.

| Model | Physical particle | Standard simulation mass range (GeV) |
| --- | --- | --- |
| `Dark-photons` | Visibly decaying vector mediator | 0.01 to 20 |
| `Inelastic-DM` | Excited fermion produced through a vector mediator | Mediator mass 0.04 to 40; see [decay threshold](INELASTIC_DM.md#inputs-and-generation) |
| `HNL` | Majorana heavy neutral lepton | 0.02 to 5.27 |
| `Scalar-mixing` | Scalar with Higgs-like couplings | 0.02 to 5.12 |
| `Scalar-quartic` | Scalar pair production through the Higgs portal | 0.02 to 5.12 |
| `ALP-photon` | Pseudoscalar coupled to photons | 0.02 to 4 for combined production |
| `ALP-fermion` | Pseudoscalar coupled universally to fermions | 0.02 to 5.13 |
| `ALP-SU2L` | Pseudoscalar coupled to weak gauge bosons | 0.01 to 5.1 |
| `ALP-mixed` | Coherent photon and weak-operator combination | 0.02 to 4 |

The calibrated production ranges use compact tables. Alternative sources can
have different coverage; out-of-range masses are rejected. The
[hadronic generation guide](EXHAD.md#model-coverage) separately states exHad's
support. Its upper limit can require an explicit Pythia8 selection within the
production range.

## Heavy neutral lepton

`HNL` is a Majorana fermion $N$ mixed with active neutrinos. Its weak charged
and neutral currents are

$$
\mathcal L\supset-\frac{g_2}{\sqrt2}W^-_\mu
\sum_\alpha U_\alpha\bar\ell_\alpha\gamma^\mu P_LN
-\frac{g_2}{2\cos\theta_W}Z_\mu
\sum_\alpha U_\alpha\bar\nu_\alpha\gamma^\mu P_LN+\mathrm{h.c.}
$$

Here $g_2$ is the weak gauge coupling, $\theta_W$ the weak mixing angle,
$P_L=(1-\gamma_5)/2$, and $\alpha=e,\mu,\tau$ labels lepton flavour.
The rates follow [arXiv:1805.08567](https://arxiv.org/abs/1805.08567).


`HNL` takes a mixing pattern $(\xi_e,\xi_\mu,\xi_\tau)$ summing to one, with
$U_\alpha^2=U^2\xi_\alpha$. Production yields, angular and energy densities,
widths, branching ratios and decay matrix elements are tabulated for the three
pure mixings and merged for the requested pattern, as described in
[physics guide](DETAILS.md).

## Fermion-coupled ALP

`ALP-fermion` couples the pseudoscalar $a$ to the fermion axial currents:

$$
\mathcal L\supset\frac{\partial_\mu a}{f_a}
\sum_f c_f\bar f\gamma^\mu\gamma_5 f.
$$

The scale $f_a$ has units of energy. The dimensionless coefficients are
$c_f=1$ at the matching scale $\Lambda=1$ TeV, with no independent gluon
operator there. Renormalization group evolution and loop contributions follow
[arXiv:2501.04525](https://arxiv.org/abs/2501.04525). The Higgs vacuum expectation
value is $v_h=246$ GeV.


For `ALP-fermion`, EventCalc uses the dimensionless coupling

$$
g_Y \equiv y \equiv \frac{2v_h}{f_a}.
$$

Every `coupling_squared` output for this model is $g_Y^2$.
Its [table dictionary](../Distributions/ALP-fermion/README.md) defines the
lifetime and production normalization.

## Higgs-like scalar

`Scalar-mixing` describes a scalar $S$ mixed with the Higgs boson. In particular,
its fermion interaction is

$$
\mathcal L\supset-\sin\theta\,S\sum_f\frac{m_f}{v_h}\bar f f,
$$

where $m_f$ is the fermion mass, $v_h=246$ GeV and $\theta$ is the scalar mixing
angle. The same mixing multiplies its Higgs-like interactions with gauge bosons.
`Scalar-quartic` uses pair production through the Higgs portal; its production
normalization is fixed by the installed yield table. The coupling input controls
the decay lifetime in both cases.


For `Scalar-mixing` the coupling is the mixing angle, $g_S=\sin\theta$.
Production rates and every partial width scale as $\sin^2\theta$, so

$$
c\tau(m_S,\sin\theta)=\frac{c\tau_1(m_S)}{\sin^2\theta},
\qquad
\sin^2\theta=\frac{c\tau_1(m_S)}{c\tau},
$$

where the tables store $c\tau_1(m_S)=c\tau(m_S,\sin\theta=1)$ in metres. Both
scalar models read one prescription-keyed pair of tables under
`Distributions/Scalar-mixing`: the hadronic widths of
[2407.13587](https://arxiv.org/abs/2407.13587) as the mean of its Monte Carlo
sample and one standard deviation below and above it
(`2407.13587-Central`, `2407.13587-Lower`, `2407.13587-Upper`), and the
calculation of [1809.01876](https://arxiv.org/abs/1809.01876) with the
corrections of [1904.10447](https://arxiv.org/abs/1904.10447)
(`1809.01876`). The default is `2407.13587-Central`; select another with
`--scalar-prescription` or the card key `scalar_prescription`. The data
dictionary is in
[`Distributions/Scalar-mixing/README.md`](../Distributions/Scalar-mixing/README.md)
and in
[`Distributions/Scalar-mixing/coupling.json`](../Distributions/Scalar-mixing/coupling.json).

## Dark photons and inelastic dark matter

For `Dark-photons`, $A'$ is a massive vector particle coupled to the
Standard Model electromagnetic current $J_{\rm EM}^\mu$:

$$
\mathcal L\supset\epsilon e A'_\mu J_{\rm EM}^\mu,
$$

where $e$ is the electric charge and $\epsilon$ is dimensionless kinetic mixing.
Production probabilities and visible widths scale as $\epsilon^2$.
`Inelastic-DM` adds an interaction between $A'$ and two dark states of different
masses; its mass definitions, dark coupling and decay law are given in the
[inelastic model guide](INELASTIC_DM.md). The target calibration and the
available production choices for both models are defined in the
[production guide](PRODUCTION_DATA.md).

## Photon-coupled ALP

For `ALP-photon`, the pseudoscalar $a$ couples to the photon field strength
$F_{\mu\nu}$ and its dual $\widetilde F^{\mu\nu}$:

$$
\mathcal L\supset\frac{g_{a\gamma\gamma}}{4}aF_{\mu\nu}\widetilde F^{\mu\nu}.
$$

The coupling $g_{a\gamma\gamma}$ has units of GeV$^{-1}$. Production includes
Primakoff conversion of primary and cascade photons and radiative meson decays;
the channel inputs follow [arXiv:1904.02091](https://arxiv.org/abs/1904.02091).
`--alp-production primary`, `cascades` or `combined` selects the photon source.

## $SU(2)_L$ ALP

The `ALP-SU2L` model uses

$$
\mathcal L\supset C_WaW^I_{\mu\nu}\widetilde W^{I\mu\nu},
\qquad
C_W=\frac{\alpha_2}{4\pi}g_W,
\qquad
\frac{c_W}{f_a}=-C_W,
$$

Here $W^I_{\mu\nu}$ is the weak field strength, the tilde denotes its dual,
$\alpha_2=g_2^2/(4\pi)$, and $g_W$ has units of GeV$^{-1}$.
The hypercharge and Higgs-current coefficients are $c_B=c_{a\Phi}=0$. The last relation converts to the convention of
[1901.02031](https://arxiv.org/abs/1901.02031), where the operator is
$-c_WaW^I_{\mu\nu}\widetilde W^{I\mu\nu}/f_a$. Electroweak symmetry breaking
induces

$$
C_\gamma^{\rm ind}=\sin^2\theta_W C_W,
\qquad
g_{a\gamma\gamma}=\frac{\alpha_{\rm em}}{\pi}g_W.
$$

Four processes contribute to the ALP flux: rare $B\to X_{s,d}a$ decays,
$K^\pm\to\pi^\pm a$ below the charged-kaon threshold, Primakoff production by
photons from the proton-target interaction, and Primakoff production by
photons in the electromagnetic cascade. Let $\widehat Y_B$ and $\widehat Y_K$
denote the production coefficients per $g_W^2$, and $\widehat Y_{\rm primary}$
and $\widehat Y_{\rm cascade}$ the Primakoff coefficients per
$g_{a\gamma\gamma}^2$. Since $g_{a\gamma\gamma}=kg_W$ with
$k=\alpha_{\rm em}/\pi$, the combined coefficient per $g_W^2$ is

$$
\widehat Y_{\rm tot}=\widehat Y_B+\widehat Y_K
+k^2(\widehat Y_{\rm primary}+\widehat Y_{\rm cascade}),
$$

and the production probability is $P_{\rm prod}=g_W^2\widehat Y_{\rm tot}$.
The normalized angular and energy density is

$$
f_{\rm tot}=
\frac{\widehat Y_Bf_B+\widehat Y_Kf_K
+k^2\widehat Y_{\rm primary}f_{\rm primary}
+k^2\widehat Y_{\rm cascade}f_{\rm cascade}}
{\widehat Y_{\rm tot}},
$$

which EventCalc reads as one production model. The charged-kaon flux is
$0.36=0.29+0.07$ kaons per proton on target, summed over the two charges; its
branching coefficient is evaluated as a function of $m_a$ and vanishes exactly
at $m_a=m_{K^\pm}-m_{\pi^\pm}$.

For a requested $c\tau$, EventCalc obtains the coupling from

$$
g_W^2=(1\,\mathrm{GeV}^{-1})^2
\frac{(c\tau)_{g_W=1\,\mathrm{GeV}^{-1}}}{c\tau},
$$

and multiplies $\widehat Y_{\rm tot}$ by its numerical value in
$\mathrm{GeV}^{-2}$. The decay model of this benchmark is
$\mathrm{Br}(a\to\gamma\gamma)=1$; hadronic ALP widths need a different decay
description. Every production, lifetime and decay table this benchmark needs
is installed under `Distributions/ALP-SU2L/`.

## Mixed photon and $SU(2)_L$ ALP

The `ALP-mixed` model implements both operators coherently. Its two additional
inputs are the operator fraction $0\leq\xi\leq1$ and the interference sign,
`constructive` or `destructive`. Here $C_\gamma^{\rm dir}$ multiplies $aF_{\mu\nu}\widetilde F^{\mu\nu}$,
$C_W$ multiplies the weak operator defined above, and $\Lambda$ is an overall
inverse-energy coefficient. The convention is

$$
C_W=\Lambda\xi,
\qquad
C_\gamma^{\rm dir}=s\Lambda(1-\xi),
\qquad s=\pm1,
$$

so that

$$
g_{a\gamma\gamma}^{\rm tot}
=4\Lambda\left[s(1-\xi)+\sin^2\!\theta_W\,\xi\right]
=g_{a\gamma\gamma}^{\rm dir}+\frac{\alpha_{\rm em}}{\pi}g_W .
$$

The two amplitudes add coherently before their squared magnitude sets the width. For the diphoton benchmark,

$$
c\tau=\frac{64\pi\hbar c}
{m_a^3\left|g_{a\gamma\gamma}^{\rm tot}\right|^2},
$$

from which EventCalc infers $|g_{a\gamma\gamma}^{\rm tot}|^2$ for a requested
$c\tau$. At fixed $\xi$ and sign, the relative weight of the
flavour-changing $B$ and charged-kaon production against the primary- and
cascade-photon production is

$$
\frac{g_W^2}{|g_{a\gamma\gamma}^{\rm tot}|^2}
=\frac{(4\pi/\alpha_2)^2\xi^2}
{16\left[s(1-\xi)+\sin^2\!\theta_W\,\xi\right]^2}.
$$

No spectrum has to be generated at launch: the $B+K$ component is
reconstructed from the installed `ALP-SU2L` table and the installed
primary- and cascade-photon tables, so $\xi=0$ reproduces the combined
photophilic production and $\xi=1$ reproduces the pure `ALP-SU2L` physics. The
common tabulated mass range is 0.02 to 4 GeV. The exact destructive
cancellation point is refused, because a diphoton-only model has no finite
signal there.
