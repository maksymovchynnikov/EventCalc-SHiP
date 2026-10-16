# Higgs-mixed scalar input tables

These tables contain branching fractions and proper decay lengths for the
Higgs-mixed scalar calculations listed in the [model guide](../../docs/models.md#higgs-like-scalar).
Decay inputs cover 0.01 to 63 GeV; the [generation guide](../../docs/EXHAD.md#model-coverage)
gives the production limits and supported hadronic calculations.

## Coupling convention

The native scalar coupling in EventCalc is

$$
g_S \equiv \sin\theta.
$$

Every scalar partial width and the mixing-induced production yield scale as
$g_S^2$:

$$
\Gamma_i(m_S,g_S)=g_S^2 f_i(m_S),
\qquad
N_{\rm prod}(m_S,g_S)=g_S^2 P(m_S).
$$

The branching ratios are independent of the common mixing factor. For `Scalar-quartic`, the same mixing controls decay widths; production
has an independent normalization fixed by its yield table.

## Lifetime tables

The files

```text
ctau-Scalar-2407.13587-Central.txt
ctau-Scalar-2407.13587-Lower.txt
ctau-Scalar-2407.13587-Upper.txt
ctau-Scalar-1809.01876.txt
```

contain JSON arrays despite their `.txt` suffix. Each
element is one two-number row,

```text
[mass_GeV, c_tau_times_sin_theta_squared_m]
```

with strictly increasing positive masses and positive lifetimes:

| Column | Meaning | Unit |
|---|---|---|
| 1 | scalar mass $m_S$ | GeV |
| 2 | $c\tau_1(m_S)=c\tau(m_S,\sin\theta=1)$ | m |

EventCalc uses

$$
c\tau(m_S,\sin\theta)=\frac{c\tau_1(m_S)}{\sin^2\theta},
\qquad
\sin^2\theta=\frac{c\tau_1(m_S)}{c\tau}.
$$

The corresponding unit-mixing total-width coefficient is

$$
f_{\rm tot}(m_S)=\frac{\hbar c}{c\tau_1(m_S)},
\qquad
\hbar c=1.973269804\times10^{-16}\ {\rm GeV\,m}.
$$

## Branching-ratio tables

The matching `BrRatio-Scalar-2407.13587-{Central,Lower,Upper}.json` files and
`BrRatio-Scalar-1809.01876.json` contain decay-channel branching-ratio
tables. The dimuon channel is named `muPmuM`. A partial-width coefficient can
be reconstructed as

$$
f_{\mu\mu}(m_S)=\operatorname{BR}_{\mu\mu}(m_S)f_{\rm tot}(m_S).
$$

The `Central`, `Lower` and `Upper` tables represent the mean and one-standard-deviation
scalar-width choices of [arXiv:2407.13587](https://arxiv.org/abs/2407.13587).
Their ordering can depend on mass and channel. The `1809.01876` choice uses
[arXiv:1809.01876](https://arxiv.org/abs/1809.01876) with the corrections of
[arXiv:1904.10447](https://arxiv.org/abs/1904.10447).
Each lifetime table must be used with its corresponding branching fractions.

## Hadronic channels

The neutral-kaon contribution contains equal $K_LK_L$ and $K_SK_S$ rates.
Proton and neutron pairs are separate final states. The selected hadronic
branching fractions set the total number of events; the
[hadronic generation model](../../docs/EXHAD.md#model-coverage) determines their
momenta and composition. Leptonic channels retain their exclusive kinematics.

## Interpretation of unit coupling

The production table evaluated formally at $\sin\theta=1$ is a quadratic
normalisation coefficient. Multiply that coefficient by $\sin^2\theta$ in the small-mixing regime
where the calculation applies. Formal unit-mixing coefficients can exceed
one because they extract the leading quadratic term in this expansion.
