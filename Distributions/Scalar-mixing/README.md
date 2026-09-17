# Higgs-mixed scalar input tables

The four branching-ratio/lifetime pairs were refreshed from the supplied
exports on 15 September 2026. Each covers 0.01--63 GeV on the same 11,012-node
mass grid. No rates were extrapolated or renormalized during import; channel
labels and PDG padding were converted to EventCalc's existing conventions.
exHad generates scalar decays from 2 to 63 GeV (showered above 5 GeV); this
input coverage does not extend any production distribution (5.12 GeV end).

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

The branching ratios are independent of the common mixing factor. This
convention applies to `Scalar-mixing`; it does not apply to the independent
Higgs-quartic production coupling used by the separate `Scalar-quartic`
model.

## Lifetime tables

The files

```text
ctau-Scalar-2407.13587-Central.txt
ctau-Scalar-2407.13587-Lower.txt
ctau-Scalar-2407.13587-Upper.txt
ctau-Scalar-1809.01876.txt
```

retain their historical `.txt` names but contain strict JSON arrays. Each
element is one two-number row,

```json
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

The `Central`, `Lower`, and `Upper` labels identify the three scalar-width
prescriptions from arXiv:2407.13587. They are model choices, not guaranteed
to be pointwise ordered numerical envelopes. The `1809.01876` choice is the
corrected Boiarska--Winkler/1904 prescription and provides an independent
prescription check.

## Matched hadronic event generation

The channel labels are `ePeM`, `muPmuM`, `tauPtauM`, `PipPim`, `2Pi0`,
`2Kch`, `KLKL`, `KSKS`, `ppbar`, `nnbar`, `2Pip2Pim`, `PipPim2Pi0`,
`Jets-cc`, `Jets-GG`, `Jets-bb`, and `Jets-ss`. In particular, the neutral
kaon rate is represented by equal `KLKL` and `KSKS` rows; there is no
mixed-CP `KLKS` row. The proton and neutron pairs are explicit rate-owning
rows.

The installed matched scalar model is active for
$2.00\leq m_S\leq5.00$ GeV for each of the four prescriptions. At each mass,
EventCalc combines all thirteen selected hadronic rows--the two-pion,
two-kaon, nucleon-pair, four-pion, gluon, strange, charm, and bottom
entries--and requests the complete scalar event measure once. The three
leptonic rows remain native EventCalc decays. The selected branching-ratio
and lifetime prescription continues to own the absolute rate; exhad changes
only the composition and kinematics of the pooled hadronic events. The
unresolved component changes continuously to source-conditioned default
fragmentation from 4 to 5 GeV.

## Input validation

The refreshed tables have finite, nonnegative branching fractions that sum
to one to floating-point precision at every mass node. The malformed pion
and kaon threshold rows in the older exports, and the artificial strange,
charm and gluon cutoffs, are absent. No threshold nodes need to be dropped
from these inputs. Lifetime values are finite and positive throughout.

The installed lifetime and branching-ratio tables use the same mass grid.
Near thresholds, interpolate

$$
f_{\rm tot}=\hbar c/c\tau_1
\quad\text{and}\quad
f_{\mu\mu}=\operatorname{BR}_{\mu\mu}f_{\rm tot},
$$

not $c\tau_1$ itself. This preserves positive widths and a branching ratio in
$[0,1]$. The experimental LHC package in
`output/lhc_absolute_yield_package` implements and validates this procedure.

## Interpretation of unit coupling

The production table evaluated formally at $\sin\theta=1$ is a quadratic
normalisation coefficient. Some unit-mixing branching coefficients used to
construct production exceed one, so it must not be interpreted as a
physically realisable large-mixing event count. In the small-mixing regime of
interest, multiply the coefficient by $\sin^2\theta$.
