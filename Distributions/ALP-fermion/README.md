# Fermion-coupled ALP input tables

## Coupling convention

The native coupling of the `ALP-fermion` tables and of EventCalc is

$$
g_Y \equiv y \equiv \frac{2v_h}{f_a}.
$$

For this model, every EventCalc quantity called `coupling_squared` or
`Squared coupling` is $g_Y^2$.

Here $v_h=246$ GeV is the Higgs vacuum expectation value and $f_a$ is the
ALP decay constant. The [model guide](../../docs/models.md#fermion-coupled-alp)
defines the interaction. The fermion coefficients are $c_f=1$ at the matching
scale $\Lambda=1$ TeV. Their evolution to the ALP mass follows
[arXiv:2501.04525](https://arxiv.org/abs/2501.04525).
`coupling.json` records the same normalization and is checked before loading.

## Lifetime table

`ctau-ALP-fermion.txt` is a tab-separated, headerless two-column table:

| Column | Meaning | Unit |
|---|---|---|
| 1 | ALP mass $m_a$ | GeV |
| 2 | $c\tau_1(m_a)\equiv c\tau(m_a,g_Y=1)$ | m |

EventCalc uses

$$
g_Y^2=\frac{c\tau_1(m_a)}{c\tau},
\qquad
c\tau(m_a,g_Y)=\frac{c\tau_1(m_a)}{g_Y^2}.
$$

## Production-yield table

`Total-yield-ALP-fermion.txt` is a tab-separated, headerless two-column table:

| Column | Meaning | Unit |
|---|---|---|
| 1 | ALP mass $m_a$ | GeV |
| 2 | $Y_1(m_a)\equiv N_{\mathrm{prod}}/(N_{\mathrm{POT}}g_Y^2)$ | per proton on target per $g_Y^2$ |

The absolute number of produced ALPs is

$$
N_{\mathrm{prod}}=N_{\mathrm{POT}}Y_1(m_a)g_Y^2.
$$

The distribution files `DoubleDistr-ALP-fermion.txt` and
`Emax-ALP-fermion.txt` determine the normalized kinematic shape and maximum
energy; they do not redefine the overall coupling normalization.

## Widths and branching ratios

If $W_i(m_a)$ is the coupling-independent partial-width function imported by
SensCalc, then

$$
\Gamma_i(m_a,g_Y)
=\left(\frac{g_Y}{2v_h}\right)^2W_i(m_a)
\equiv g_Y^2f_i(m_a),
\qquad
f_i(m_a)=\frac{W_i(m_a)}{(2v_h)^2}.
$$

Thus

$$
c\tau=\frac{\hbar c}{g_Y^2f_{\mathrm{tot}}},
\qquad
\operatorname{BR}_i=\frac{f_i}{f_{\mathrm{tot}}}.
$$

`ALP-fermion-decay.json` contains the branching-ratio tables and decay
kinematics. Its branching ratios are
independent of the common $g_Y$ normalization. Two switches act on the
hadronic event composition and on nothing else: the launch-card field
`hadronization` turns the matched generator on or off, and `exhad.json` fixes
the benchmark it runs and its mass window. Enabling exHad does not alter
production yields, branching ratios, partial widths, or lifetime.

## Hadronic events and input checks

The [hadronic generation guide](../../docs/EXHAD.md#model-coverage) defines the
mass coverage and channel selection. EventCalc uses the branching fractions
in these tables to set decay probabilities; exHad generates the corresponding
hadronic momenta. `input_revision.json` records input checksums.
The [known squared-matrix-element limitation](../../README.md#known-bugsproblems)
applies to three exclusive channels.
