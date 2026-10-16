# Fermion-coupled ALP input tables

## Coupling convention

The native coupling of the `ALP-fermion` tables and of EventCalc is

$$
g_Y \equiv y \equiv \frac{2v_h}{f_a}.
$$

For this model, every EventCalc quantity called `coupling_squared` or
`Squared coupling` is $g_Y^2$. There is no additional factor-of-two or
factor-of-four conversion between the tabulated coupling and $g_Y$.

The authoritative SensCalc definition is the phenomenology chapter heading
in `3. ALP-fermion sensitivity.nb`, which states $y=2v_h/f_a$. A later
lifetime-plot annotation in that notebook states $y=v_h/f_a$; that annotation
is stale and non-normative and must not be used to interpret these files.

BC10 sets the ultraviolet fermion coefficient $c_f=1$ at the EFT/RG matching
scale $\Lambda=1\,\mathrm{TeV}$. The source label `scale-1000.-GeV` refers to
that $\Lambda$, not to $f_a=1\,\mathrm{TeV}$; $f_a$ remains encoded through
$g_Y=2v_h/f_a$.

The same contract is stored in machine-readable form in `coupling.json`.
EventCalc validates its normalization-bearing fields before loading the ALP
tables, so a missing or silently redefined convention fails closed.

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

The post-processing script may rescale $c\tau_1$ to keep it consistent with
the surviving kinematically allowed width, but it does not change the
$g_Y=1$ normalization or the $1/g_Y^2$ scaling.

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
kinematics, not coupling-dependent physical widths. Its branching ratios are
independent of the common $g_Y$ normalization. Two switches act on the
hadronic event composition and on nothing else: the launch-card field
`hadronization` turns the matched generator on or off, and `exhad.json` fixes
the benchmark it runs and its mass window. Enabling exHad does not alter
production yields, branching ratios, partial widths, or lifetime.

## Matched hadronic event generation

The installed matched ALP model is active for
$1.910\leq m_a\leq5.00$ GeV. At each mass, EventCalc combines the selected
`Jets-GG`, `Jets-ss`, `Jets-cc`, `ppbar`, and `nnbar` branching fractions
and requests that number of events once from the pseudoscalar event measure.
The generator therefore samples the aggregate hadronic composition rather
than treating these five table rows as separate fragmentation sources.
The `ePeM`, `muPmuM`, `tauPtauM`, and `2gamma` rows remain native EventCalc
decays. The unresolved component changes continuously to source-conditioned
default fragmentation from 4 to 5 GeV; see [EXHAD.md](../../docs/EXHAD.md).

## Runtime mapping

For a requested proper decay length $c\tau$, EventCalc evaluates

```text
coupling_squared = ctau_at_gY_1 / requested_ctau = g_Y^2
N_LLP_tot = N_POT * yield_per_POT_per_gY2 * g_Y^2
```

The generated `coupling_squared` column and the `Squared coupling` field in
event-file headers retain their generic names for compatibility, but both
mean $g_Y^2$ for `ALP-fermion`. Each generated ALP event or total file also
has a sibling `<output>.coupling.json` sidecar containing this complete
coupling and table-normalization contract.


## Input revision: 14–15 October 2026

The lifetime and branching tables use the regenerated fermion-universal
benchmark at the matching scale Lambda = 1 TeV. The exclusive-to-partonic
handoff is at 1.910 GeV. Unmasked partial widths are interpolated before that
handoff is applied; their sum defines the total width. The grid includes the
one-sided boundary node and the 1.910–1.915 GeV nodes. The physical matching
step is retained. The existing 32-channel roster and its threshold corrections
are retained, with branching fractions normalized over the listed channels.
The accompanying production yield, angular-energy distribution and maximum
energy tables come from the same regenerated export.

The squared matrix elements use the latest source export: the BaBar term is
removed at amplitude level before squaring, the approved KLKSpi0 phase-space
expression is supplied by the exporter, and omega-pion masses are consistent
with the runtime values (0.78265, 0.13957, 0.13957 GeV). The matrix-element-only
update did not modify the regenerated lifetime or branching fractions.
`input_revision.json` records hashes of the installed input files.

## Known bugs/problems

The ALP-fermion squared matrix elements exported from the ALP phenomenology
inputs (15 October 2026; [arXiv:2501.04525](https://arxiv.org/abs/2501.04525))
can be negative in physical phase space for `piPpiMpi0`, `3pi0` and `KPKMpi0`.
These three entries use `abs(source_squared_matrix_element)` as a sampling
workaround. The underlying source expressions still require correction;
the tabulated lifetimes and branching fractions are unchanged.
