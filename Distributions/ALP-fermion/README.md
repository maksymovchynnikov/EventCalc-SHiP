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
$1.911\leq m_a\leq5.00$ GeV. At each mass, EventCalc combines the selected
`Jets-GG`, `Jets-ss`, `Jets-cc`, `ppbar`, and `nnbar` branching fractions
and requests that number of events once from the pseudoscalar event measure.
The generator therefore samples the aggregate hadronic composition rather
than treating these five table rows as separate fragmentation sources.
The `ePeM`, `muPmuM`, `tauPtauM`, and `2gamma` rows remain native EventCalc
decays. The unresolved component changes continuously to source-conditioned
default fragmentation from 4 to 5 GeV; see [EXHAD.md](../../EXHAD.md).

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
