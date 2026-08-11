## SU(2)L ALP

This builder generates EventCalc input tables for the benchmark

$$
\mathcal L \supset \frac{\alpha_2}{4\pi}g_W\,a\,W\widetilde W,
\qquad g_W=\frac{c_W}{f_a},
$$

with $c_B=c_{a\Phi}=0$. The FCNC and diphoton expressions from
arXiv:1901.02031 are converted to this project coupling convention.

The production tables calculate $B\to K a$ and the CKM-suppressed
$B\to\pi a$ channel directly. Additional strange-resonance channels use the
relative rates in `Br-ratios-scalar.csv`, derived from the scalar study used in
this project (arXiv:1904.10447). The decay table retains only
$a\to\gamma\gamma$, with $\mathrm{Br}(a\to\gamma\gamma)=1$ by construction.

### Generate the tables

The B-meson momentum sample is not stored in Git. Place it at

```text
table_builders/ALP_SU2L/exp1.txt
```

and run from the repository root:

```bash
python -m table_builders.ALP_SU2L.build_tables
```

The first run creates an on-shell B-momentum cache. Its metadata records the
input file and B-meson mass, so the cache is rebuilt if either changes.

Generated EventCalc inputs:

- `DoubleDistr-ALP-SU2L.txt`
- `Emax-ALP-SU2L.txt`
- `Total-yield-ALP-SU2L.txt`
- `ctau-ALP-SU2L.txt`
- `ALP-SU2L-decay.json`

The resonance rescaling and diphoton-only decay model are approximations of this
benchmark rather than a complete GeV-scale ALP treatment.
