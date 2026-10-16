# Inelastic dark matter

`Inelastic-DM` describes stable dark matter $\chi$ and a heavier fermion
$\chi'$ coupled off-diagonally to a dark photon $A'$. In addition to its
kinetic-mixing interaction with ordinary matter, the mediator has

$$
\mathcal L\supset i g_D A'_\mu\bar\chi'\gamma^\mu\chi,
\qquad
\alpha_D=\frac{g_D^2}{4\pi},\qquad
\Delta=\frac{m_{\chi'}-m_\chi}{m_\chi}.
$$

The factor $i$ is the convention for the two Majorana fields.
Production gives $\chi\chi'$; the excited state subsequently decays through
an off-shell mediator into $\chi$ and Standard Model particles. The
phenomenology follows the vector-mediated model discussed in
[arXiv:1810.01879](https://arxiv.org/abs/1810.01879).
The standard calculation fixes $m_{A'}/m_\chi=3$ and $\alpha_D=0.1$.

## Inputs and generation

Install [exHad](EXHAD.md) with its `exhad.inelastic_dark_matter` module, then run:

```sh
python simulate.py --model Inelastic-DM --masses 20 --idm-delta .05 \
  --ctaus 0.1 1000 --events 10000 --seed 1
python run_batch.py --llp Inelastic-DM --masses 20 --idm-delta .05 \
  --ctaus 0.1 1000 --nevents 10000 --exhad-seed 1
```

Input masses denote $m_{A'}$ in GeV; proper decay lengths refer to $\chi'$ in
metres. The example has $m_\chi=20/3$ GeV and $m_{\chi'}=7$ GeV.
`--couplings-squared` can specify $\epsilon^2$; at fixed masses, splitting and
dark coupling, the lifetime is

$$
c\tau_{\chi'}=\frac{(c\tau_{\chi'}\epsilon^2)}{\epsilon^2}.
$$

The numerator is computed from the inclusive excited-state width.
The [production guide](PRODUCTION_DATA.md) defines primary and cascade sources,
compact-table coverage and the optional calculation with collision positions.
Both reuse production when lifetimes change.

The inclusive channel `chi + SM` samples all open electromagnetic final states
with their branching fractions. This launcher requires the inclusive selection
and exHad. Its implemented decay range is
$2m_e<\Delta m_\chi<1.7$ GeV, with $m_e$ the electron mass and
$\Delta m_\chi<m_{A'}$. Leptons, two pions, three pions and radiative final
states use vector-current angular distributions; higher multiplicities use
phase space. Invisible daughters stay in the event record to conserve
four-momentum. Particle identifiers are 4900101 for $\chi$ and 4900102 for
$\chi'$; unused particle slots have identifier -999.

## Python interface

Widths and complete decay chains are computed by `exhad.inelastic_dark_matter`.
The EventCalc interface returns daughters with components
`[px, py, pz, E, mass, PDG]`:

```python
from funcs import inelasticDM

epsilon_squared = 1e-8
result = inelasticDM.generate(20., 1024, delta=.05, ratio=3., alpha_D=.1, seed=1)
daughters = result['particles']
visible = inelasticDM.visibility(daughters)
ctau_m = result['ctau_times_eps2'] / epsilon_squared
```

For production, `ProductionModel` in `eventcalc.production.dark_photon`
integrates the virtual pair mass and mediator polarization. The
`inelasticDM.produce` helper describes decay of an on-shell mediator whose
momenta are supplied by the caller.

For a complete uncertainty scan with the full EOS dataset:

```sh
python -m eventcalc.model_sensitivity --model idm --mass 20 --delta .05 --output /tmp/idm20
```

The [sensitivity guide](SENSITIVITY.md) defines the two selections, event-count
threshold and production bands on both mixing boundaries.
