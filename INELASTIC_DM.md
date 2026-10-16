# Inelastic dark matter

Select `Inelastic-DM` in the interactive `simulate.py` menu, or use:

```sh
python simulate.py --model Inelastic-DM --masses 20 --idm-delta .05 \
  --ctaus 0.1 1000 --events 10000 --seed 1
python run_batch.py --llp Inelastic-DM --masses 20 --idm-delta .05 \
  --ctaus 0.1 1000 --nevents 10000 --exhad-seed 1
```

Both commands accept **mediator masses**, and **chi-prime proper decay lengths**.
At the example point, m_chi=20/3 GeV and m_chi_prime=7 GeV. The source fixes
m_Aprime/m_chi=3 and alpha_D=0.1; `idm_delta` is the splitting in a launch card.
`--couplings-squared` supplies epsilon squared instead of a lifetime.
`--dp-production primary`, `cascade` or `combined` selects the source;
combined is the default. Old flux-table modes and `--exhad off` are rejected.

The inclusive menu channel `chi + SM` samples all allowed electromagnetic
final states according to their branching fractions. Individual exclusive
channels are labelled in the output. Channel-restricted inelastic generation
is not offered by this launcher. A configured exHad release containing
`inelastic_dark_matter.py` is required; see the release checklist in README.

`--production-power` controls the integration sample (default 15); output
`events` is independent. Production momenta and collision origins remain paired,
including finite-width mediator production and its polarization. All lifetimes
at one mass reuse the same folded source. Decays are sampled along each ray's
exact intersection with the volume, measured from its collision position.
The stable chi daughter has PDG 4900101 and chi-prime has PDG 4900102.
Production metadata and row-aligned collision origins accompany the event files.


`funcs/inelasticDM.py` is a compatibility interface to
`exhad.inelastic_dark_matter`. Decay widths, invariant-mass sampling and full
final states come from that single implementation. Set `EXHAD_ROOT` to a
configured current exHad release (or use an installed package).

```python
from funcs import inelasticDM
result = inelasticDM.generate(20., 1024, delta=.05, ratio=3., alpha_D=.1, seed=1)
daughters = result['particles']  # [event, particle, px/py/pz/E/m/PDG]
visible = inelasticDM.visibility(daughters)
ctau_m = result['ctau_times_eps2'] / epsilon_squared
```

The energy release must satisfy `2*m_e < m_chi_prime-m_chi < 1.7 GeV`, and
the mediator must be off shell in the excited-state decay. This covers the
manuscript's mass scans through 40 GeV at mA/mChi=3 and Delta=.1 or .05.
Leptons, two pions, three pions and radiative final states use the implemented
vector-current angular laws; higher multiplicities retain the documented
phase-space approximation. Invisible daughters remain in each event for
four-momentum conservation. The stable state has PDG 4900101; padding is -999.

For calibrated primary and cascade production, use `ProductionModel` in
`production_model`. It includes the finite mediator width and polarization,
with mA/mChi=3 and alpha_D=.1. The `produce` compatibility helper instead
describes an on-shell mediator supplied by the caller. It is not a substitute
for integrating the virtual pair mass near the production threshold.

Run a complete one-mass scan without writing HepMC:

```sh
python model_sensitivity.py --model idm --mass 20 --delta .05 --output /tmp/idm20
```

The two outputs require at least two visible particles with E>1 GeV at ECAL
and IP<=2.5 m, or all visible particles with E>1 GeV at ECAL. Charged particles
include magnetic deflection. `SENSITIVITY.md` describes the generic interface,
normalization, performance and explicit access to older production tables.
