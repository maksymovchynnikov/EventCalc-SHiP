# Repository layout

Commands below run from the repository root. `simulate.py` and `run_batch.py`
remain the public simulation entry points. Other commands run as modules so
imports and data paths resolve consistently.

| Location | Contents |
| --- | --- |
| `eventcalc/` | Python event generator, lifetime integration, sensitivity scans and HepMC export |
| `eventcalc/production/dark_photon/` | Calibrated dark-photon production and mediator decay to excited dark matter |
| `eventcalc/analysis/` | Event displays, distributions, pointing and cascade comparisons |
| `funcs/` | Shared model/decay helpers, input validation, detector geometry and event I/O |
| `Distributions/` | Compact production and decay inputs, grouped by physical model |
| `cards/` | Model, geometry and sensitivity cards |
| `examples/` | Short Python examples |
| `tests/` | Unit tests and numerical validation scripts |
| `docs/` | Detailed physics, usage and performance guides |

## Commands and Python interfaces

| Task | Command or import |
| --- | --- |
| Interactive or card-based simulation | `python simulate.py` |
| Mass and lifetime grid | `python run_batch.py` |
| Generate event batches | `python -m eventcalc.runtime_generator` |
| Weighted ALP decays | `python -m eventcalc.weighted_runtime` |
| Automatic production bands | `python -m eventcalc.model_sensitivity` |
| General model sensitivity | `python -m eventcalc.scan_sensitivity` |
| Distributions and acceptance | `python -m eventcalc.analysis.events_analysis` |
| Daughter pointing | `python -m eventcalc.analysis.events_pointing_analysis` |
| Event display | `python -m eventcalc.analysis.event_display` |
| Cascade/primary comparison | `python -m eventcalc.analysis.analyze_cascade_vs_primary` |
| Parent kinematic distributions | `python -m eventcalc.analysis.plot_event_distributions` |
| Scan plots | `python -m eventcalc.analysis.total_plots` |
| Two-body decay example | `python examples/llp_two_body_decay.py` |
| Scan API | `from eventcalc.runtime_generator import RuntimeModelScan` |
| Acceptance API | `from eventcalc.sensitivity import DecayBatch` |
| Full production API | `from eventcalc.production.dark_photon import ProductionModel` |

The default compact distributions live in
`Distributions/Dark-photons/production/` and
`Distributions/Inelastic-DM/production/`. The full-data manifest lives beside
the dark-photon provider as `dataset.json`; the large dataset remains outside
the checkout. Fetch it with:

```sh
python -m eventcalc.production.dark_photon.datasets fetch --ssh-user YOUR_CERN_USERNAME
```
