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

## Updated commands and imports

| Previous command or import | Current command or import |
| --- | --- |
| `python runtime_generator.py` | `python -m eventcalc.runtime_generator` |
| `python weighted_runtime.py` | `python -m eventcalc.weighted_runtime` |
| `python model_sensitivity.py` | `python -m eventcalc.model_sensitivity` |
| `python scan_sensitivity.py` | `python -m eventcalc.scan_sensitivity` |
| `python events_analysis.py` | `python -m eventcalc.analysis.events_analysis` |
| `python events_pointing_analysis.py` | `python -m eventcalc.analysis.events_pointing_analysis` |
| `python event_display.py` | `python -m eventcalc.analysis.event_display` |
| `python analyze_cascade_vs_primary.py` | `python -m eventcalc.analysis.analyze_cascade_vs_primary` |
| `python plot_event_distributions.py` | `python -m eventcalc.analysis.plot_event_distributions` |
| `python total-plots.py` | `python -m eventcalc.analysis.total_plots` |
| `python LLP-two-body-decay.py` | `python examples/llp_two_body_decay.py` |
| `from runtime_generator import RuntimeModelScan` | `from eventcalc.runtime_generator import RuntimeModelScan` |
| `from sensitivity import DecayBatch` | `from eventcalc.sensitivity import DecayBatch` |
| `from production_model import ProductionModel` | `from eventcalc.production.dark_photon import ProductionModel` |

The default compact distributions live in
`Distributions/Dark-photons/production/` and
`Distributions/Inelastic-DM/production/`. The full-data manifest lives beside
the dark-photon provider as `dataset.json`; the large dataset remains outside
the checkout. Fetch it with:

```sh
python -m eventcalc.production.dark_photon.datasets fetch --ssh-user YOUR_CERN_USERNAME
```

This directory change does not change the tabulated physics, event-weight
normalization, target description, decay models or uncertainty prescription.
