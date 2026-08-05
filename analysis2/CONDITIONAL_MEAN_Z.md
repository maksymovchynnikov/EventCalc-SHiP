# Conditional mean-z Week-8 workflow

This package-native workflow implements the validated score

\[
\log L = \log L_E + \log L_{\bar z\mid E},
\]

with lifetime independently profiled under the photon and SU(2)_L hypotheses.

For one existing template bank, run the stages in this order:

```bash
python -m analysis2.workflows.conditional_mean_z_rangefinder \
  --bank-path <template_bank.npz> \
  --output-dir <rangefinder-output> \
  --workers 2

python -m analysis2.workflows.conditional_mean_z_full_domain \
  --rangefinder-summary <rangefinder_summary.json> \
  --output-dir <full-2k-output> \
  --workers 2

python -m analysis2.workflows.conditional_mean_z_selected \
  --full-domain-summary <full_domain_2k_summary.json> \
  --output-dir <selected-5k-output> \
  --workers 2

python -m analysis2.workflows.conditional_mean_z_decision_audit \
  --full-domain-summary <full_domain_2k_summary.json> \
  --selected-5k-dir <selected-5k-output> \
  --output-dir <decision-audit-output> \
  --workers 2

python -m analysis2.workflows.conditional_mean_z_uniform \
  --full-domain-summary <full_domain_2k_summary.json> \
  --decision-audit-dir <decision-audit-output> \
  --output-dir <uniform-10k-output> \
  --workers 2
```

The workflows are generic in mass, detector selection, lifetime-grid size and
disconnected lifetime intervals. They reuse existing EventCalc proposal caches.

The next architecture layer is the multi-mass controller and mean-z-aware
lifetime-bank refinement. This integration first freezes the validated
statistical implementation before changing bank construction.
