"""Compatibility interface to the shared exHad inelastic-dark-matter current.

Use production_model.ProductionModel for the calibrated finite-width source,
and model_sensitivity.py for direct in-memory event-yield scans.
"""
import os
import numpy as np
from model_sensitivity import configure_exhad

_root=os.environ.get('EXHAD_ROOT') or os.environ.get('DP_EXHAD_ROOT')
configure_exhad(_root)
from exhad.inelastic_dark_matter import (
    ALPHA, HBARC, ME, MAX_CURRENT_MASS, catalog, table_data, lepton_width,
    current_widths, differential_width, spectrum, width, ctau, directions,
    boost, rotate_z, angular_weight, generate, produce,
)

def visibility(particles):
    """Exclude the stable dark state, neutrinos and padded rows."""
    ids=np.abs(np.asarray(particles)[...,5]).astype(int)
    return ~np.isin(ids,[12,14,16,4900101,999])
