"""Frozen NA62 lepton response and ideal-response hadronic p>1 GeV projection.

No tracking/PID/trigger/veto efficiencies, photon merging or mass-window model.
The hadronic vertex discretization is a numerical approximation.
"""
from dataclasses import dataclass
import numpy as np
from eventcalc.sensitivity import PreparedAcceptance, Selection
from . import _na62_leptons as leptons
from . import _na62_hadrons as hadrons
from ._na62_primitives import ElectronFV
from ._na62_cutflow import evaluate as cutflow

def classify(ids):
    """Disjoint terminal-state diagnostics, not primary resonance labels."""
    result=[]
    for row in ids:
        s=set(np.abs(row));s.discard(0)
        if s and s<={211}:g=0
        elif 211 in s and s<={211,22}:g=1
        elif 321 in s:g=2
        elif s & {130,310}:g=3
        elif s & {2112,2212}:g=4
        else:g=5
        result.append(g)
    return np.asarray(result,dtype=np.int64)



@dataclass(frozen=True)
class NA62Config:
    vertex_cells: int = 128
    hadron_p_min_GeV: float = 1.
    lepton_p_min_GeV: float = 10.

    def __post_init__(self):
        if isinstance(self.vertex_cells, bool) or not isinstance(self.vertex_cells, int) or self.vertex_cells < 2:
            raise ValueError('At least two vertex cells required')
        if self.hadron_p_min_GeV != 1. or self.lepton_p_min_GeV != 10.:
            raise ValueError('This frozen response supports hadron p>1 and lepton p>10 GeV only')


class NA62Response:
    def __init__(self, config=None):
        self.config = config or NA62Config()
        self.geometry = None  # geometry belongs to this response, not the lifetime engine
        self.selections = tuple(Selection(n, min_particles=0, energy_min=0.) for n in
            (*leptons.NAMES, *(f'hadronic_group_{i}' for i in range(6))))

    def prepare(self, batch):
        arrays, mass = batch.arrays()
        ids = np.asarray(batch.pdg_ids)
        charges = np.asarray(batch.charges, float)
        if ids.shape != arrays[2].shape or not np.issubdtype(ids.dtype, np.integer):
            raise ValueError('NA62 requires integer daughter pdg_ids matching the visibility mask')
        if charges.shape != ids.shape or not np.all(np.isfinite(charges)):
            raise ValueError('NA62 requires finite daughter charges')
        a = leptons.prepare_na62(batch, ids)
        b = hadrons.prepare(batch, ids, classify(ids), nz=self.config.vertex_cells)
        return PreparedAcceptance(np.r_[a.lower,b.lower], np.r_[a.upper,b.upper],
            np.r_[a.weights,b.weights], np.r_[a.selected,b.selected+len(a.selections)],
            self.selections, self.geometry, event_index=np.r_[a.event_index,b.event_index])

    def cutflow(self, batch, ctaus):
        self.prepare(batch)  # validate the same complete-event contract
        return cutflow(batch,batch.pdg_ids,classify(batch.pdg_ids),ctaus,
                       pmins=(self.config.hadron_p_min_GeV,),nz=self.config.vertex_cells)
