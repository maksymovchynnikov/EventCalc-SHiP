"""Experiment responses prepare proper-flight-distance acceptance intervals."""
from typing import Protocol

class ExperimentResponse(Protocol):
    selections: tuple
    geometry: object

    def prepare(self, batch):
        """Return PreparedAcceptance without exposure or lifetime factors."""
        ...
