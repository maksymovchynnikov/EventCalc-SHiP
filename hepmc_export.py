"""Optional HepMC3 export for EventCalc runtime events."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any


def _load_pyhepmc() -> Any:
    try:
        return importlib.import_module("pyhepmc")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "HepMC3 export requires the optional pyhepmc package; "
            "install it with `python3 -m pip install pyhepmc`"
        ) from exc


#: The code EventCalc fills the mother's PDG column with while it generates.
#: It is a marker, not a particle, and it must never reach a written event.
EVENTCALC_MOTHER_PLACEHOLDER = 12345678


class HepMC3Writer:
    """Stream :class:`runtime_generator.EventRecord` objects to HepMC3 ASCII.

    ``mother_pdg`` is the decaying particle's real PDG code, which the caller
    takes from ``llp.LLP_pdg``.  A HepMC reader keys on that code, so the
    generator's internal placeholder is not written: when no code is available
    and the record still carries the placeholder, the export stops and says so.
    """

    def __init__(self, path: Path, *, precision: int = 16,
                 mother_pdg: int | None = None) -> None:
        self.path = Path(path)
        self.mother_pdg = None if mother_pdg is None else int(mother_pdg)
        self._pyhepmc = _load_pyhepmc()
        self._run_info = self._pyhepmc.GenRunInfo()
        self._run_info.weight_names = ["decay_weight"]
        self._run_info.tools = [
            (
                "EventCalc-SHiP",
                "runtime",
                "Table-driven LLP event generator",
            )
        ]
        self._stream = self._pyhepmc.open(
            self.path, "w", precision=int(precision)
        )
        self.events_written = 0
        self._closed = False

    def __enter__(self) -> "HepMC3Writer":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._stream.close()
            self._closed = True

    @staticmethod
    def _decay_time_mm(mother: Any) -> float:
        momentum = math.sqrt(
            float(mother[0]) ** 2
            + float(mother[1]) ** 2
            + float(mother[2]) ** 2
        )
        distance_m = math.sqrt(
            float(mother[7]) ** 2
            + float(mother[8]) ** 2
            + float(mother[9]) ** 2
        )
        if momentum <= 0.0:
            return 0.0
        return 1000.0 * distance_m * float(mother[3]) / momentum

    def _mother_pdg_code(self, mother: Any) -> int:
        if self.mother_pdg is not None:
            return self.mother_pdg
        recorded = int(round(float(mother[5])))
        if recorded != EVENTCALC_MOTHER_PLACEHOLDER:
            return recorded
        raise ValueError(
            "the mother's PDG code is EventCalc's internal placeholder "
            f"{EVENTCALC_MOTHER_PLACEHOLDER}; pass mother_pdg (the driver "
            "takes it from llp.LLP_pdg, and --llp-pdg sets it explicitly)"
        )

    def event_from_record(self, record: Any) -> Any:
        """Convert one runtime event without writing it."""
        hepmc = self._pyhepmc
        mother = record.mother
        if len(mother) != 10:
            raise ValueError("the EventCalc mother record must have 10 entries")

        event = hepmc.GenEvent(
            self._run_info, hepmc.Units.GEV, hepmc.Units.MM
        )
        event.event_number = int(record.index)
        event.weights = [float(record.decay_weight)]
        event.attributes["eventcalc_inside_fiducial"] = bool(
            record.inside_volume
        )
        event.attributes["eventcalc_decay_channel"] = str(record.channel)
        event.attributes["eventcalc_status"] = str(record.status)

        vertex = hepmc.GenVertex(
            hepmc.FourVector(
                1000.0 * float(mother[7]),
                1000.0 * float(mother[8]),
                1000.0 * float(mother[9]),
                self._decay_time_mm(mother),
            )
        )
        parent = hepmc.GenParticle(
            hepmc.FourVector(*map(float, mother[:4])),
            self._mother_pdg_code(mother),
            2,
        )
        parent.generated_mass = float(mother[4])
        vertex.add_particle_in(parent)

        for daughter in record.daughters:
            if len(daughter) != 6:
                raise ValueError(
                    "each EventCalc daughter record must have 6 entries"
                )
            particle = hepmc.GenParticle(
                hepmc.FourVector(*map(float, daughter[:4])),
                int(round(float(daughter[5]))),
                1,
            )
            particle.generated_mass = float(daughter[4])
            vertex.add_particle_out(particle)

        event.add_vertex(vertex)
        return event

    def write(self, record: Any) -> None:
        if self._closed:
            raise RuntimeError("cannot write to a closed HepMC3 stream")
        self._stream.write(self.event_from_record(record))
        self.events_written += 1

    def write_batch(self, batch: Any) -> None:
        for position in range(len(batch)):
            self.write(batch.event(position))
