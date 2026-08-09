"""The two physical hypotheses and their source composition, defined once."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alp_discrimination.paths import REPOSITORY_ROOT


@dataclass(frozen=True)
class ProductionSource:
    identifier: str
    eventcalc_mode: str | None


@dataclass(frozen=True)
class ModelDefinition:
    identifier: str
    eventcalc_name: str
    legacy_name: str
    plot_label: str
    distribution_directory: Path
    sources: tuple[ProductionSource, ...]

    @property
    def particle_selection(self) -> dict[str, str]:
        return {"LLP_name": self.eventcalc_name, "particle_path": str(self.distribution_directory)}

    def input_paths(self, source: ProductionSource) -> tuple[Path, ...]:
        """Files that determine production, lifetime and visible decays."""
        directory = self.distribution_directory
        if self.eventcalc_name == "ALP-photon":
            suffix = "cascades" if source.identifier == "cascade" else source.identifier
            names = (
                f"DoubleDistr-ALP-photon_{suffix}.txt", f"Emax-ALP-photon_{suffix}.txt",
                f"Total-yield-ALP-photon_{suffix}.txt", "ctau-ALP-photon.txt",
                "ALP-photon-decay.json",
            )
        else:
            names = (
                "DoubleDistr-ALP-SU2L.txt", "Emax-ALP-SU2L.txt",
                "Total-yield-ALP-SU2L.txt", "ctau-ALP-SU2L.txt", "ALP-SU2L-decay.json",
            )
        code = tuple(REPOSITORY_ROOT / "funcs" / name for name in (
            "initLLP.py", "kinematics.py", "ship_setup.py", "interpolation_functions.py"
        ))
        return tuple(directory / name for name in names) + code


ALP_PHOTON_COMBINED = ModelDefinition(
    identifier="alp_photon_combined", eventcalc_name="ALP-photon",
    legacy_name="ALP-photon-combined", plot_label="ALP-photon, primary + cascade",
    distribution_directory=REPOSITORY_ROOT / "Distributions" / "ALP-photon",
    sources=(ProductionSource("primary", "primary"), ProductionSource("cascade", "cascade")),
)

ALP_SU2L = ModelDefinition(
    identifier="alp_su2l", eventcalc_name="ALP-SU2L", legacy_name="ALP-SU2L",
    plot_label=r"ALP-$SU(2)_L$",
    distribution_directory=REPOSITORY_ROOT / "Distributions" / "ALP-SU2L",
    # The inclusive B spectrum already contains cascade production implicitly.
    sources=(ProductionSource("inclusive", None),),
)

MODELS = (ALP_PHOTON_COMBINED, ALP_SU2L)
MODEL_BY_ID = {model.identifier: model for model in MODELS}
MODEL_BY_LEGACY_NAME = {model.legacy_name: model for model in MODELS}


def get_model(identifier: str) -> ModelDefinition:
    try:
        return MODEL_BY_ID[identifier]
    except KeyError as error:
        raise ValueError(f"Unknown model {identifier!r}; choose from {sorted(MODEL_BY_ID)}") from error
