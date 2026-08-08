"""Plot FORESEE polygons with manually editable experiment-label positions."""

from argparse import ArgumentParser
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from alp_discrimination.config import PROFILES
from alp_discrimination.constraints.convert import COUPLING_CONVERSION_FACTOR
from alp_discrimination.paths import profile_output_dir
from alp_discrimination.plot_style import style_axis, use_report_style

LABEL_CONFIG_PATH = Path(__file__).with_name("label_positions.json")
LABEL_CONTEXTS = ("constraint_only", "event_density_overlay")

PHOTON_SPECS = (
    ("bounds_BESIII_2024.txt", "BESIII"), ("bounds_BESIII_2022.txt", "BESIII"),
    ("bounds_Belle2.txt", "Belle II"), ("bounds_PrimEx.txt", "PrimEx"),
    ("bounds_LEP.txt", "LEP"), ("bounds_SN1987.txt", "SN1987"),
    ("bounds_E137.txt", "E137"), ("bounds_NuCal.txt", "NuCal"),
    ("bounds_CHARM.txt", "CHARM"), ("bounds_NA64.txt", "NA64"),
    ("bounds_E141.txt", "E141"),
)
SU2_SPECS = (
    ("bounds_BaBar.txt", "BaBar"), ("bounds_SN1987.txt", "SN1987"),
    ("bounds_E137.txt", "E137"), ("bounds_LEP.txt", "LEP"),
    ("bounds_E949_displ.txt", "E949"), ("bounds_NA62_1.txt", "NA62"),
    ("bounds_NA62_2.txt", "NA62"), ("bounds_KOTO.txt", "KOTO"),
    ("bounds_KTEV.txt", "KTEV"), ("bounds_NA6264.txt", "+ NA48/2"),
    ("bounds_E949_prompt.txt", "E949"), ("bounds_CDF.txt", "CDF"),
)
MODEL_SPECS = {"alp_photon": PHOTON_SPECS, "alp_su2l": SU2_SPECS}


@dataclass(frozen=True)
class LabelPlacement:
    text: str
    x: float
    y: float
    rotation: float
    horizontal_alignment: str
    vertical_alignment: str
    fontsize: float
    color: str
    visible: bool


LabelConfig = dict[str, dict[str, dict[str, LabelPlacement]]]


def _mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object")
    return value


def _number(value: Any, location: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{location} must be a finite number")
    if positive and value <= 0.0:
        raise ValueError(f"{location} must be positive")
    return float(value)


def _placement(defaults: dict[str, Any], raw: Any, text: str, location: str) -> LabelPlacement:
    raw = _mapping(raw, location)
    allowed = {
        "x", "y", "coordinate_system", "rotation", "horizontal_alignment",
        "vertical_alignment", "fontsize", "color", "visible",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"unknown {location} fields: {sorted(unknown)}")
    values = defaults | raw
    if values.get("coordinate_system") != "axes":
        raise ValueError(f"{location}.coordinate_system must be 'axes'")
    horizontal, vertical = values.get("horizontal_alignment"), values.get("vertical_alignment")
    if horizontal not in {"left", "center", "right"}:
        raise ValueError(f"invalid {location}.horizontal_alignment")
    if vertical not in {"bottom", "baseline", "center", "center_baseline", "top"}:
        raise ValueError(f"invalid {location}.vertical_alignment")
    color, visible = values.get("color"), values.get("visible")
    if not isinstance(color, str) or not color:
        raise ValueError(f"{location}.color must be a non-empty string")
    if not isinstance(visible, bool):
        raise ValueError(f"{location}.visible must be boolean")
    return LabelPlacement(
        text=text, x=_number(values.get("x"), f"{location}.x"),
        y=_number(values.get("y"), f"{location}.y"),
        rotation=_number(values.get("rotation"), f"{location}.rotation"),
        horizontal_alignment=horizontal, vertical_alignment=vertical,
        fontsize=_number(values.get("fontsize"), f"{location}.fontsize", positive=True),
        color=color, visible=visible,
    )


def load_label_config(path: Path = LABEL_CONFIG_PATH) -> LabelConfig:
    """Load and fully validate axes-fraction positions for every plotted label."""
    try:
        root = _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not load label configuration {path}: {error}") from error
    if root.get("schema_version") != 1:
        raise ValueError("unsupported label configuration schema_version")
    defaults = _mapping(root.get("defaults"), "defaults")
    models = _mapping(root.get("models"), "models")
    if set(models) != set(MODEL_SPECS):
        raise ValueError(f"models must be exactly {sorted(MODEL_SPECS)}")

    result: LabelConfig = {}
    for model, specs in MODEL_SPECS.items():
        entries = _mapping(models[model], f"models.{model}")
        expected = {filename: label for filename, label in specs}
        if set(entries) != set(expected):
            raise ValueError(f"models.{model} must cover exactly {sorted(expected)}")
        result[model] = {}
        for filename, label in expected.items():
            entry = _mapping(entries[filename], f"models.{model}.{filename}")
            if set(entry) != {"text", "positions"} or entry["text"] != label:
                raise ValueError(f"invalid label entry models.{model}.{filename}")
            positions = _mapping(entry["positions"], f"models.{model}.{filename}.positions")
            if set(positions) != set(LABEL_CONTEXTS):
                raise ValueError(f"{model}.{filename} must define {list(LABEL_CONTEXTS)}")
            result[model][filename] = {
                context: _placement(defaults, positions[context], label, f"{model}.{filename}.{context}")
                for context in LABEL_CONTEXTS
            }
    return result


def load_constraint(path: Path) -> np.ndarray:
    data = np.loadtxt(path, comments="#", ndmin=2)
    invalid = (data.shape[1] < 2 or not np.all(np.isfinite(data[:, :2]))
               or np.any(data[:, 0] < 0.0) or np.any(data[:, 1] <= 0.0))
    if invalid:
        raise ValueError(f"invalid constraint polygon {path}")
    return data


def draw_constraints(
    axis: plt.Axes, directory: Path, specs, *, model: str, context: str,
    config: LabelConfig | None = None,
) -> None:
    if model not in MODEL_SPECS:
        raise ValueError(f"unknown constraint model {model!r}")
    if context not in LABEL_CONTEXTS:
        raise ValueError(f"unknown label context {context!r}")
    config = load_label_config() if config is None else config
    for index, (filename, label) in enumerate(specs):
        data = load_constraint(directory / filename)
        axis.fill(data[:, 0], data[:, 1], color="gainsboro", zorder=-100 + index)
        axis.plot(data[:, 0], data[:, 1], color="dimgray", linewidth=1.0, zorder=-100 + index)
        try:
            placement = config[model][filename][context]
        except KeyError as error:
            raise ValueError(f"missing label configuration for {model}/{filename}/{context}") from error
        if placement.text != label:
            raise ValueError(f"label mismatch for {model}/{filename}")
        if placement.visible:
            axis.text(
                placement.x, placement.y, placement.text, transform=axis.transAxes,
                rotation=placement.rotation, ha=placement.horizontal_alignment,
                va=placement.vertical_alignment, fontsize=placement.fontsize,
                color=placement.color, clip_on=True, zorder=100,
            )


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("alp_photon", "alp_su2l"))
    parser.add_argument("--profile", choices=sorted(PROFILES), default="production")
    parser.add_argument("--label-config", type=Path, default=LABEL_CONFIG_PATH)
    args = parser.parse_args()
    use_report_style()
    output_dir = profile_output_dir(args.profile, "constraints")
    if args.model == "alp_photon":
        input_dir = output_dir / "raw/alp_photon"
        specs, limits = PHOTON_SPECS, ((1e-2, 1.0), (3e-7, 2e-2))
        ylabel = r"$g_{a\gamma\gamma}$ [GeV$^{-1}$]"
    else:
        input_dir = output_dir / "converted/alp_su2l"
        specs = SU2_SPECS
        limits = ((5e-2, 1.0), tuple(value * COUPLING_CONVERSION_FACTOR for value in (7e-7, 1e-3)))
        ylabel = r"$c_W/f_a$ [GeV$^{-1}$]"
    figure, axis = plt.subplots(figsize=(8.0, 6.2))
    draw_constraints(
        axis, input_dir, specs, model=args.model, context="constraint_only",
        config=load_label_config(args.label_config),
    )
    axis.set(xscale="log", yscale="log", xlim=limits[0], ylim=limits[1],
             xlabel=r"$m_a$ [GeV]", ylabel=ylabel)
    axis.tick_params(which="both", direction="in", top=True, right=True)
    axis.grid(False)
    style_axis(axis)
    figure.tight_layout()
    path = output_dir / f"{args.model}_constraints.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
