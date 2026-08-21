"""Configuration and command-line parsing for the EventCalc launcher.

This module deliberately uses only the Python standard library.  In particular,
it can validate a launch card without importing the numerical simulation stack
or any module that prompts for user input.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


DEFAULT_N_POT = 6.0e20
DEFAULT_MIN_EVENTS_THRESHOLD = 0.1
SIN2_THETA_W = 0.23122
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigurationError(ValueError):
    """Raised when a non-interactive EventCalc launch is incomplete or invalid."""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    directory: str
    decay_file: str
    common_files: tuple[str, ...]


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "Scalar-mixing",
        "Scalar-mixing",
        "HLS-decay.json",
        (
            "DoubleDistr-Scalar-mixing.txt",
            "Emax-Scalar-mixing.txt",
            "Total-yield-Scalar-mixing.txt",
            "ctau-Scalar.txt",
        ),
    ),
    ModelSpec(
        "ALP-photon",
        "ALP-photon",
        "ALP-photon-decay.json",
        ("ctau-ALP-photon.txt",),
    ),
    ModelSpec(
        "Scalar-quartic",
        "Scalar-quartic",
        "HLS-decay.json",
        (
            "DoubleDistr-Scalar-quartic.txt",
            "Emax-Scalar-quartic.txt",
            "Total-yield-Scalar-quartic.txt",
            "ctau-Scalar.txt",
        ),
    ),
    ModelSpec(
        "Dark-photons",
        "Dark-photons",
        "DP-decay.json",
        ("ctau-DP.txt",),
    ),
    ModelSpec(
        "HNL",
        "HNL",
        "HNL-decay.json",
        (
            "DoubleDistr-HNL-mixing-e.txt",
            "DoubleDistr-HNL-mixing-mu.txt",
            "DoubleDistr-HNL-mixing-tau.txt",
            "Emax-HNL.txt",
            "Total-yield-HNL-e.txt",
            "Total-yield-HNL-mu.txt",
            "Total-yield-HNL-tau.txt",
            "HNLdecayWidth.dat",
        ),
    ),
    ModelSpec(
        "ALP-SU2L",
        "ALP-SU2L",
        "ALP-SU2L-decay.json",
        (
            "DoubleDistr-ALP-SU2L.txt",
            "Emax-ALP-SU2L.txt",
            "Total-yield-ALP-SU2L.txt",
            "ctau-ALP-SU2L.txt",
        ),
    ),
    ModelSpec(
        "ALP-mixed",
        "ALP-mixed",
        "ALP-mixed-decay.json",
        (),
    ),
)


def _name_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


_MODEL_BY_KEY = {_name_key(spec.name): spec for spec in MODEL_SPECS}
_MODEL_BY_KEY.update(
    {
        "alpsu2": _MODEL_BY_KEY["alpsu2l"],
        "alpmix": _MODEL_BY_KEY["alpmixed"],
        "darkphoton": _MODEL_BY_KEY["darkphotons"],
        "scalar": _MODEL_BY_KEY["scalarmixing"],
    }
)


def resolve_model(value: Any) -> ModelSpec:
    """Resolve a user-facing model name against a fixed, deterministic registry."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError("'model' must be a non-empty model name.")
    spec = _MODEL_BY_KEY.get(_name_key(value))
    if spec is None:
        choices = ", ".join(item.name for item in MODEL_SPECS)
        raise ConfigurationError(f"Unknown model {value!r}. Choose one of: {choices}.")
    return spec


def available_models(project_root: Path = PROJECT_ROOT) -> tuple[ModelSpec, ...]:
    """Return registered models whose distribution directories are installed."""
    distributions = Path(project_root) / "Distributions"
    return tuple(spec for spec in MODEL_SPECS if (distributions / spec.directory).is_dir())


def _finite_float(value: Any, field: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool):
        raise ConfigurationError(f"'{field}' must be a number, not a boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"'{field}' must be a number; received {value!r}.") from exc
    if not math.isfinite(result):
        raise ConfigurationError(f"'{field}' must be finite; received {value!r}.")
    if positive and result <= 0.0:
        raise ConfigurationError(f"'{field}' must be greater than zero.")
    if nonnegative and result < 0.0:
        raise ConfigurationError(f"'{field}' must be non-negative.")
    return result


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ConfigurationError(f"'{field}' must be a positive integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"'{field}' must be a positive integer.") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ConfigurationError(f"'{field}' must be a positive integer.")
    if result <= 0:
        raise ConfigurationError(f"'{field}' must be a positive integer.")
    return result


def _number_list(value: Any, field: str) -> tuple[float, ...]:
    raw = value if isinstance(value, (list, tuple)) else [value]
    if not raw:
        raise ConfigurationError(f"'{field}' must contain at least one value.")
    return tuple(_finite_float(item, field, positive=True) for item in raw)


def _lifetime_grid(value: Any, number_of_masses: int) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (list, tuple)):
        common = _number_list(value, "c_taus")
        return tuple(common for _ in range(number_of_masses))
    if not value:
        raise ConfigurationError("'c_taus' must contain at least one value.")

    nested = any(isinstance(item, (list, tuple)) for item in value)
    if not nested:
        common = _number_list(value, "c_taus")
        return tuple(common for _ in range(number_of_masses))
    if not all(isinstance(item, (list, tuple)) for item in value):
        raise ConfigurationError("'c_taus' cannot mix scalar values and per-mass lists.")
    if len(value) != number_of_masses:
        raise ConfigurationError(
            "Nested 'c_taus' must provide exactly one lifetime list per mass "
            f"({number_of_masses} expected, {len(value)} received)."
        )
    return tuple(_number_list(item, "c_taus") for item in value)


def _decay_selection(value: Any) -> tuple[str | int, ...]:
    if value is None:
        return ("all",)
    raw = [value] if isinstance(value, (str, int)) and not isinstance(value, bool) else value
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ConfigurationError("'decay_channels' must be 'all' or a non-empty list of names/indices.")
    result: list[str | int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise ConfigurationError("Decay channels must be names or one-based integer indices.")
        if isinstance(item, str):
            item = item.strip()
            if not item:
                raise ConfigurationError("Decay-channel names cannot be empty.")
        result.append(item)
    return tuple(result)


def _mixing_pattern(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ConfigurationError("'mixing_pattern' must contain xi_e, xi_mu, and xi_tau.")
    entries = tuple(_finite_float(item, "mixing_pattern", nonnegative=True) for item in value)
    total = sum(entries)
    if total <= 0.0:
        raise ConfigurationError("At least one HNL mixing component must be positive.")
    return tuple(item / total for item in entries)  # type: ignore[return-value]


def _strict_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigurationError(f"'{field}' must be true or false.")
    return value


_CARD_ALIASES = {
    "ctaus": "c_taus",
    "decays": "decay_channels",
    "mixing": "mixing_pattern",
    "alp_production": "alp_production_mode",
    "relative_sign": "interference",
    "sign": "interference",
    "plot_phenomenology": "plots",
}
_CONFIG_KEYS = {
    "model",
    "events",
    "masses",
    "c_taus",
    "decay_channels",
    "mixing_pattern",
    "uncertainty",
    "alp_production_mode",
    "xi",
    "interference",
    "plots",
    "export_events",
    "n_pot",
    "min_events_threshold",
    "seed",
}


def _canonical_card_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for raw_key, value in mapping.items():
        if not isinstance(raw_key, str):
            raise ConfigurationError("Every launch-card key must be a string.")
        key = _CARD_ALIASES.get(raw_key, raw_key)
        if key not in _CONFIG_KEYS:
            choices = ", ".join(sorted(_CONFIG_KEYS))
            raise ConfigurationError(f"Unknown launch-card field {raw_key!r}. Supported fields: {choices}.")
        if key in canonical:
            raise ConfigurationError(f"Launch card specifies {key!r} more than once.")
        canonical[key] = value
    return canonical


@dataclass(frozen=True)
class SimulationConfig:
    model: str
    events: int
    masses: tuple[float, ...]
    c_taus: tuple[tuple[float, ...], ...]
    decay_channels: tuple[str | int, ...]
    mixing_pattern: tuple[float, float, float] | None
    uncertainty: str | None
    alp_production_mode: str | None
    xi: float | None
    interference: str | None
    plots: bool
    export_events: bool
    n_pot: float
    min_events_threshold: float
    seed: int | None
    project_root: Path

    @property
    def n_events(self) -> int:
        """Number of raw interpolation samples used by the original workflow."""
        return self.events * 10

    @property
    def particle_path(self) -> Path:
        spec = resolve_model(self.model)
        return self.project_root / "Distributions" / spec.directory

    @property
    def particle_selection(self) -> dict[str, str]:
        return {"particle_path": str(self.particle_path), "LLP_name": self.model}

    def selected_decay_indices(self, available: Sequence[str] | None = None) -> list[int]:
        channels = tuple(available) if available is not None else load_decay_channel_names(self)
        return resolve_decay_channels(self.decay_channels, channels)

    def as_dict(self) -> dict[str, Any]:
        common_lifetimes = all(item == self.c_taus[0] for item in self.c_taus)
        lifetimes: list[Any]
        if common_lifetimes:
            lifetimes = list(self.c_taus[0])
        else:
            lifetimes = [list(item) for item in self.c_taus]
        return {
            "model": self.model,
            "events": self.events,
            "masses": list(self.masses),
            "c_taus": lifetimes,
            "decay_channels": list(self.decay_channels),
            "mixing_pattern": list(self.mixing_pattern) if self.mixing_pattern is not None else None,
            "uncertainty": self.uncertainty,
            "alp_production_mode": self.alp_production_mode,
            "xi": self.xi,
            "interference": self.interference,
            "plots": self.plots,
            "export_events": self.export_events,
            "n_pot": self.n_pot,
            "min_events_threshold": self.min_events_threshold,
            "seed": self.seed,
        }


def _required_files(spec: ModelSpec, uncertainty: str | None, alp_mode: str | None) -> tuple[str, ...]:
    files = list(spec.common_files)
    if spec.name == "ALP-photon":
        assert alp_mode is not None
        files.extend(
            (
                f"DoubleDistr-ALP-photon_{alp_mode}.txt",
                f"Emax-ALP-photon_{alp_mode}.txt",
                f"Total-yield-ALP-photon_{alp_mode}.txt",
            )
        )
    elif spec.name == "Dark-photons":
        assert uncertainty is not None
        files.extend(
            (
                f"DoubleDistr-DP-{uncertainty}.txt",
                f"Emax-DP-{uncertainty}.txt",
                f"Total-yield-DP-{uncertainty}.txt",
            )
        )
    files.append(spec.decay_file)
    return tuple(files)


def _validate_installed_files(config: SimulationConfig) -> None:
    spec = resolve_model(config.model)
    model_dir = config.particle_path
    if not model_dir.is_dir():
        raise ConfigurationError(f"Distribution directory does not exist: {model_dir}")
    missing = [name for name in _required_files(spec, config.uncertainty, config.alp_production_mode)
               if not (model_dir / name).is_file()]
    if missing:
        raise ConfigurationError(
            f"Model {config.model} is missing required distribution file(s): {', '.join(missing)}."
        )
    if config.model == "ALP-mixed":
        distributions = config.project_root / "Distributions"
        source_files = (
            distributions / "ALP-SU2L" / "DoubleDistr-ALP-SU2L.txt",
            distributions / "ALP-SU2L" / "Emax-ALP-SU2L.txt",
            distributions / "ALP-SU2L" / "Total-yield-ALP-SU2L.txt",
            distributions / "ALP-photon" / "DoubleDistr-ALP-photon_primary.txt",
            distributions / "ALP-photon" / "Emax-ALP-photon_primary.txt",
            distributions / "ALP-photon" / "Total-yield-ALP-photon_primary.txt",
            distributions / "ALP-photon" / "DoubleDistr-ALP-photon_cascades.txt",
            distributions / "ALP-photon" / "Emax-ALP-photon_cascades.txt",
            distributions / "ALP-photon" / "Total-yield-ALP-photon_cascades.txt",
        )
        missing_sources = [str(path.relative_to(config.project_root)) for path in source_files if not path.is_file()]
        if missing_sources:
            raise ConfigurationError(
                "Model ALP-mixed is missing installed source table(s): "
                + ", ".join(missing_sources)
                + "."
            )


def config_from_mapping(
    mapping: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
    check_files: bool = True,
) -> SimulationConfig:
    """Validate and normalize a launch-card-like mapping."""
    values = _canonical_card_mapping(mapping)
    missing = [key for key in ("model", "events", "masses", "c_taus") if values.get(key) is None]
    if missing:
        raise ConfigurationError(f"Missing required launch field(s): {', '.join(missing)}.")

    spec = resolve_model(values["model"])
    events = _positive_int(values["events"], "events")
    masses = _number_list(values["masses"], "masses")
    c_taus = _lifetime_grid(values["c_taus"], len(masses))
    decays = _decay_selection(values.get("decay_channels", "all"))
    mixing = _mixing_pattern(values.get("mixing_pattern"))

    uncertainty_value = values.get("uncertainty")
    uncertainty = uncertainty_value.strip().casefold() if isinstance(uncertainty_value, str) else uncertainty_value
    alp_value = values.get("alp_production_mode")
    alp_mode = alp_value.strip().casefold() if isinstance(alp_value, str) else alp_value
    if alp_mode == "cascade":
        alp_mode = "cascades"

    xi_value = values.get("xi")
    interference_value = values.get("interference")
    interference_aliases = {
        "+": "constructive",
        "plus": "constructive",
        "constructive": "constructive",
        "-": "destructive",
        "minus": "destructive",
        "destructive": "destructive",
    }
    interference_key = (
        interference_value.strip().casefold()
        if isinstance(interference_value, str)
        else interference_value
    )
    interference = (
        interference_aliases.get(interference_key)
        if isinstance(interference_key, str)
        else None
    )

    if spec.name == "HNL":
        if mixing is None:
            raise ConfigurationError("Model HNL requires 'mixing_pattern' with three components.")
    elif mixing is not None:
        raise ConfigurationError(f"'mixing_pattern' applies only to HNL, not {spec.name}.")

    if spec.name == "Dark-photons":
        if uncertainty not in {"lower", "central", "upper"}:
            raise ConfigurationError("Dark-photons requires 'uncertainty': lower, central, or upper.")
    elif uncertainty is not None:
        raise ConfigurationError(f"'uncertainty' applies only to Dark-photons, not {spec.name}.")

    if spec.name == "ALP-photon":
        if alp_mode not in {"primary", "cascades"}:
            raise ConfigurationError("ALP-photon requires 'alp_production_mode': primary or cascades.")
    elif alp_mode is not None:
        raise ConfigurationError(f"'alp_production_mode' applies only to ALP-photon, not {spec.name}.")

    if spec.name == "ALP-mixed":
        if xi_value is None:
            raise ConfigurationError("ALP-mixed requires 'xi' between 0 and 1.")
        xi = _finite_float(xi_value, "xi", nonnegative=True)
        if xi > 1.0:
            raise ConfigurationError("'xi' must lie in the closed interval [0, 1].")
        if interference is None:
            raise ConfigurationError(
                "ALP-mixed requires 'interference': constructive or destructive."
            )
        sign = 1.0 if interference == "constructive" else -1.0
        coefficient = sign * (1.0 - xi) + SIN2_THETA_W * xi
        if abs(coefficient) <= 1.0e-12:
            raise ConfigurationError(
                "This destructive xi cancels the diphoton amplitude; the "
                "diphoton-only ALP-mixed model has no finite signal."
            )
    else:
        if xi_value is not None:
            raise ConfigurationError(f"'xi' applies only to ALP-mixed, not {spec.name}.")
        if interference_value is not None:
            raise ConfigurationError(
                f"'interference' applies only to ALP-mixed, not {spec.name}."
            )
        xi = None
        interference = None

    seed_value = values.get("seed")
    if seed_value is not None:
        if isinstance(seed_value, bool):
            raise ConfigurationError("'seed' must be an integer.")
        try:
            seed = int(seed_value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("'seed' must be an integer.") from exc
        if isinstance(seed_value, float) and not seed_value.is_integer():
            raise ConfigurationError("'seed' must be an integer.")
        if seed < 0 or seed > 2**32 - 1:
            raise ConfigurationError("'seed' must be between 0 and 2**32 - 1.")
    else:
        seed = None

    config = SimulationConfig(
        model=spec.name,
        events=events,
        masses=masses,
        c_taus=c_taus,
        decay_channels=decays,
        mixing_pattern=mixing,
        uncertainty=uncertainty,
        alp_production_mode=alp_mode,
        xi=xi,
        interference=interference,
        plots=_strict_bool(values.get("plots", False), "plots"),
        export_events=_strict_bool(values.get("export_events", True), "export_events"),
        n_pot=_finite_float(values.get("n_pot", DEFAULT_N_POT), "n_pot", positive=True),
        min_events_threshold=_finite_float(
            values.get("min_events_threshold", DEFAULT_MIN_EVENTS_THRESHOLD),
            "min_events_threshold",
            nonnegative=True,
        ),
        seed=seed,
        project_root=Path(project_root).resolve(),
    )
    if check_files:
        _validate_installed_files(config)
        # Resolve names/indices now so --validate-only catches channel typos.
        config.selected_decay_indices()
    return config


def load_card(path: Path) -> dict[str, Any]:
    """Load a JSON launch card and require a single top-level object."""
    card_path = Path(path)
    try:
        with card_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise ConfigurationError(f"Could not read launch card {card_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Launch card {card_path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigurationError("A launch card must contain one top-level JSON object.")
    return _canonical_card_mapping(data)


def load_decay_channel_names(config: SimulationConfig) -> tuple[str, ...]:
    spec = resolve_model(config.model)
    path = config.particle_path / spec.decay_file
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Could not read decay-channel file {path}: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise ConfigurationError(f"Decay-channel file {path} contains no channels.")
    try:
        channels = tuple(str(row[0]) for row in data)
    except (TypeError, IndexError) as exc:
        raise ConfigurationError(f"Decay-channel file {path} has an unsupported structure.") from exc
    if len(set(channels)) != len(channels):
        raise ConfigurationError(f"Decay-channel file {path} contains duplicate channel names.")
    return channels


def resolve_decay_channels(selection: Sequence[str | int], available: Sequence[str]) -> list[int]:
    """Resolve ``all``, names, or one-based indices into zero-based indices."""
    channels = tuple(str(item) for item in available)
    if not channels:
        raise ConfigurationError("The selected model has no decay channels.")

    requests_all = any(
        (isinstance(item, str) and item.casefold() == "all") or item == 0 or item == "0"
        for item in selection
    )
    if requests_all:
        if len(selection) != 1:
            raise ConfigurationError("Decay channel 'all' (or index 0) cannot be combined with other channels.")
        return list(range(len(channels)))

    folded: dict[str, list[int]] = {}
    for index, channel in enumerate(channels):
        folded.setdefault(channel.casefold(), []).append(index)

    resolved: list[int] = []
    for item in selection:
        index: int
        if isinstance(item, int) or (isinstance(item, str) and re.fullmatch(r"[+-]?\d+", item)):
            one_based = int(item)
            index = one_based - 1
            if index < 0 or index >= len(channels):
                raise ConfigurationError(
                    f"Decay-channel index {one_based} is out of range 1..{len(channels)}."
                )
        else:
            name = str(item)
            if name in channels:
                index = channels.index(name)
            else:
                matches = folded.get(name.casefold(), [])
                if len(matches) != 1:
                    choices = ", ".join(channels)
                    raise ConfigurationError(f"Unknown decay channel {name!r}. Available channels: {choices}.")
                index = matches[0]
        if index in resolved:
            raise ConfigurationError(f"Decay channel {channels[index]!r} was selected more than once.")
        resolved.append(index)
    return resolved


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run EventCalc without prompts using a JSON card and/or explicit command-line flags. "
            "Running simulate.py with no arguments retains the interactive launcher."
        )
    )
    parser.add_argument("--card", type=Path, help="JSON launch card; explicit flags override card fields")
    parser.add_argument("--model", help="LLP model name, for example ALP-SU2L")
    parser.add_argument("--events", type=int, help="number of accepted decay events to sample")
    parser.add_argument("--masses", nargs="+", type=float, help="LLP masses in GeV")
    parser.add_argument("--c-taus", "--ctaus", dest="c_taus", nargs="+", type=float,
                        help="proper decay lengths in metres, applied to every mass")
    parser.add_argument("--decay-channels", "--decays", dest="decay_channels", nargs="+",
                        help="channel names, one-based indices, or 'all' (default)")
    parser.add_argument("--mixing-pattern", nargs=3, type=float, metavar=("XI_E", "XI_MU", "XI_TAU"))
    parser.add_argument("--uncertainty", choices=("lower", "central", "upper"))
    parser.add_argument("--alp-production-mode", choices=("primary", "cascades"))
    parser.add_argument("--xi", type=float, help="ALP-mixed SU(2)_L operator fraction in [0, 1]")
    parser.add_argument(
        "--interference",
        choices=("constructive", "destructive"),
        help="relative sign of the direct and induced diphoton amplitudes",
    )
    parser.add_argument("--n-pot", type=float, help=f"protons on target (default {DEFAULT_N_POT:.1e})")
    parser.add_argument("--min-events-threshold", type=float,
                        help=f"event-rate cutoff (default {DEFAULT_MIN_EVENTS_THRESHOLD})")
    parser.add_argument("--seed", type=int, help="NumPy random seed")

    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument("--plots", dest="plots", action="store_true",
                            help="write phenomenology plots (off in non-interactive mode by default)")
    plot_group.add_argument("--no-plots", dest="plots", action="store_false")
    export_group = parser.add_mutually_exclusive_group()
    export_group.add_argument("--export-events", dest="export_events", action="store_true")
    export_group.add_argument("--no-export-events", dest="export_events", action="store_false")
    parser.set_defaults(plots=None, export_events=None)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate and print the normalized launch configuration, then exit without simulation",
    )
    return parser


def config_from_command_line(
    argv: Sequence[str],
    *,
    project_root: Path = PROJECT_ROOT,
) -> tuple[SimulationConfig, bool]:
    parser = build_argument_parser()
    namespace = parser.parse_args(list(argv))
    try:
        values: dict[str, Any] = load_card(namespace.card) if namespace.card is not None else {}
        for key in _CONFIG_KEYS:
            cli_value = getattr(namespace, key, None)
            if cli_value is not None:
                values[key] = cli_value
        config = config_from_mapping(values, project_root=project_root)
    except ConfigurationError as exc:
        parser.error(str(exc))
    return config, bool(namespace.validate_only)
