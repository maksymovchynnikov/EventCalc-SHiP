"""Interactive input helpers used by the legacy ``python simulate.py`` flow.

Importing this module is intentionally side-effect free.  The launcher calls
these functions only when it was started with no command-line arguments.
"""

from __future__ import annotations

from pathlib import Path

from funcs.simulation_config import MODEL_SPECS, PROJECT_ROOT, SIN2_THETA_W


N_pot = 6.0e20


def prompt_resample_size() -> int:
    try:
        value = int(input("\nEnter the number of events to simulate: "))
        if value <= 0:
            raise ValueError("The number of events must be a positive integer.")
    except ValueError as exc:
        raise ValueError(f"Invalid input for the number of events: {exc}") from exc
    return value


def select_particle(distributions_folder: str | Path | None = None) -> dict[str, str]:
    main_folder = Path(distributions_folder) if distributions_folder is not None else PROJECT_ROOT / "Distributions"
    installed = tuple(spec for spec in MODEL_SPECS if (main_folder / spec.directory).is_dir())
    if not installed:
        raise ValueError(f"No registered LLP distributions were found in {main_folder}.")

    print("\nParticle Selector\n")
    for index, spec in enumerate(installed, 1):
        print(f"{index}. {spec.name}")

    try:
        selected = int(input("Select particle: ")) - 1
        spec = installed[selected]
        if selected < 0:
            raise IndexError
    except (IndexError, ValueError) as exc:
        raise ValueError("Invalid selection. Please select a valid particle.") from exc

    return {
        "particle_path": str(main_folder / spec.directory),
        "LLP_name": spec.name,
    }


def prompt_uncertainty(particle_selection: dict[str, str]) -> str | None:
    if particle_selection["LLP_name"] != "Dark-photons":
        return None
    print("\nWhich variation of the dark photon flux within the uncertainty to select?")
    print("1. lower")
    print("2. central")
    print("3. upper")

    choices = {1: "lower", 2: "central", 3: "upper"}
    try:
        return choices[int(input("Select uncertainty level (1-3): "))]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid input for uncertainty level: {exc}") from exc


def prompt_alp_production_mode(particle_selection: dict[str, str]) -> str | None:
    if particle_selection["LLP_name"] != "ALP-photon":
        return None
    print("\nUse distribution from primary collision, or from cascades?")
    print("1. primary")
    print("2. cascades")

    choices = {1: "primary", 2: "cascades"}
    try:
        return choices[int(input("Select ALP-photon distribution source (1-2): "))]
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid input for ALP-photon distribution source: {exc}") from exc


def prompt_alp_mixing(
    particle_selection: dict[str, str],
) -> tuple[float | None, str | None]:
    if particle_selection["LLP_name"] != "ALP-mixed":
        return None, None
    try:
        xi = float(input("\nEnter the SU(2)_L operator fraction xi (0 <= xi <= 1): "))
        if not 0.0 <= xi <= 1.0:
            raise ValueError("xi must lie in [0, 1]")
        print("1. constructive")
        print("2. destructive")
        choices = {1: "constructive", 2: "destructive"}
        interference = choices[int(input("Select the interference sign (1-2): "))]
        sign = 1.0 if interference == "constructive" else -1.0
        if abs(sign * (1.0 - xi) + SIN2_THETA_W * xi) <= 1.0e-12:
            raise ValueError("this destructive xi cancels the diphoton amplitude")
        return xi, interference
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Invalid ALP-mixed parameters: {exc}") from exc


def prompt_mixing_pattern(particle_selection: dict[str, str]) -> tuple[float, float, float] | None:
    if particle_selection["LLP_name"] != "HNL":
        return None
    try:
        mixing_input = input(
            "\nEnter xi_e, xi_mu, xi_tau: (Ue2, Umu2, Utau2) = "
            "U2(xi_e,xi_mu,xi_tau), summing to 1, separated by spaces: "
        ).strip().split()
        if len(mixing_input) != 3:
            raise ValueError("Please enter exactly three numerical values separated by spaces.")

        entries = tuple(map(float, mixing_input))
        if any(value < 0.0 for value in entries):
            raise ValueError("Mixing components must be non-negative.")
        total = sum(entries)
        if total <= 0.0:
            raise ValueError("At least one mixing component must be positive.")
        if total != 1.0:
            print("The entered pattern is not normalized by 1. Renormalizing...")
        return tuple(value / total for value in entries)
    except ValueError as exc:
        raise ValueError(
            "Invalid input. Please enter three numerical values separated by spaces: "
            f"{exc}"
        ) from exc


def prompt_masses_and_c_taus() -> tuple[list[float], list[list[float]]]:
    try:
        masses_input = input("\nEnter LLP masses in GeV (separated by spaces): ").split()
        masses = [float(m.rstrip(".")) for m in masses_input]
        if not masses:
            raise ValueError("At least one mass is required.")

        c_taus_input = input("Enter lifetimes c*tau in m for all masses (separated by spaces): ")
        c_taus = [float(tau) for tau in c_taus_input.replace(",", " ").split()]
        if not c_taus:
            raise ValueError("At least one lifetime is required.")
        return masses, [list(c_taus) for _ in masses]
    except ValueError as exc:
        raise ValueError(
            "Invalid input for masses or c*taus. Please enter numerical values."
        ) from exc


def prompt_decay_channels(decay_channels) -> list[int]:
    print("\nSelect the decay modes:")
    print("0. All")
    for index, channel in enumerate(decay_channels, 1):
        print(f"{index}. {channel}")

    user_input = input("Enter the numbers of the decay channels to select (separated by spaces): ")
    try:
        selected = [int(value) for value in user_input.strip().split()]
        if not selected:
            raise ValueError("No selection made.")
        if 0 in selected:
            if len(selected) != 1:
                raise ValueError("Channel 0 (All) cannot be combined with other channels.")
            return list(range(len(decay_channels)))
        indices = [value - 1 for value in selected]
        for index in indices:
            if index < 0 or index >= len(decay_channels):
                raise ValueError(f"Invalid index {index + 1}.")
        if len(set(indices)) != len(indices):
            raise ValueError("A decay channel was selected more than once.")
        return indices
    except ValueError as exc:
        raise ValueError(f"Invalid input for decay channel selection: {exc}") from exc
