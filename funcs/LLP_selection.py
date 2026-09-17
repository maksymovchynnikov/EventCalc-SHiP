# funcs/LLP_selection.py
"""Interactive input helpers.

Importing this module is side-effect free.  The launcher calls these
functions only when it was started with no command-line arguments.
"""

import math
import os

import numpy as np

from .channel_selection import prompt_decay_channels
from funcs.simulation_config import MODEL_SPECS, SIN2_THETA_W

EVENTCALC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DISTRIBUTIONS_DIR = os.path.join(EVENTCALC_DIR, "Distributions")
N_pot = 6.0e20


def validate_event_count(value):
    """Return a strictly positive integer event count."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("The number of events must be a positive integer.")
    value = int(value)
    if value <= 0:
        raise ValueError("The number of events must be a positive integer.")
    return value


def validate_positive_finite(values, label):
    """Return numeric inputs after rejecting empty/non-finite/non-positive data."""
    validated = [float(value) for value in values]
    if not validated:
        raise ValueError(f"{label} must contain at least one value.")
    invalid = [
        value for value in validated
        if not math.isfinite(value) or value <= 0.0
    ]
    if invalid:
        rendered = ", ".join(repr(value) for value in invalid)
        raise ValueError(
            f"{label} must contain only finite values greater than zero; "
            f"invalid: {rendered}."
        )
    return validated


def validate_masses_in_domain(masses, lower, upper):
    """Validate masses against a closed tabulated domain.

    Text tables can represent a decimal endpoint one floating-point step below
    the value typed by a user (for example ``5.269999999999999`` versus
    ``5.27``).  Accept that one-ULP representation difference, then replace it
    with the exact table endpoint so downstream interpolators never
    extrapolate.
    """
    masses = validate_positive_finite(masses, "masses")
    lower = float(lower)
    upper = float(upper)
    if (not math.isfinite(lower) or not math.isfinite(upper)
            or lower <= 0.0 or upper < lower):
        raise ValueError("The tabulated mass domain is invalid.")

    lower_guard = math.nextafter(lower, -math.inf)
    upper_guard = math.nextafter(upper, math.inf)
    invalid = [
        mass for mass in masses
        if mass < lower_guard or mass > upper_guard
    ]
    if invalid:
        rendered = ", ".join(f"{mass:.17g}" for mass in invalid)
        raise ValueError(
            "masses must lie in the inclusive tabulated domain "
            f"[{lower:.17g}, {upper:.17g}] GeV; invalid: {rendered}."
        )

    return [
        lower if mass < lower else upper if mass > upper else mass
        for mass in masses
    ]


def normalize_hnl_mixing(values):
    """Validate and normalize the three non-negative HNL mixing weights."""
    mixing = np.asarray(values, dtype=float)
    if mixing.shape != (3,):
        raise ValueError("HNL mixing requires exactly three values.")
    if not np.all(np.isfinite(mixing)):
        raise ValueError("HNL mixing values must be finite.")
    if np.any(mixing < 0.0):
        raise ValueError("HNL mixing values must be non-negative.")
    scale = float(np.max(mixing))
    if scale <= 0.0:
        raise ValueError("HNL mixing values must have a finite positive sum.")
    scaled = mixing / scale
    total = float(np.sum(scaled))
    return scaled / total


def prompt_event_count():
    try:
        value = int(input("\nEnter the number of events to simulate: "))
        return validate_event_count(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid input for the number of events: {exc}") from exc


# The launcher and the two-body sampler use both names for the same prompt.
prompt_resample_size = prompt_event_count


def select_particle(distributions_dir=None):
    """Offer the registered models whose distribution folder is installed."""
    main_folder = os.path.abspath(distributions_dir or DISTRIBUTIONS_DIR)
    installed = tuple(
        spec for spec in MODEL_SPECS
        if os.path.isdir(os.path.join(main_folder, spec.directory))
    )
    if not installed:
        raise ValueError(
            f"No registered LLP distributions were found in {main_folder}.")

    print("\nParticle Selector\n")
    for index, spec in enumerate(installed, 1):
        print(f"{index}. {spec.name}")

    try:
        selected = int(input("Select particle: ")) - 1
        if selected < 0:
            raise IndexError
        spec = installed[selected]
    except (IndexError, ValueError) as exc:
        raise ValueError(
            "Invalid selection. Please select a valid particle.") from exc

    return {
        'particle_path': os.path.join(main_folder, spec.directory),
        'LLP_name': spec.name,
    }


def prompt_uncertainty(selection=None):
    # Always defined. If not Dark-photons, return None.
    selection = selection or particle_selection
    if selection['LLP_name'] != "Dark-photons":
        return None
    print("\nWhich variation of the dark photon flux within the uncertainty to select?")
    print("1. lower")
    print("2. central")
    print("3. upper")

    try:
        selected_uncertainty = int(input("Select uncertainty level (1-3): "))
        if selected_uncertainty == 1:
            uncertainty = "lower"
        elif selected_uncertainty == 2:
            uncertainty = "central"
        elif selected_uncertainty == 3:
            uncertainty = "upper"
        else:
            raise ValueError("Invalid selection.")
    except ValueError as e:
        raise ValueError(f"Invalid input for uncertainty level: {e}")
    return uncertainty

def prompt_dp_production_mode(selection=None):
    # Always defined. If not Dark-photons, return None.
    # 'primary' is the standard flux (bremsstrahlung + mixing + meson decays +
    # Drell-Yan from the first proton-target collision). 'cascade' is the
    # additional production off *secondary* protons of the hadronic cascade in
    # the target (hybrid Geant4+Pythia8 proton flux): bremsstrahlung +
    # fragmentation/meson-decay mixing + Drell-Yan, merged. 'brem-cascade' is
    # the legacy bremsstrahlung-only cascade channel (kept for reference).
    selection = selection or particle_selection
    if selection['LLP_name'] != "Dark-photons":
        return None
    print("\nWhich dark-photon production source to use?")
    print("1. primary (standard channels from the primary collision)")
    print("2. cascade (brem + fragmentation + Drell-Yan off secondary protons)")
    print("3. combined (primary + full cascade)")
    print("4. brem-cascade (legacy: bremsstrahlung off secondary protons only)")

    try:
        selected_mode = int(input("Select production source (1-4): "))
        if selected_mode == 1:
            dp_production_mode = "primary"
        elif selected_mode == 2:
            dp_production_mode = "cascade"
        elif selected_mode == 3:
            dp_production_mode = "combined"
        elif selected_mode == 4:
            dp_production_mode = "brem-cascade"
        else:
            raise ValueError("Invalid selection.")
    except ValueError as e:
        raise ValueError(f"Invalid input for dark-photon production source: {e}")
    return dp_production_mode

def prompt_alp_production_mode(selection=None):
    # Always defined. If not ALP-photon, return None.
    selection = selection or particle_selection
    if selection['LLP_name'] != "ALP-photon":
        return None
    print("\nUse distribution from primary collision, or from cascades?")
    print("1. primary")
    print("2. cascades")
    print("3. combined (primary + full cascade)")

    try:
        selected_mode = int(input("Select ALP-photon distribution source (1-3): "))
        if selected_mode == 1:
            alp_production_mode = "primary"
        elif selected_mode == 2:
            alp_production_mode = "cascades"
        elif selected_mode == 3:
            alp_production_mode = "combined"
        else:
            raise ValueError("Invalid selection.")
    except ValueError as e:
        raise ValueError(f"Invalid input for ALP-photon distribution source: {e}")
    return alp_production_mode


def prompt_alp_mixing(selection=None):
    # Always defined. If not ALP-mixed, return (None, None).
    selection = selection or particle_selection
    if selection['LLP_name'] != "ALP-mixed":
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


def prompt_mixing_pattern(selection=None):
    # Always defined. If not HNL, return None.
    selection = selection or particle_selection
    if selection['LLP_name'] != "HNL":
        return None
    try:
        mixing_input = input("\nEnter xi_e, xi_mu, xi_tau: (Ue2, Umu2, Utau2) = U2(xi_e,xi_mu,xi_tau), summing to 1, separated by spaces: ").strip().split()
        if len(mixing_input) != 3:
            raise ValueError("Please enter exactly three numerical values separated by spaces.")

        raw_mixing = np.asarray(list(map(float, mixing_input)), dtype=float)
        MixingPatternArray = normalize_hnl_mixing(raw_mixing)
        if not np.array_equal(raw_mixing, MixingPatternArray):
            print("The entered pattern is not normalized by 1. Renormalizing...")
        return MixingPatternArray

    except ValueError as e:
        raise ValueError(f"Invalid input. Please enter three numerical values separated by spaces: {e}")


def prompt_scalar_lifetime(selection=None):
    # Always defined. If this is not a scalar, return None. If shared
    # prescription tables are unavailable, None makes the scalar loader seek
    # the shared 2407.13587 central card and fail clearly if it is absent.
    selection = selection or particle_selection
    if "Scalar" not in selection['LLP_name']:
        return None
    # The lifetime/BR prescription tables are shared between the scalar
    # types (identical decay physics) and live in the Scalar-mixing folder.
    _pth = selection['particle_path']
    _shared = os.path.join(os.path.dirname(_pth.rstrip("/")), "Scalar-mixing")
    if not os.path.exists(os.path.join(_shared, "ctau-Scalar-1809.01876.txt")):
        return None
    print("\nWhich scalar lifetime (decay-width) prescription?")
    print("1. 1809.01876")
    print("2. 2407.13587 (dispersive; then choose the uncertainty position)")
    try:
        selected = int(input("Select lifetime prescription (1-2): "))
    except ValueError as e:
        raise ValueError(f"Invalid input for scalar lifetime prescription: {e}")
    if selected == 1:
        return "1809.01876"
    elif selected == 2:
        print("\nWhich uncertainty position of the decay width?")
        print("1. central")
        print("2. lower")
        print("3. upper")
        try:
            u = int(input("Select uncertainty (1-3): "))
        except ValueError as e:
            raise ValueError(f"Invalid input for scalar decay-width uncertainty: {e}")
        position = {1: "Central", 2: "Lower", 3: "Upper"}.get(u)
        if position is None:
            raise ValueError("Invalid selection.")
        return f"2407.13587-{position}"
    else:
        raise ValueError("Invalid selection.")

def prompt_masses_and_c_taus(m_min_tabulated=None, m_max_tabulated=None):
    try:
        masses_input = input("\nEnter LLP masses in GeV (separated by spaces): ").split()
        masses = validate_positive_finite(
            [float(m.rstrip('.')) for m in masses_input], "masses")
        if (m_min_tabulated is None) != (m_max_tabulated is None):
            raise ValueError(
                "Both tabulated mass endpoints must be provided together.")
        if m_min_tabulated is not None:
            masses = validate_masses_in_domain(
                masses, m_min_tabulated, m_max_tabulated)

        ifSameLifetimes = True
        c_taus_list = []
        if ifSameLifetimes:
            c_taus_input = input("Enter lifetimes c*tau in m for all masses (separated by spaces): ")
            c_taus = validate_positive_finite(
                [float(tau) for tau in
                 c_taus_input.replace(',', ' ').split()],
                "lifetimes")
            c_taus_list = [c_taus] * len(masses)

        return masses, c_taus_list
    except ValueError as exc:
        raise ValueError(
            "Invalid input for masses or c*taus. "
            f"Please enter finite positive values: {exc}") from exc


_INTERACTIVE_NAMES = {
    "resampleSize",
    "nEvents",
    "particle_selection",
    "mixing_pattern",
    "uncertainty",
    "alp_production_mode",
    "dp_production_mode",
    "scalar_lifetime",
    "xi",
    "interference",
}
_interactive_config = None


def configure_interactively():
    """Collect the historical interactive prompts once and return their state."""
    global _interactive_config
    if _interactive_config is None:
        event_count = prompt_event_count()
        selection = select_particle()
        config = {
            "resampleSize": event_count,
            "nEvents": event_count * 10,
            "particle_selection": selection,
            "uncertainty": prompt_uncertainty(selection),
            "dp_production_mode": prompt_dp_production_mode(selection),
            "alp_production_mode": prompt_alp_production_mode(selection),
            "mixing_pattern": prompt_mixing_pattern(selection),
            "scalar_lifetime": prompt_scalar_lifetime(selection),
        }
        config["xi"], config["interference"] = prompt_alp_mixing(selection)
        globals().update(config)
        _interactive_config = config
    return dict(_interactive_config)


def __getattr__(name):
    """Preserve legacy ``from LLP_selection import <prompted value>`` users."""
    if name in _INTERACTIVE_NAMES:
        configure_interactively()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
