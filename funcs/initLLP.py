# initLLP.py
import json
import os
import re  # Added for regex operations
import numpy as np
import pandas as pd
from funcs import ALPmerging, HNLmerging
from funcs import coupling_conventions
from funcs.alp_fermion import (build_br_interpolator, load_exhad_boundaries,
                               prepare_decay_rows)
from funcs.kinematics import ProductionMixture
from scipy.interpolate import RegularGridInterpolator, PchipInterpolator
import sympy as sp


DEFAULT_SCALAR_PRESCRIPTION = "2407.13587-Central"


def make_log_pchip_interpolator(masses, values):
    masses = np.asarray(masses, dtype=float)
    values = np.asarray(values, dtype=float)

    if np.any(masses <= 0.0):
        raise ValueError("All masses must be positive.")
    if np.any(values <= 0.0):
        raise ValueError("All interpolated values must be positive.")

    interpolator = PchipInterpolator(
        np.log(masses),
        np.log(values),
        extrapolate=False,
    )

    def evaluate(mass):
        mass = float(mass)
        return float(np.exp(interpolator(np.log(mass))))

    return evaluate


def _normalize_alp_production_mode(mode):
    aliases = {
        "primary": "primary",
        "cascade": "cascades",
        "cascades": "cascades",
        "combined": "combined",
        "primary+cascade": "combined",
        "cascade+primary": "combined",
    }
    try:
        return aliases[str(mode).strip().lower()]
    except KeyError as exc:
        raise ValueError(
            "ALP-photon production mode must be primary, cascade, or "
            "combined") from exc


def _normalize_dp_production_mode(mode):
    aliases = {
        "primary": "primary",
        "cascade": "cascade",
        "brem-cascade": "brem-cascade",
        "combined": "combined",
        "primary+cascade": "combined",
        "cascade+primary": "combined",
    }
    try:
        return aliases[str(mode).strip().lower()]
    except KeyError as exc:
        raise ValueError(
            "Dark-photon production mode must be primary, cascade, "
            "brem-cascade, or combined") from exc


def _read_production_tables(distribution_path, energy_path, yield_path):
    distribution = pd.read_csv(distribution_path, header=None, sep="\t")
    maximum_energy = pd.read_csv(energy_path, header=None, sep="\t")
    yield_data = pd.read_csv(yield_path, header=None, sep="\t")
    masses = yield_data.iloc[:, 0].to_numpy()
    values = yield_data.iloc[:, 1].to_numpy()
    interpolator = RegularGridInterpolator(
        (masses,), values, bounds_error=True)

    def yield_function(mass, source=interpolator):
        return float(source([float(mass)])[0])

    limits = (
        max(
            float(np.min(masses)),
            float(distribution.iloc[:, 0].min()),
            float(maximum_energy.iloc[:, 0].min()),
        ),
        min(
            float(np.max(masses)),
            float(distribution.iloc[:, 0].max()),
            float(maximum_energy.iloc[:, 0].max()),
        ),
    )
    return (
        distribution,
        maximum_energy,
        yield_data,
        yield_function,
        limits,
    )


def _set_combined_tabulated_range(llp, source_limits, ctau_data):
    mass_ctau = ctau_data.iloc[:, 0].to_numpy()
    llp.m_min_tabulated = max(
        *(limits[0] for limits in source_limits),
        float(np.min(mass_ctau)),
    )
    llp.m_max_tabulated = min(
        *(limits[1] for limits in source_limits),
        float(np.max(mass_ctau)),
    )
    if llp.m_min_tabulated >= llp.m_max_tabulated:
        raise ValueError(
            "combined production sources have no common tabulated mass range")


def _load_scalar_ctau_json(path):
    """Load one modern scalar lifetime table as strict JSON ``[m, ctau]`` rows.

    The prescription-specific scalar files retain their historical ``.txt``
    names, but their installed schema is JSON.  Parse them explicitly instead
    of relying on a delimiter heuristic, and reject malformed grids before
    constructing an interpolator.
    """
    try:
        with open(path) as stream:
            raw = json.load(stream)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"Cannot load scalar lifetime JSON table: {path}") from exc

    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError(
            f"Scalar lifetime table must contain at least two JSON rows: {path}")

    rows = []
    previous_mass = None
    for index, node in enumerate(raw):
        if not isinstance(node, list) or len(node) != 2:
            raise ValueError(
                "Scalar lifetime row %d must be [mass_GeV, c_tau_m]: %s"
                % (index, path))
        if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                for value in node):
            raise ValueError(
                "Scalar lifetime row %d contains a non-numeric value: %s"
                % (index, path))
        mass, ctau = (float(node[0]), float(node[1]))
        if not np.isfinite(mass) or mass <= 0.0:
            raise ValueError(
                "Scalar lifetime row %d has an invalid mass: %s"
                % (index, path))
        if not np.isfinite(ctau) or ctau <= 0.0:
            raise ValueError(
                "Scalar lifetime row %d has an invalid c_tau: %s"
                % (index, path))
        if previous_mass is not None and mass <= previous_mass:
            raise ValueError(
                "Scalar lifetime masses must be strictly increasing at row "
                "%d: %s" % (index, path))
        rows.append((mass, ctau))
        previous_mass = mass

    return pd.DataFrame(rows, columns=[0, 1], dtype=np.float64)


def _validate_scalar_branching_vector(
        mass, prescription, decay_channels, branching_fractions):
    """Reject unphysical scalar-table evaluations before event allocation."""
    values = np.asarray(branching_fractions, dtype=float)
    if values.ndim != 1 or len(values) != len(decay_channels):
        raise ValueError(
            "Scalar branching table returned an invalid channel vector at "
            f"m={float(mass):.9g} GeV for {prescription}")
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "Scalar branching table returned a non-finite probability at "
            f"m={float(mass):.9g} GeV for {prescription}")

    tolerance = 1.0e-9
    negative = np.flatnonzero(values < 0.0)
    if len(negative):
        labels = ", ".join(str(decay_channels[index]) for index in negative)
        raise ValueError(
            "Scalar branching table contains negative probabilities at "
            f"m={float(mass):.9g} GeV for {prescription}: {labels}. "
            "Regenerate the source table; probabilities are not clipped.")

    total = float(np.sum(values))
    if total > 1.0 + tolerance:
        raise ValueError(
            "Scalar branching probabilities exceed unity at "
            f"m={float(mass):.9g} GeV for {prescription}: "
            f"sum={total:.12g}. Regenerate the source table.")
    return values


class LLP:
    """
    Define LLP, now also define m_min_tabulated and m_max_tabulated.
    For non-HNL:
      m_min_yield, m_max_yield from yield data
      m_min_distribution, m_max_distribution from distr
      m_min_lifetime, m_max_lifetime from ctau data
    For HNL:
      from DistrHNL_e (distribution), HNL_yield_e (yield), DW_e (decay width)
    """

    def __init__(self, mass, particle_selection, mixing_pattern=None, uncertainty=None, alp_production_mode=None, dp_production_mode=None, scalar_lifetime=None, xi=None, interference=None):
        # Dense kinematics grids may be reused only under this explicit,
        # caller-owned source revision.  The namespace prevents accidental
        # collisions between separately loaded LLP objects; the revision is
        # advanced whenever the source distribution is replaced.
        self._kinematics_cache_namespace = object()
        self._kinematics_source_revision = 0
        self.main_folder = "./Distributions"
        self.LLP_name = particle_selection['LLP_name']
        self.mass = mass
        self.particle_path = particle_selection['particle_path']
        self.MixingPatternArray = mixing_pattern if mixing_pattern is not None else None
        self.uncertainty = uncertainty if self.LLP_name == "Dark-photons" else None
        self.alp_production_mode = (
            _normalize_alp_production_mode(alp_production_mode)
            if self.LLP_name == "ALP-photon"
            and alp_production_mode is not None else None)
        # Dark-photon production source: ``primary`` uses the standard
        # first-collision channels; ``cascade`` combines bremsstrahlung,
        # fragmentation/meson mixing, and Drell--Yan from secondary protons;
        # ``brem-cascade`` retains the legacy secondary-proton
        # bremsstrahlung-only sample.
        self.dp_production_mode = (
            _normalize_dp_production_mode(dp_production_mode or "primary")
            if self.LLP_name == "Dark-photons" else None)
        # Both scalar production models share the prescription-specific
        # Scalar-mixing lifetime/BR cards.  Missing prescriptions resolve to
        # the central table; the obsolete per-folder KLKS fallback is not a
        # supported decay model.
        self.scalar_lifetime = (
            scalar_lifetime or DEFAULT_SCALAR_PRESCRIPTION
            if "Scalar" in self.LLP_name else None)
        # The photon/SU(2)_L mixture is defined by the operator fraction xi
        # and the sign of the interference between the two amplitudes.
        self.xi = float(xi) if self.LLP_name == "ALP-mixed" and xi is not None else None
        self.interference = interference if self.LLP_name == "ALP-mixed" else None
        # label used in output file names: keeps the historic pure-uncertainty
        # naming for the primary source
        if self.LLP_name == "Dark-photons" and self.dp_production_mode != "primary":
            self.uncertainty_label = f"{self.dp_production_mode}-{self.uncertainty}"
        elif "Scalar" in self.LLP_name and self.scalar_lifetime is not None:
            self.uncertainty_label = self.scalar_lifetime
        else:
            self.uncertainty_label = self.uncertainty
        self.Matrix_elements = None
        self.Matrix_elements_expr = []  # To store symbolic expressions
        # Set only for models with a machine-readable native coupling contract.
        # In particular, ALP-fermion uses g_Y == y == 2*v_h/f_a.
        self.coupling_metadata = None

        self.import_particle()
        self.mark_kinematics_source_changed()
        self.mass_range = self._get_mass_range()

    def _get_mass_range(self):
        if "HNL" in self.LLP_name:
            # For HNL defined differently below after data load
            pass
        else:
            # For non-HNL, we already define after loading in import_particle
            pass
        return (self.m_min_tabulated, self.m_max_tabulated)

    def set_mass(self, mass):
        self.mass = mass

    def set_c_tau(self, c_tau_input):
        self.c_tau_input = c_tau_input

    def mark_kinematics_source_changed(self):
        """Advance the explicit cache revision for ``Distr/Energy_distr``.

        Call this after replacing or mutating either source table.  Lifetime
        changes intentionally leave the revision untouched, allowing the
        expensive dense grids to be reused throughout a fixed-source scan.
        """
        self._kinematics_source_revision += 1

    def kinematics_cache_token(self):
        """Return the hashable identity/revision consumed by ``Grids``."""
        return (self._kinematics_cache_namespace,
                self._kinematics_source_revision)

    def _replace_kinematics_distribution(self, distribution):
        """Install a new mass-dependent distribution and revise its token."""
        self.Distr = distribution
        self.mark_kinematics_source_changed()

    def compute_mass_dependent_properties(self):
        if self.mass is None:
            raise ValueError("Mass must be set before computing mass-dependent properties.")
        if "Scalar" in self.LLP_name:
            self.compute_mass_dependent_properties_scalars()
        elif self.LLP_name == "HNL":
            self.compute_mass_dependent_properties_HNL()
        elif self.LLP_name == "ALP-photon":
            self.compute_mass_dependent_properties_ALP_photon()
        elif self.LLP_name == "ALP-fermion":
            self.compute_mass_dependent_properties_ALP_fermion()
        elif self.LLP_name == "Dark-photons":
            self.compute_mass_dependent_properties_dark_photons()
        elif self.LLP_name == "ALP-SU2L":
            self.compute_mass_dependent_properties_ALP_SU2L()
        elif self.LLP_name == "ALP-mixed":
            self.compute_mass_dependent_properties_ALP_mixed()
        else:
            raise ValueError("Unknown LLP name.")

    def compute_mass_dependent_properties_scalars(self):
        rates = self.get_Br(self.mass)
        # The matched exHad scalar deployments own the explicit extrapolation
        # of the release's rate and lifetime tables. Native tables are unchanged.
        from funcs import exhadDecays
        if exhadDecays.uses_portable_model1_backend():
            from funcs.scalar_hadronic_continuation import load_continuation
            tables = exhadDecays.portable_model1_configuration()["tables"]
            values = load_continuation(tables["decay"], tables["ctau"]).at(self.mass)
            rates = np.array([values[str(k)] for k in self.decayChannels])
        self.BrRatios_distr = _validate_scalar_branching_vector(
            self.mass, self.scalar_lifetime, self.decayChannels,
            rates)
        self.c_tau_int = self.get_ctau(self.mass)
        self.Yield = self.get_total_yield(self.mass)

    def compute_mass_dependent_properties_ALP_photon(self):
        self.BrRatios_distr = self.get_Br(self.mass)
        self.c_tau_int = self.get_ctau(self.mass)
        self.Yield = self.get_total_yield(self.mass)

    def compute_mass_dependent_properties_ALP_fermion(self):
        self.BrRatios_distr = self.get_Br(self.mass)
        self.c_tau_int = self.get_ctau(self.mass)
        self.Yield = self.get_total_yield(self.mass)

    def compute_mass_dependent_properties_dark_photons(self):
        self.BrRatios_distr = self.get_Br(self.mass)
        self.c_tau_int = self.get_ctau(self.mass)
        self.Yield = self.get_total_yield(self.mass)

    def compute_mass_dependent_properties_ALP_SU2L(self):
        self.BrRatios_distr = self.get_Br(self.mass)
        self.c_tau_int = self.get_ctau(self.mass)
        self.Yield = self.get_total_yield(self.mass)

    def compute_mass_dependent_properties_ALP_mixed(self):
        self.BrRatios_distr = self.get_Br(self.mass)
        self.c_tau_int = self.get_ctau(self.mass)
        self.Yield = self.get_total_yield(self.mass)

    def compute_mass_dependent_properties_HNL(self):
        self.c_tau_int = self.get_ctau(self.mass)
        self.Yield = self.get_total_yield(self.mass)
        self.BrRatios_distr = self.get_Br(self.mass)
        self.Matrix_elements = self.get_MatrixElements(self.mass)
        self._replace_kinematics_distribution(
            self.get_distribution(self.mass))

    def import_particle(self):
        if "Scalar" in self.LLP_name:
            self.import_scalars()
        elif self.LLP_name == "HNL":
            if self.MixingPatternArray is None:
                raise ValueError("Mixing pattern must be provided for HNL.")
            self.import_HNL()
        elif self.LLP_name == "ALP-photon":
            if self.alp_production_mode is None:
                raise ValueError("ALP-photon production mode must be provided.")
            self.import_ALP_photon()
        elif self.LLP_name == "ALP-fermion":
            self.import_ALP_fermion()
        elif self.LLP_name == "Dark-photons":
            if self.uncertainty is None:
                raise ValueError("Uncertainty must be provided for Dark-photons.")
            self.import_dark_photons()
        elif self.LLP_name == "ALP-SU2L":
            self.import_ALP_SU2L()
        elif self.LLP_name == "ALP-mixed":
            if self.xi is None or self.interference is None:
                raise ValueError("ALP-mixed requires xi and an interference sign.")
            self.import_ALP_mixed()
        else:
            raise ValueError("Unknown LLP name.")

    def define_tabulated_range_nonHNL(self, mass_yield, yield_values, distr_data, ctau_data):
        # yield data range
        m_min_yield = mass_yield.min()
        m_max_yield = mass_yield.max()

        # distribution range
        # distr_data assumed to have mass in column 0
        masses_distr = np.unique(distr_data.iloc[:,0])
        m_min_distribution = masses_distr.min()
        m_max_distribution = masses_distr.max()

        # ctau data range
        mass_ctau = ctau_data.iloc[:,0].to_numpy()
        m_min_lifetime = mass_ctau.min()
        m_max_lifetime = mass_ctau.max()

        self.m_min_tabulated = max(m_min_yield, m_min_distribution, m_min_lifetime)
        self.m_max_tabulated = min(m_max_yield, m_max_distribution, m_max_lifetime)

    def define_tabulated_range_HNL(self):
        # For HNL:
        # distribution e mixing: from DistrHNL_e
        e_masses = np.unique(self.DistrDataFrames[0].iloc[:,0])
        m_min_distribution = e_masses.min()
        m_max_distribution = e_masses.max()

        # yield e mixing: from HNL_yield_e
        yield_mass = self.yieldData[0]
        m_min_yield = yield_mass.min()
        m_max_yield = yield_mass.max()

        # decay width e mixing: from DW_e = self.decayWidthData[1]
        decay_mass = self.decayWidthData[0]
        m_min_lifetime = decay_mass.min()
        m_max_lifetime = decay_mass.max()

        self.m_min_tabulated = max(m_min_yield, m_min_distribution, m_min_lifetime)
        self.m_max_tabulated = min(m_max_yield, m_max_distribution, m_max_lifetime)

    def import_scalars(self):
        if "mixing" in self.LLP_name:
            suffix = "mixing"
        elif "quartic" in self.LLP_name:
            suffix = "quartic"
        else:
            raise ValueError("Unknown Scalar type.")

        distribution_file_path = os.path.join(self.particle_path, f"DoubleDistr-Scalar-{suffix}.txt")
        energy_file_path = os.path.join(self.particle_path, f"Emax-Scalar-{suffix}.txt")
        yield_path = os.path.join(self.particle_path, f"Total-yield-Scalar-{suffix}.txt")
        # Lifetime and branching-ratio tables are shared between the two
        # scalar types (identical decay physics; the types differ only in
        # production): both are read from the Scalar-mixing folder, with
        # matched ctau/BrRatio pairs per prescription. The BrRatio tables
        # bridge the exclusive (pi pi, K Kbar) and inclusive (Jets) regimes
        # through a rho-rho-mediated 4pi ansatz.
        shared_path = os.path.join(
            os.path.dirname(self.particle_path.rstrip("/")), "Scalar-mixing")
        ctau_path = os.path.join(
            shared_path, f"ctau-Scalar-{self.scalar_lifetime}.txt")
        decay_json_path = os.path.join(
            shared_path, f"BrRatio-Scalar-{self.scalar_lifetime}.json")
        if not os.path.exists(ctau_path):
            raise FileNotFoundError(
                f"Scalar lifetime table not found: {ctau_path}")
        if not os.path.exists(decay_json_path):
            raise FileNotFoundError(
                f"Scalar BrRatio table not found: {decay_json_path}")

        self.Distr = pd.read_csv(distribution_file_path, header=None, sep="\t")
        self.Energy_distr = pd.read_csv(energy_file_path, header=None, sep="\t")
        self.Yield_data = pd.read_csv(yield_path, header=None, sep="\t")
        self.ctau_data = _load_scalar_ctau_json(ctau_path)

        mass_ctau = self.ctau_data.iloc[:, 0].to_numpy()
        ctau_values = self.ctau_data.iloc[:, 1].to_numpy()
        self.ctau_interpolator = RegularGridInterpolator((mass_ctau,), ctau_values, bounds_error=True)

        mass_yield = self.Yield_data.iloc[:, 0].to_numpy()
        yield_values = self.Yield_data.iloc[:, 1].to_numpy()
        self.yield_interpolator = RegularGridInterpolator((mass_yield,), yield_values, bounds_error=True)

        HLS_decay = pd.read_json(decay_json_path)
        self.decayChannels = HLS_decay.iloc[:, 0].to_numpy()
        self.PDGs = HLS_decay.iloc[:, 1].apply(np.array).to_numpy()
        self.BrRatios = HLS_decay.iloc[:, 2].to_numpy()

        # For scalars without specific matrix elements, default to "1."
        self.Matrix_elements_raw = ["1."] * len(self.decayChannels)

        # Compile matrix elements
        self.Matrix_elements = self.compile_matrix_elements(self.Matrix_elements_raw)

        self.get_ctau = lambda m: self.ctau_interpolator([m])[0]
        self.get_total_yield = lambda m: self.yield_interpolator([m])[0]
        self.get_Br = self.setup_br_interpolators(self.BrRatios)
        self.get_distribution = lambda m: self.Distr
        self.get_MatrixElements = lambda m: self.Matrix_elements

        # define tabulated range
        self.define_tabulated_range_nonHNL(mass_yield, yield_values, self.Distr, self.ctau_data)

        # Print the matrix elements table
        #self.print_matrix_elements()

    def import_ALP_photon(self):
        def source_file(prefix, mode):
            suffix_candidates = [mode]
            if mode == "cascades":
                suffix_candidates.append("cascade")
            elif mode == "cascade":
                suffix_candidates.append("cascades")
            for suffix in suffix_candidates:
                path = os.path.join(self.particle_path, f"{prefix}_{suffix}.txt")
                if os.path.exists(path):
                    return path
            searched = ", ".join(f"{prefix}_{suffix}.txt" for suffix in suffix_candidates)
            raise FileNotFoundError(f"Could not find ALP-photon source file. Tried: {searched}")

        ctau_path = os.path.join(self.particle_path, "ctau-ALP-photon.txt")
        decay_json_path = os.path.join(self.particle_path, "ALP-photon-decay.json")

        if self.alp_production_mode == "combined":
            source_rows = []
            source_limits = []
            source_yield_data = {}
            for label in ("primary", "cascades"):
                tables = _read_production_tables(
                    source_file("DoubleDistr-ALP-photon", label),
                    source_file("Emax-ALP-photon", label),
                    source_file("Total-yield-ALP-photon", label),
                )
                distribution, maximum_energy, yield_data, yield_function, limits = tables
                source_rows.append(
                    (label, distribution, maximum_energy, yield_function))
                source_limits.append(limits)
                source_yield_data[label] = yield_data
            self.Distr = ProductionMixture(source_rows)
            self.Energy_distr = None
            self.Yield_data = source_yield_data
            self.production_component_labels = ("primary", "cascades")
        else:
            distribution_file_path = source_file(
                "DoubleDistr-ALP-photon", self.alp_production_mode)
            energy_file_path = source_file(
                "Emax-ALP-photon", self.alp_production_mode)
            yield_path = source_file(
                "Total-yield-ALP-photon", self.alp_production_mode)
            self.Distr = pd.read_csv(
                distribution_file_path, header=None, sep="\t")
            self.Energy_distr = pd.read_csv(
                energy_file_path, header=None, sep="\t")
            self.Yield_data = pd.read_csv(
                yield_path, header=None, sep="\t")
        self.ctau_data = pd.read_csv(ctau_path, header=None, sep="\t")

        mass_ctau = self.ctau_data.iloc[:, 0].to_numpy()
        ctau_values = self.ctau_data.iloc[:, 1].to_numpy()
        self.ctau_interpolator = RegularGridInterpolator((mass_ctau,), ctau_values, bounds_error=True)

        if self.alp_production_mode != "combined":
            mass_yield = self.Yield_data.iloc[:, 0].to_numpy()
            yield_values = self.Yield_data.iloc[:, 1].to_numpy()
            self.yield_interpolator = RegularGridInterpolator(
                (mass_yield,), yield_values, bounds_error=True)

        ALP_decay = pd.read_json(decay_json_path)
        self.decayChannels = ALP_decay.iloc[:, 0].to_numpy()
        self.PDGs = ALP_decay.iloc[:, 1].apply(np.array).to_numpy()
        self.BrRatios = ALP_decay.iloc[:, 2].to_numpy()

        self.Matrix_elements_raw = ALP_decay.iloc[:, -1].to_numpy()

        # Compile matrix elements
        self.Matrix_elements = self.compile_matrix_elements(self.Matrix_elements_raw)

        self.get_ctau = lambda m: self.ctau_interpolator([m])[0]
        if self.alp_production_mode == "combined":
            self.get_total_yield = self.Distr.total_yield
        else:
            self.get_total_yield = lambda m: self.yield_interpolator([m])[0]
        self.get_Br = self.setup_br_interpolators(self.BrRatios)
        self.get_distribution = lambda m: self.Distr
        self.get_MatrixElements = lambda m: self.Matrix_elements

        # define tabulated range
        if self.alp_production_mode == "combined":
            _set_combined_tabulated_range(
                self, source_limits, self.ctau_data)
        else:
            self.define_tabulated_range_nonHNL(
                mass_yield, yield_values, self.Distr, self.ctau_data)

        # Print the matrix elements table
        #self.print_matrix_elements()

    def import_ALP_fermion(self):
        """Load the fermion-universal ALP and its coupling-normalized tables.

        The native coupling is ``g_Y == y == 2*v_h/f_a``.  The lifetime table
        stores ``c_tau*g_Y^2`` and the yield table stores production probability
        per POT per ``g_Y^2``.  Loading their validated sidecar before either
        table prevents a silent reinterpretation as ``v_h/f_a``.
        """
        self.coupling_metadata = coupling_conventions.load_coupling_metadata(
            self.particle_path)
        distribution_file_path = os.path.join(
            self.particle_path, "DoubleDistr-ALP-fermion.txt")
        energy_file_path = os.path.join(
            self.particle_path, "Emax-ALP-fermion.txt")
        yield_path = os.path.join(
            self.particle_path, "Total-yield-ALP-fermion.txt")
        ctau_path = os.path.join(
            self.particle_path, "ctau-ALP-fermion.txt")
        decay_json_path = os.path.join(
            self.particle_path, "ALP-fermion-decay.json")

        self.Distr = pd.read_csv(distribution_file_path, header=None, sep="\t")
        self.Energy_distr = pd.read_csv(energy_file_path, header=None, sep="\t")
        self.Yield_data = pd.read_csv(yield_path, header=None, sep="\t")
        self.ctau_data = pd.read_csv(ctau_path, header=None, sep="\t")

        mass_ctau = self.ctau_data.iloc[:, 0].to_numpy()
        ctau_values = self.ctau_data.iloc[:, 1].to_numpy()
        self.ctau_interpolator = RegularGridInterpolator(
            (mass_ctau,), ctau_values, bounds_error=True)
        mass_yield = self.Yield_data.iloc[:, 0].to_numpy()
        yield_values = self.Yield_data.iloc[:, 1].to_numpy()
        self.yield_interpolator = RegularGridInterpolator(
            (mass_yield,), yield_values, bounds_error=True)

        with open(decay_json_path) as f:
            decay_rows = json.load(f)
        labels, pdgs, br_tables, matrix_elements = prepare_decay_rows(
            decay_rows)
        self.decayChannels = np.array(labels, dtype=object)
        self.PDGs = np.empty(len(decay_rows), dtype=object)
        self.PDGs[:] = [np.array(row) for row in pdgs]
        self.BrRatios = np.empty(len(decay_rows), dtype=object)
        self.BrRatios[:] = br_tables
        self.Matrix_elements_raw = np.array(matrix_elements, dtype=object)
        self.Matrix_elements = self.compile_matrix_elements(
            self.Matrix_elements_raw)

        m_start, m_charm = load_exhad_boundaries(self.particle_path)
        br_interp = build_br_interpolator(
            self.decayChannels,
            self.PDGs,
            self.BrRatios,
            m_start,
            m_charm,
            self.ctau_data.itertuples(index=False, name=None),
        )
        self.get_Br = lambda m: np.array(br_interp(m), dtype=float)
        self.get_ctau = lambda m: self.ctau_interpolator([m])[0]
        self.get_total_yield = lambda m: self.yield_interpolator([m])[0]
        self.get_distribution = lambda m: self.Distr
        self.get_MatrixElements = lambda m: self.Matrix_elements

        self.define_tabulated_range_nonHNL(
            mass_yield, yield_values, self.Distr, self.ctau_data)

    def import_ALP_SU2L(self):
        distribution_file_path = os.path.join(self.particle_path, "DoubleDistr-ALP-SU2L.txt")
        energy_file_path = os.path.join(self.particle_path, "Emax-ALP-SU2L.txt")
        yield_path = os.path.join(self.particle_path, "Total-yield-ALP-SU2L.txt")
        ctau_path = os.path.join(self.particle_path, "ctau-ALP-SU2L.txt")
        decay_json_path = os.path.join(self.particle_path, "ALP-SU2L-decay.json")

        self.Distr = pd.read_csv(distribution_file_path, header=None, sep="\t")
        self.Energy_distr = pd.read_csv(energy_file_path, header=None, sep="\t")
        self.Yield_data = pd.read_csv(yield_path, header=None, sep="\t")
        self.ctau_data = pd.read_csv(ctau_path, header=None, sep="\t")

        mass_ctau = self.ctau_data.iloc[:, 0].to_numpy()
        ctau_values = self.ctau_data.iloc[:, 1].to_numpy()
        self.ctau_interpolator = RegularGridInterpolator((mass_ctau,), ctau_values, bounds_error=True)

        mass_yield = self.Yield_data.iloc[:, 0].to_numpy()
        yield_values = self.Yield_data.iloc[:, 1].to_numpy()
        self.yield_interpolator = RegularGridInterpolator((mass_yield,), yield_values, bounds_error=True)

        ALP_SU2L_decay = pd.read_json(decay_json_path, dtype=False)
        self.decayChannels = ALP_SU2L_decay.iloc[:, 0].to_numpy()
        self.PDGs = (ALP_SU2L_decay.iloc[:, 1].apply(np.array).to_numpy())
        self.BrRatios = ALP_SU2L_decay.iloc[:, 2].to_numpy()

        self.Matrix_elements_raw = ALP_SU2L_decay.iloc[:, -1].to_numpy()

        # Compile matrix elements
        self.Matrix_elements = self.compile_matrix_elements(self.Matrix_elements_raw)

        # The lifetime and the yield vary over many orders of magnitude across
        # this mass range; both are read on a log-log monotone interpolant.
        self.get_ctau = make_log_pchip_interpolator(mass_ctau, ctau_values)
        self.get_total_yield = make_log_pchip_interpolator(mass_yield, yield_values)
        self.get_Br = self.setup_br_interpolators(self.BrRatios)
        self.get_distribution = lambda m: self.Distr
        self.get_MatrixElements = lambda m: self.Matrix_elements

        # define tabulated range
        self.define_tabulated_range_nonHNL(mass_yield, yield_values, self.Distr, self.ctau_data)

        # Print the matrix elements table
        #self.print_matrix_elements()

    def import_ALP_mixed(self):
        distributions_root = os.path.dirname(self.particle_path)
        tables = ALPmerging.build_mixed_tables(
            distributions_root,
            self.xi,
            self.interference,
        )
        self.Distr = tables.distribution
        self.Energy_distr = tables.emax
        self.Yield_data = tables.yield_table
        self.ctau_data = tables.ctau_table
        self.flavor_to_photon_ratio = tables.source_ratio

        decay_json_path = os.path.join(self.particle_path, "ALP-mixed-decay.json")
        decay_data = pd.read_json(decay_json_path, dtype=False)
        self.decayChannels = decay_data.iloc[:, 0].to_numpy()
        self.PDGs = decay_data.iloc[:, 1].apply(np.array).to_numpy()
        self.BrRatios = decay_data.iloc[:, 2].to_numpy()
        self.Matrix_elements_raw = decay_data.iloc[:, -1].to_numpy()
        self.Matrix_elements = self.compile_matrix_elements(self.Matrix_elements_raw)

        mass_yield = self.Yield_data.iloc[:, 0].to_numpy()
        yield_values = self.Yield_data.iloc[:, 1].to_numpy()
        common_support = (
            (mass_yield >= tables.mass_min)
            & (mass_yield <= tables.mass_max)
            & (yield_values > 0.0)
        )
        self.yield_interpolator = make_log_pchip_interpolator(
            mass_yield[common_support], yield_values[common_support]
        )

        self.get_ctau = ALPmerging.diphoton_ctau_coefficient
        self.get_total_yield = self.yield_interpolator
        self.get_Br = self.setup_br_interpolators(self.BrRatios)
        self.get_distribution = lambda m: self.Distr
        self.get_MatrixElements = lambda m: self.Matrix_elements
        self.m_min_tabulated = tables.mass_min
        self.m_max_tabulated = tables.mass_max

    def import_dark_photons(self):
        # 'primary': historic file names; 'cascade': merged secondary-proton
        # production (bremsstrahlung + fragmentation/meson-decay mixing +
        # Drell-Yan; produced by secondary-channels/merge_cascade_channels.py
        # from the hybrid Geant4+Pythia8 proton flux inside the SHiP target);
        # 'brem-cascade': legacy bremsstrahlung-only cascade tables
        if self.dp_production_mode in ("cascade", "brem-cascade"):
            suffix = f"{self.dp_production_mode}-{self.uncertainty}"
        else:
            suffix = self.uncertainty
        ctau_path = os.path.join(self.particle_path, "ctau-DP.txt")
        decay_json_path = os.path.join(self.particle_path, "DP-decay.json")

        if self.dp_production_mode == "combined":
            source_rows = []
            source_limits = []
            source_yield_data = {}
            for label, source_suffix in (
                    ("primary", self.uncertainty),
                    ("cascade", f"cascade-{self.uncertainty}")):
                tables = _read_production_tables(
                    os.path.join(
                        self.particle_path,
                        f"DoubleDistr-DP-{source_suffix}.txt"),
                    os.path.join(
                        self.particle_path,
                        f"Emax-DP-{source_suffix}.txt"),
                    os.path.join(
                        self.particle_path,
                        f"Total-yield-DP-{source_suffix}.txt"),
                )
                distribution, maximum_energy, yield_data, yield_function, limits = tables
                source_rows.append(
                    (label, distribution, maximum_energy, yield_function))
                source_limits.append(limits)
                source_yield_data[label] = yield_data
            self.Distr = ProductionMixture(source_rows)
            self.Energy_distr = None
            self.Yield_data = source_yield_data
            self.production_component_labels = ("primary", "cascade")
        else:
            distribution_file_path = os.path.join(
                self.particle_path, f"DoubleDistr-DP-{suffix}.txt")
            energy_file_path = os.path.join(
                self.particle_path, f"Emax-DP-{suffix}.txt")
            yield_path = os.path.join(
                self.particle_path, f"Total-yield-DP-{suffix}.txt")
            self.Distr = pd.read_csv(
                distribution_file_path, header=None, sep="\t")
            self.Energy_distr = pd.read_csv(
                energy_file_path, header=None, sep="\t")
            self.Yield_data = pd.read_csv(
                yield_path, header=None, sep="\t")
        self.ctau_data = pd.read_csv(ctau_path, header=None, sep="\t")

        mass_ctau = self.ctau_data.iloc[:, 0].to_numpy()
        ctau_values = self.ctau_data.iloc[:, 1].to_numpy()
        self.ctau_interpolator = RegularGridInterpolator((mass_ctau,), ctau_values, bounds_error=True)

        if self.dp_production_mode != "combined":
            mass_yield = self.Yield_data.iloc[:, 0].to_numpy()
            yield_values = self.Yield_data.iloc[:, 1].to_numpy()
            self.yield_interpolator = RegularGridInterpolator(
                (mass_yield,), yield_values, bounds_error=True)

        DP_decay = pd.read_json(decay_json_path)
        self.decayChannels = DP_decay.iloc[:, 0].to_numpy()
        self.PDGs = DP_decay.iloc[:, 1].apply(np.array).to_numpy()
        self.BrRatios = DP_decay.iloc[:, 2].to_numpy()

        self.Matrix_elements_raw = DP_decay.iloc[:, -1].to_numpy()

        # Compile matrix elements
        self.Matrix_elements = self.compile_matrix_elements(self.Matrix_elements_raw)

        self.get_ctau = lambda m: self.ctau_interpolator([m])[0]
        if self.dp_production_mode == "combined":
            self.get_total_yield = self.Distr.total_yield
        else:
            self.get_total_yield = lambda m: self.yield_interpolator([m])[0]
        self.get_Br = self.setup_br_interpolators(self.BrRatios)
        self.get_distribution = lambda m: self.Distr
        self.get_MatrixElements = lambda m: self.Matrix_elements

        # define tabulated range
        if self.dp_production_mode == "combined":
            _set_combined_tabulated_range(
                self, source_limits, self.ctau_data)
        else:
            self.define_tabulated_range_nonHNL(
                mass_yield, yield_values, self.Distr, self.ctau_data)

        # Print the matrix elements table
        #self.print_matrix_elements()

    def import_HNL(self):
        (
            self.decayChannels,
            self.PDGs,
            self.BrRatios_raw,
            self.Matrix_elements_raw,
            self.decayWidthData,
            self.yieldData,
            self.massDistrData,
            self.DistrDataFrames
        ) = HNLmerging.load_data((
            os.path.join(self.particle_path, "HNL-decay.json"),
            os.path.join(self.particle_path, "HNLdecayWidth.dat"),
            os.path.join(self.particle_path, "Total-yield-HNL-e.txt"),
            os.path.join(self.particle_path, "Total-yield-HNL-mu.txt"),
            os.path.join(self.particle_path, "Total-yield-HNL-tau.txt"),
            os.path.join(self.particle_path, "DoubleDistr-HNL-mixing-e.txt"),
            os.path.join(self.particle_path, "DoubleDistr-HNL-mixing-mu.txt"),
            os.path.join(self.particle_path, "DoubleDistr-HNL-mixing-tau.txt")
        ))

        # HNLmerging has already verified these digests against the installed
        # files.  Expose the signed source id so raw and matched output names
        # can carry the same rate-table provenance.
        with open(os.path.join(self.particle_path, "HNL-source.json")) as src:
            hnl_source_metadata = json.load(src)
        source_block = hnl_source_metadata["source"]
        self.hnl_decay_source_sha256 = source_block["decay_json_sha256"]
        self.hnl_width_source_sha256 = source_block["width_table_sha256"]

        energy_file_path = os.path.join(self.particle_path, "Emax-HNL.txt")
        self.Energy_distr = pd.read_csv(energy_file_path, header=None, sep="\t")

        Ue2, Umu2, Utau2 = self.MixingPatternArray
        self.DWe_func, self.DWmu_func, self.DWtau_func = HNLmerging.get_decay_width_interpolators(self.decayWidthData)
        self.yield_e_func, self.yield_mu_func, self.yield_tau_func = HNLmerging.get_yield_interpolators(self.yieldData)

        self.get_Br = HNLmerging.get_BrMerged_func(
            self.BrRatios_raw, self.decayWidthData, self.MixingPatternArray,
            self.PDGs)
        self.get_distribution = HNLmerging.get_distribution_func(self.massDistrData, self.MixingPatternArray, self.yieldData, self.DistrDataFrames)
        self.get_ctau = HNLmerging.get_ctau_func(self.decayWidthData, self.MixingPatternArray)
        self.get_total_yield = HNLmerging.get_yield_func(self.yieldData, self.MixingPatternArray)

        self.func_e, self.func_mu, self.func_tau = HNLmerging.get_MatrixElements_funcs(self.Matrix_elements_raw)

        def get_MatrixElements(m):
            return self.create_Msquared_triple_functions(
                self.func_e, self.func_mu, self.func_tau,
                Ue2, Umu2, Utau2,
                self.DWe_func, self.DWmu_func, self.DWtau_func
            )
        self.get_MatrixElements = get_MatrixElements

        # define tabulated range for HNL
        self.define_tabulated_range_HNL()

        # Print the matrix elements table
        #self.print_matrix_elements()

    def setup_br_interpolators(self, BrRatios_raw):
        self.Br_interpolators = []
        for br_data in BrRatios_raw:
            if isinstance(br_data, (float, int)):
                self.Br_interpolators.append((None, float(br_data)))
            else:
                arr = np.array(br_data, dtype=float)
                if arr.ndim == 1 and arr.size == 1:
                    val = float(arr.item())
                    self.Br_interpolators.append((None, val))
                else:
                    masses = arr[:,0]
                    brvals = arr[:,1]
                    interp = RegularGridInterpolator((masses,), brvals, bounds_error=False, fill_value=0.0)
                    self.Br_interpolators.append((interp, None))

        def get_Br(m):
            br_list = []
            for (interp, cval) in self.Br_interpolators:
                if interp is not None:
                    val = interp([m])[0]
                else:
                    val = cval
                br_list.append(val)
            return np.array(br_list, dtype=float)
        return get_Br

    def compile_matrix_elements(self, matrix_elements_raw):
        """
        Compiles matrix element expressions into callable functions and stores symbolic expressions.
        """
        # exHad 0.2.0 writes daughter masses symbolically (m1/m2/m3) instead of
        # freezing them as decimals, so every compiled matrix element takes
        # six arguments: (mLLP, E_1, E_3, m1, m2, m3).  The three-body sampler
        # supplies the row's daughter masses; rows that do not mention them
        # simply ignore the extra arguments.
        mLLP, E_1, E_3 = sp.symbols('mLLP E_1 E_3')
        m1_s, m2_s, m3_s = sp.symbols('m1 m2 m3')
        signature = (mLLP, E_1, E_3, m1_s, m2_s, m3_s)
        local_dict = {
            'E_1': E_1,
            'E_3': E_3,
            'mLLP': mLLP,
            'm1': m1_s,
            'm2': m2_s,
            'm3': m3_s,
            # Mathematica UnitStep is SymPy's Heaviside. Its value exactly at
            # zero is immaterial for continuous phase-space sampling.
            'UnitStep': sp.Heaviside,
            'Symbol': sp.Symbol,
            'Float': float,
            'Integer': int
        }

        compiled_expressions = []
        for expr_str in matrix_elements_raw:
            if expr_str not in [None, "", "-"]:
                if expr_str.strip() == "1.":
                    # Define a function that returns 1.0
                    func = lambda m, e1, e3, mass1=0., mass2=0., mass3=0.: 1.0
                    compiled_expressions.append(func)
                    self.Matrix_elements_expr.append("1.0")
                else:
                    # Replace '***' with 'e' to handle scientific notation
                    expr_str_corrected = expr_str.replace('***', 'e')
                    expr_str_corrected = expr_str_corrected.replace('\\\\/', '/')
                    expr_str_corrected = expr_str_corrected.replace('E1','E_1').replace('E3','E_3')

                    # Remove 'Symbol', 'Float', and 'Integer' wrappers using regex
                    expr_str_corrected = re.sub(r"Symbol\(\s*'([^']+)'\s*\)", r"\1", expr_str_corrected)
                    expr_str_corrected = re.sub(r"Float\(\s*'([^']+)'\s*\)", r"\1", expr_str_corrected)
                    expr_str_corrected = re.sub(r"Integer\(\s*(\d+)\s*\)", r"\1", expr_str_corrected)

                    try:
                        # Parse the corrected expression with local variables
                        expr = sp.sympify(expr_str_corrected, locals=local_dict)
                    except Exception as e:
                        raise ValueError(f"Failed to sympify expression: {expr_str_corrected}") from e
                    # A constant expression sympifies to a plain number, which
                    # has no free_symbols attribute.
                    unbound = getattr(expr, 'free_symbols', set()) - set(signature)
                    if unbound:
                        raise ValueError(
                            "Matrix element contains unbound symbols %s; the parser "
                            "binds only mLLP, E_1, E_3, m1, m2, m3."
                            % sorted(str(s) for s in unbound))
                    func = sp.lambdify(signature, expr, 'numpy')
                    compiled_expressions.append(func)
                    self.Matrix_elements_expr.append(str(expr))
            else:
                compiled_expressions.append(None)
                self.Matrix_elements_expr.append("None")
        return compiled_expressions

    def create_Msquared_functions(self, func_elems):
        Msquared_list = []
        for f_ in func_elems:
            def MsqFactory(f_):
                def Msquared3BodyLLP(m_val, E_1_val, E_3_val,
                                     m1_val=0., m2_val=0., m3_val=0.):
                    if f_ is not None:
                        return f_(m_val, E_1_val, E_3_val, m1_val, m2_val, m3_val)
                    else:
                        return 0.0
                return Msquared3BodyLLP
            Msquared_list.append(MsqFactory(f_))
        return Msquared_list

    def create_Msquared_triple_functions(self, func_e, func_mu, func_tau, Ue2, Umu2, Utau2, DWe_func, DWmu_func, DWtau_func):
        Msquared_list = []
        for fe_, fmu_, ftau_ in zip(func_e, func_mu, func_tau):
            def MsqFactory(fe_, fmu_, ftau_):
                def Msquared3BodyLLP(m_val, E_1_val, E_3_val,
                                     m1_val=0., m2_val=0., m3_val=0.):
                    args = (m_val, E_1_val, E_3_val, m1_val, m2_val, m3_val)
                    val_e = fe_(*args) if fe_ is not None else 0.0
                    val_mu = fmu_(*args) if fmu_ is not None else 0.0
                    val_tau = ftau_(*args) if ftau_ is not None else 0.0
                    DWe = DWe_func(m_val)
                    DWmu = DWmu_func(m_val)
                    DWtau = DWtau_func(m_val)
                    return Ue2 * DWe * val_e + Umu2 * DWmu * val_mu + Utau2 * DWtau * val_tau
                return Msquared3BodyLLP
            Msquared_list.append(MsqFactory(fe_, fmu_, ftau_))
        return Msquared_list

    def print_matrix_elements(self):
        """
        Prints a table of decay channels and their corresponding symbolic matrix elements.
        """
        print("\nDecay Channels and Their Symbolic Matrix Elements:")
        df = pd.DataFrame({
            'Channel': self.decayChannels,
            'Mprocess(channel)': self.Matrix_elements_expr
        })
        print(df.to_string(index=False))
