# initLLP.py
import os
import re  # Added for regex operations
import numpy as np
import pandas as pd
from funcs import HNLmerging
from scipy.interpolate import RegularGridInterpolator
import sympy as sp

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

    def __init__(self, mass, particle_selection, mixing_pattern=None, uncertainty=None):
        self.main_folder = "./Distributions"
        self.LLP_name = particle_selection['LLP_name']
        self.mass = mass
        self.particle_path = particle_selection['particle_path']
        self.MixingPatternArray = mixing_pattern if mixing_pattern is not None else None
        self.uncertainty = uncertainty if self.LLP_name == "Dark-photons" else None
        self.Matrix_elements = None
        self.Matrix_elements_expr = []  # To store symbolic expressions

        self.import_particle()
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

    def compute_mass_dependent_properties(self):
        if self.mass is None:
            raise ValueError("Mass must be set before computing mass-dependent properties.")
        if "Scalar" in self.LLP_name:
            self.compute_mass_dependent_properties_scalars()
        elif self.LLP_name == "HNL":
            self.compute_mass_dependent_properties_HNL()
        elif self.LLP_name == "ALP-photon":
            self.compute_mass_dependent_properties_ALP_photon()
        elif self.LLP_name == "Dark-photons":
            self.compute_mass_dependent_properties_dark_photons()
        else:
            raise ValueError("Unknown LLP name.")

    def compute_mass_dependent_properties_scalars(self):
        self.BrRatios_distr = self.get_Br(self.mass)
        self.c_tau_int = self.get_ctau(self.mass)
        self.Yield = self.get_total_yield(self.mass)

    def compute_mass_dependent_properties_ALP_photon(self):
        self.BrRatios_distr = self.get_Br(self.mass)
        self.c_tau_int = self.get_ctau(self.mass)
        self.Yield = self.get_total_yield(self.mass)

    def compute_mass_dependent_properties_dark_photons(self):
        self.BrRatios_distr = self.get_Br(self.mass)
        self.c_tau_int = self.get_ctau(self.mass)
        self.Yield = self.get_total_yield(self.mass)

    def compute_mass_dependent_properties_HNL(self):
        self.c_tau_int = self.get_ctau(self.mass)
        self.Yield = self.get_total_yield(self.mass)
        self.BrRatios_distr = self.get_Br(self.mass)
        self.Matrix_elements = self.get_MatrixElements(self.mass)
        self.Distr = self.get_distribution(self.mass)

    def import_particle(self):
        if "Scalar" in self.LLP_name:
            self.import_scalars()
        elif self.LLP_name == "HNL":
            if self.MixingPatternArray is None:
                raise ValueError("Mixing pattern must be provided for HNL.")
            self.import_HNL()
        elif self.LLP_name == "ALP-photon":
            self.import_ALP_photon()
        elif self.LLP_name == "Dark-photons":
            if self.uncertainty is None:
                raise ValueError("Uncertainty must be provided for Dark-photons.")
            self.import_dark_photons()
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
        ctau_path = os.path.join(self.particle_path, "ctau-Scalar.txt")
        decay_json_path = os.path.join(self.particle_path, "HLS-decay.json")

        self.Distr = pd.read_csv(distribution_file_path, header=None, sep="\t")
        self.Energy_distr = pd.read_csv(energy_file_path, header=None, sep="\t")
        self.Yield_data = pd.read_csv(yield_path, header=None, sep="\t")
        self.ctau_data = pd.read_csv(ctau_path, header=None, sep="\t")

        mass_ctau = self.ctau_data.iloc[:, 0].to_numpy()
        ctau_values = self.ctau_data.iloc[:, 1].to_numpy()
        self.ctau_interpolator = RegularGridInterpolator((mass_ctau,), ctau_values, bounds_error=False, fill_value=None)

        mass_yield = self.Yield_data.iloc[:, 0].to_numpy()
        yield_values = self.Yield_data.iloc[:, 1].to_numpy()
        self.yield_interpolator = RegularGridInterpolator((mass_yield,), yield_values, bounds_error=False, fill_value=None)

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
        distribution_file_path = os.path.join(self.particle_path, "DoubleDistr-ALP-photon.txt")
        energy_file_path = os.path.join(self.particle_path, "Emax-ALP-photon.txt")
        yield_path = os.path.join(self.particle_path, "Total-yield-ALP-photon.txt")
        ctau_path = os.path.join(self.particle_path, "ctau-ALP.txt")
        decay_json_path = os.path.join(self.particle_path, "ALP-photon-decay.json")

        self.Distr = pd.read_csv(distribution_file_path, header=None, sep="\t")
        self.Energy_distr = pd.read_csv(energy_file_path, header=None, sep="\t")
        self.Yield_data = pd.read_csv(yield_path, header=None, sep="\t")
        self.ctau_data = pd.read_csv(ctau_path, header=None, sep="\t")

        mass_ctau = self.ctau_data.iloc[:, 0].to_numpy()
        ctau_values = self.ctau_data.iloc[:, 1].to_numpy()
        self.ctau_interpolator = RegularGridInterpolator((mass_ctau,), ctau_values, bounds_error=False, fill_value=None)

        mass_yield = self.Yield_data.iloc[:, 0].to_numpy()
        yield_values = self.Yield_data.iloc[:, 1].to_numpy()
        self.yield_interpolator = RegularGridInterpolator((mass_yield,), yield_values, bounds_error=False, fill_value=None)

        ALP_decay = pd.read_json(decay_json_path)
        self.decayChannels = ALP_decay.iloc[:, 0].to_numpy()
        self.PDGs = ALP_decay.iloc[:, 1].apply(np.array).to_numpy()
        self.BrRatios = ALP_decay.iloc[:, 2].to_numpy()

        self.Matrix_elements_raw = ALP_decay.iloc[:, -1].to_numpy()

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

    def import_dark_photons(self):
        distribution_file_path = os.path.join(self.particle_path, f"DoubleDistr-DP-{self.uncertainty}.txt")
        energy_file_path = os.path.join(self.particle_path, f"Emax-DP-{self.uncertainty}.txt")
        yield_path = os.path.join(self.particle_path, f"Total-yield-DP-{self.uncertainty}.txt")
        ctau_path = os.path.join(self.particle_path, "ctau-DP.txt")
        decay_json_path = os.path.join(self.particle_path, "DP-decay.json")

        self.Distr = pd.read_csv(distribution_file_path, header=None, sep="\t")
        self.Energy_distr = pd.read_csv(energy_file_path, header=None, sep="\t")
        self.Yield_data = pd.read_csv(yield_path, header=None, sep="\t")
        self.ctau_data = pd.read_csv(ctau_path, header=None, sep="\t")

        mass_ctau = self.ctau_data.iloc[:, 0].to_numpy()
        ctau_values = self.ctau_data.iloc[:, 1].to_numpy()
        self.ctau_interpolator = RegularGridInterpolator((mass_ctau,), ctau_values, bounds_error=False, fill_value=None)

        mass_yield = self.Yield_data.iloc[:, 0].to_numpy()
        yield_values = self.Yield_data.iloc[:, 1].to_numpy()
        self.yield_interpolator = RegularGridInterpolator((mass_yield,), yield_values, bounds_error=False, fill_value=None)

        DP_decay = pd.read_json(decay_json_path)
        self.decayChannels = DP_decay.iloc[:, 0].to_numpy()
        self.PDGs = DP_decay.iloc[:, 1].apply(np.array).to_numpy()
        self.BrRatios = DP_decay.iloc[:, 2].to_numpy()

        self.Matrix_elements_raw = DP_decay.iloc[:, -1].to_numpy()

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

        energy_file_path = os.path.join(self.particle_path, "Emax-HNL.txt")
        self.Energy_distr = pd.read_csv(energy_file_path, header=None, sep="\t")

        Ue2, Umu2, Utau2 = self.MixingPatternArray
        self.DWe_func, self.DWmu_func, self.DWtau_func = HNLmerging.get_decay_width_interpolators(self.decayWidthData)
        self.yield_e_func, self.yield_mu_func, self.yield_tau_func = HNLmerging.get_yield_interpolators(self.yieldData)

        self.get_Br = HNLmerging.get_BrMerged_func(self.BrRatios_raw, self.decayWidthData, self.MixingPatternArray)
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
        mLLP, E_1, E_3 = sp.symbols('mLLP E_1 E_3')
        local_dict = {
            'E_1': E_1,
            'E_3': E_3,
            'mLLP': mLLP,
            'Symbol': sp.Symbol,
            'Float': float,
            'Integer': int
        }

        compiled_expressions = []
        for expr_str in matrix_elements_raw:
            if expr_str not in [None, "", "-"]:
                if expr_str.strip() == "1.":
                    # Define a function that returns 1.0
                    func = lambda m, e1, e3: 1.0
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
                    func = sp.lambdify((mLLP, E_1, E_3), expr, 'numpy')
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
                def Msquared3BodyLLP(m_val, E_1_val, E_3_val):
                    if f_ is not None:
                        return f_(m_val, E_1_val, E_3_val)
                    else:
                        return 0.0
                return Msquared3BodyLLP
            Msquared_list.append(MsqFactory(f_))
        return Msquared_list

    def create_Msquared_triple_functions(self, func_e, func_mu, func_tau, Ue2, Umu2, Utau2, DWe_func, DWmu_func, DWtau_func):
        Msquared_list = []
        for fe_, fmu_, ftau_ in zip(func_e, func_mu, func_tau):
            def MsqFactory(fe_, fmu_, ftau_):
                def Msquared3BodyLLP(m_val, E_1_val, E_3_val):
                    val_e = fe_(m_val, E_1_val, E_3_val) if fe_ is not None else 0.0
                    val_mu = fmu_(m_val, E_1_val, E_3_val) if fmu_ is not None else 0.0
                    val_tau = ftau_(m_val, E_1_val, E_3_val) if ftau_ is not None else 0.0
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

