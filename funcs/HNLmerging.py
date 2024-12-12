# HNLmerging.py
import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import sympy as sp

def load_data(paths):
    (decay_json_path,
     decay_width_path,
     yield_e_path,
     yield_mu_path,
     yield_tau_path,
     distrHNL_e_path,
     distrHNL_mu_path,
     distrHNL_tau_path) = paths

    HNL_decay = pd.read_json(decay_json_path)
    decayChannels = HNL_decay.iloc[:, 0].to_numpy()
    PDGs = HNL_decay.iloc[:, 1].apply(np.array).to_numpy()

    Br_e = HNL_decay.iloc[:, 2].to_numpy()
    Br_mu = HNL_decay.iloc[:, 3].to_numpy()
    Br_tau = HNL_decay.iloc[:, 4].to_numpy()
    BrRatios = np.array([Br_e, Br_mu, Br_tau], dtype=object)

    Matrix_elements = np.column_stack((
        HNL_decay.iloc[:, 5],
        HNL_decay.iloc[:, 6],
        HNL_decay.iloc[:, 7]
    ))

    HNL_decay_width = pd.read_csv(decay_width_path, header=None, sep="\t")
    decay_mass = np.array(HNL_decay_width.iloc[:, 0])
    DW_e = np.array(HNL_decay_width.iloc[:, 1])
    DW_mu = np.array(HNL_decay_width.iloc[:, 2])
    DW_tau = np.array(HNL_decay_width.iloc[:, 3])
    decayWidthData = np.array([decay_mass, DW_e, DW_mu, DW_tau], dtype=object)

    HNL_yield_e = pd.read_csv(yield_e_path, header=None, sep="\t")
    HNL_yield_mu = pd.read_csv(yield_mu_path, header=None, sep="\t")
    HNL_yield_tau = pd.read_csv(yield_tau_path, header=None, sep="\t")

    yield_mass = np.array(HNL_yield_e.iloc[:, 0])
    ye = np.array(HNL_yield_e.iloc[:, 1])
    ymu = np.array(HNL_yield_mu.iloc[:, 1])
    ytau = np.array(HNL_yield_tau.iloc[:, 1])
    yieldData = np.array([yield_mass, ye, ymu, ytau], dtype=object)

    DistrHNL_e = pd.read_csv(distrHNL_e_path, header=None, sep="\t")
    DistrHNL_mu = pd.read_csv(distrHNL_mu_path, header=None, sep="\t")
    DistrHNL_tau = pd.read_csv(distrHNL_tau_path, header=None, sep="\t")

    DistrDataFrames = (DistrHNL_e, DistrHNL_mu, DistrHNL_tau)
    massDistrData = np.array([
        np.unique(DistrHNL_e.iloc[:,0]),
        np.unique(DistrHNL_mu.iloc[:,0]),
        np.unique(DistrHNL_tau.iloc[:,0])
    ], dtype=object)

    return (decayChannels, PDGs, BrRatios, Matrix_elements, decayWidthData, yieldData, massDistrData, DistrDataFrames)

def get_decay_width_interpolators(decayWidthData):
    decay_mass, DW_e, DW_mu, DW_tau = decayWidthData
    def DWe(m):
        return regular_interpolator(m, decay_mass, DW_e)
    def DWmu(m):
        return regular_interpolator(m, decay_mass, DW_mu)
    def DWtau(m):
        return regular_interpolator(m, decay_mass, DW_tau)
    return DWe, DWmu, DWtau

def get_yield_interpolators(yieldData):
    yield_mass, ye, ymu, ytau = yieldData
    def Ye(m):
        return regular_interpolator(m, yield_mass, ye)
    def Ymu(m):
        return regular_interpolator(m, yield_mass, ymu)
    def Ytau(m):
        return regular_interpolator(m, yield_mass, ytau)
    return Ye, Ymu, Ytau

def regular_interpolator(point, axis, distr):
    interp = RegularGridInterpolator((axis,), distr, bounds_error=False, fill_value=0.0)
    if np.isscalar(point):
        return interp([point])[0]
    else:
        return interp(point)

def get_BrMerged_func(BrRatios, decayWidthData, MixingPatternArray):
    Ue2, Umu2, Utau2 = MixingPatternArray
    decay_mass, DW_e, DW_mu, DW_tau = decayWidthData
    channel_count = BrRatios[0].shape[0]

    # For each channel, we have arrays for Br_e, Br_mu, Br_tau
    interps_e = []
    interps_mu = []
    interps_tau = []
    for i in range(channel_count):
        Br_e_arr = np.array(BrRatios[0][i])
        Br_mu_arr = np.array(BrRatios[1][i])
        Br_tau_arr = np.array(BrRatios[2][i])

        interp_e = RegularGridInterpolator((Br_e_arr[:,0],), Br_e_arr[:,1], bounds_error=False, fill_value=0.0)
        interp_mu = RegularGridInterpolator((Br_mu_arr[:,0],), Br_mu_arr[:,1], bounds_error=False, fill_value=0.0)
        interp_tau = RegularGridInterpolator((Br_tau_arr[:,0],), Br_tau_arr[:,1], bounds_error=False, fill_value=0.0)

        interps_e.append(interp_e)
        interps_mu.append(interp_mu)
        interps_tau.append(interp_tau)

    def get_BrMerged(m):
        DWe_val = regular_interpolator(m, decay_mass, DW_e)
        DWmu_val = regular_interpolator(m, decay_mass, DW_mu)
        DWtau_val = regular_interpolator(m, decay_mass, DW_tau)
        denominator = Ue2*DWe_val + Umu2*DWmu_val + Utau2*DWtau_val
        if denominator == 0:
            return np.zeros(channel_count)
        BrMerged = []
        for i in range(channel_count):
            br_e_val = interps_e[i]([m])[0]
            br_mu_val = interps_mu[i]([m])[0]
            br_tau_val = interps_tau[i]([m])[0]
            numerator = Ue2*DWe_val*br_e_val + Umu2*DWmu_val*br_mu_val + Utau2*DWtau_val*br_tau_val
            BrMerged.append(numerator/denominator)
        return np.array(BrMerged)
    return get_BrMerged

def get_ctau_func(decayWidthData, MixingPatternArray):
    Ue2, Umu2, Utau2 = MixingPatternArray
    decay_mass, DW_e, DW_mu, DW_tau = decayWidthData
    def get_ctau(m):
        DWe_val = regular_interpolator(m, decay_mass, DW_e)
        DWmu_val = regular_interpolator(m, decay_mass, DW_mu)
        DWtau_val = regular_interpolator(m, decay_mass, DW_tau)
        total_dw = Ue2*DWe_val + Umu2*DWmu_val + Utau2*DWtau_val
        if total_dw == 0:
            return 1e100
        return 1.973269788e-16 / total_dw
    return get_ctau

def get_yield_func(yieldData, MixingPatternArray):
    Ue2, Umu2, Utau2 = MixingPatternArray
    yield_mass, ye, ymu, ytau = yieldData
    def get_total_yield(m):
        ye_val = regular_interpolator(m, yield_mass, ye)
        ymu_val = regular_interpolator(m, yield_mass, ymu)
        ytau_val = regular_interpolator(m, yield_mass, ytau)
        return Ue2*ye_val + Umu2*ymu_val + Utau2*ytau_val
    return get_total_yield

def get_distribution_func(massDistrData, MixingPatternArray, yieldData, DistrDataFrames):
    Ue2, Umu2, Utau2 = MixingPatternArray
    DistrHNL_e, DistrHNL_mu, DistrHNL_tau = DistrDataFrames
    yield_mass, ye, ymu, ytau = yieldData

    def get_distribution(m):
        ye_val = regular_interpolator(m, yield_mass, ye)
        ymu_val = regular_interpolator(m, yield_mass, ymu)
        ytau_val = regular_interpolator(m, yield_mass, ytau)
        total_y = Ue2*ye_val + Umu2*ymu_val + Utau2*ytau_val
        if total_y == 0:
            total_y = 1e-100
        f_e_scaled = DistrHNL_e[3]* (Ue2*ye_val/total_y)
        f_mu_scaled = DistrHNL_mu[3]*(Umu2*ymu_val/total_y)
        f_tau_scaled = DistrHNL_tau[3]*(Utau2*ytau_val/total_y)
        merged = f_e_scaled.add(f_mu_scaled, fill_value=0).add(f_tau_scaled, fill_value=0)
        Merged_dataframe = DistrHNL_e.copy()
        Merged_dataframe[3] = merged
        return Merged_dataframe
    return get_distribution

def get_MatrixElements_funcs(Matrix_elements_raw):
    mLLP, E_1, E_3 = sp.symbols('mLLP E_1 E_3')
    def parse_expr(expr):
        if expr not in [None, "", "-"]:
            expr_str = str(expr).replace('***','e')
            expr_str = expr_str.replace('E1','E_1').replace('E3','E_3')
            fexpr = sp.sympify(expr_str)
            return sp.lambdify((mLLP,E_1,E_3), fexpr, 'numpy')
        return None

    func_e = []
    func_mu = []
    func_tau = []

    for i in range(Matrix_elements_raw.shape[0]):
        fe_ = parse_expr(Matrix_elements_raw[i,0])
        fmu_ = parse_expr(Matrix_elements_raw[i,1])
        ftau_ = parse_expr(Matrix_elements_raw[i,2])
        func_e.append(fe_)
        func_mu.append(fmu_)
        func_tau.append(ftau_)

    return func_e, func_mu, func_tau

