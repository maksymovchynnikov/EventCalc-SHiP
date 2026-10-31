"""Load existing decay tables without consulting production spectra or yields.

The interpolation and matrix-element routines remain owned by initLLP.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from . import ALPmerging, HNLmerging, coupling_conventions
from .alp_fermion import prepare_decay_rows, load_exhad_boundaries, build_br_interpolator


def load(llp):
    from .initLLP import _load_scalar_ctau_json, make_log_pchip_interpolator
    root=Path(llp.particle_path)
    name=llp.LLP_name
    llp.Distr=llp.Energy_distr=llp.Yield_data=None
    llp.get_total_yield=lambda mass: 0.  # never applied by the external decay adapter
    llp.get_distribution=lambda mass: None
    llp.decay_input_paths=[]
    if name=='HNL':
        if llp.MixingPatternArray is None: raise ValueError('HNL requires mixing_pattern')
        (llp.decayChannels,llp.PDGs,llp.BrRatios_raw,llp.Matrix_elements_raw,llp.decayWidthData)=HNLmerging.load_data(
            (str(root/'HNL-decay.json'),str(root/'HNLdecayWidth.dat')),decay_only=True)
        source=json.loads((root/'HNL-source.json').read_text())['source']
        llp.decay_input_paths=[root/'HNL-decay.json',root/'HNLdecayWidth.dat',root/'HNL-source.json']
        llp.hnl_decay_source_sha256=source['decay_json_sha256']
        llp.hnl_width_source_sha256=source['width_table_sha256']
        llp.DWe_func,llp.DWmu_func,llp.DWtau_func=HNLmerging.get_decay_width_interpolators(llp.decayWidthData)
        llp.get_Br=HNLmerging.get_BrMerged_func(llp.BrRatios_raw,llp.decayWidthData,llp.MixingPatternArray,llp.PDGs)
        llp.get_ctau=HNLmerging.get_ctau_func(llp.decayWidthData,llp.MixingPatternArray)
        llp.func_e,llp.func_mu,llp.func_tau=HNLmerging.get_MatrixElements_funcs(llp.Matrix_elements_raw)
        llp.get_MatrixElements=lambda m: llp.create_Msquared_triple_functions(
            llp.func_e,llp.func_mu,llp.func_tau,*llp.MixingPatternArray,
            llp.DWe_func,llp.DWmu_func,llp.DWtau_func,mass=m)
        masses=np.asarray(llp.decayWidthData[0],float)
        llp.m_min_tabulated=float(masses.min());llp.m_max_tabulated=float(masses.max())
        return
    if 'Scalar' in name:
        root=root.parent/'Scalar-mixing'
        decay_path=root/f'BrRatio-Scalar-{llp.scalar_lifetime}.json'
        ctau_path=root/f'ctau-Scalar-{llp.scalar_lifetime}.txt'
        ctau=_load_scalar_ctau_json(str(ctau_path))
    else:
        stem='DP' if name=='Dark-photons' else name
        decay_path=root/f'{stem}-decay.json'
        ctau=None if name=='ALP-mixed' else pd.read_csv(root/f'ctau-{stem}.txt',header=None,sep='\t')
    rows=json.loads(decay_path.read_text())
    llp.decay_input_paths=[decay_path]
    if name!='ALP-mixed':llp.decay_input_paths.append(ctau_path if 'Scalar' in name else root/f'ctau-{stem}.txt')
    if name=='ALP-fermion':
        llp.decay_input_paths.extend([root/'coupling.json',root/'exhad.json'])
    if name=='ALP-fermion':
        llp.coupling_metadata=coupling_conventions.load_coupling_metadata(str(root))
        labels,pdgs,brs,matrices=prepare_decay_rows(rows)
        llp.decayChannels=np.array(labels,dtype=object)
        llp.PDGs=np.empty(len(rows),dtype=object);llp.PDGs[:]=[np.array(x) for x in pdgs]
        llp.BrRatios=np.empty(len(rows),dtype=object);llp.BrRatios[:]=brs
        low,charm=load_exhad_boundaries(str(root))
        interp=build_br_interpolator(llp.decayChannels,llp.PDGs,llp.BrRatios,low,charm,ctau.itertuples(index=False,name=None))
        llp.get_Br=lambda m: np.array(interp(m),float)
    else:
        frame=pd.DataFrame(rows)
        llp.decayChannels=frame.iloc[:,0].to_numpy()
        llp.PDGs=frame.iloc[:,1].apply(np.array).to_numpy()
        llp.BrRatios=frame.iloc[:,2].to_numpy()
        matrices=['1.']*len(rows) if 'Scalar' in name else frame.iloc[:,-1].to_numpy()
        llp.get_Br=llp.setup_br_interpolators(llp.BrRatios)
    llp.Matrix_elements_raw=np.array(matrices,dtype=object)
    llp.Matrix_elements=llp.compile_matrix_elements(llp.Matrix_elements_raw)
    llp.get_MatrixElements=lambda mass: llp.Matrix_elements
    llp.ctau_data=ctau
    if name=='ALP-mixed':
        # Mixed operators only change production here; decay is diphoton.
        llp.get_ctau=ALPmerging.diphoton_ctau_coefficient
        low,high=0.,np.inf  # analytic diphoton decay, no tabulated production domain
    else:
        masses=ctau.iloc[:,0].to_numpy();values=ctau.iloc[:,1].to_numpy()
        low=masses.min();high=masses.max()
        llp.ctau_interpolator=RegularGridInterpolator((masses,),values,bounds_error=True)
        llp.get_ctau=(make_log_pchip_interpolator(masses,values) if name=='ALP-SU2L'
                      else lambda mass: llp.ctau_interpolator([mass])[0])
    llp.m_min_tabulated=float(low);llp.m_max_tabulated=float(high)
