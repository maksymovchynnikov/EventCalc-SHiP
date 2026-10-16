"""Compact compatibility tables for EventCalc's existing kinematics interface.

The weighted ProductionModel API retains the original collision positions.
These tables marginalize positions and azimuth for the older table interface.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def load(category='combined'):
    root=Path(__file__).parent/'data/tables'
    manifest=json.loads((root/'manifest.json').read_text())
    if manifest['target']['z_origin']!='front face':raise ValueError('Incompatible production coordinates')
    parts=('primary','cascade') if category=='combined' else (category,)
    density=None;yield_=None
    for part in parts:
        with np.load(root/(part+'.npz')) as z:
            m=z['mass'];theta=z['theta'];energy=z['energy'];y=z['yield_per_pot_epsilon2'];d=z['density']
            if density is None:
                mass=m.copy();angles=theta.copy();energies=energy.copy();density=d*y[:,None,None];yield_=y.copy()
            else:
                for a,b in ((m,mass),(theta,angles),(energy,energies)):np.testing.assert_array_equal(a,b)
                density+=d*y[:,None,None];yield_+=y
    density=np.divide(density,yield_[:,None,None],out=np.zeros_like(density),where=yield_[:,None,None]>0)
    mm,tt,ee=np.meshgrid(mass,angles,energies,indexing='ij')
    distribution=pd.DataFrame(np.column_stack([mm.ravel(),tt.ravel(),ee.ravel(),density.ravel()]))
    m2,t2=np.meshgrid(mass,angles,indexing='ij')
    maximum_energy=pd.DataFrame(np.column_stack([m2.ravel(),t2.ravel(),np.full(m2.size,energies[-1])]))
    yields=pd.DataFrame(np.column_stack([mass,yield_]))
    interp=RegularGridInterpolator((mass,),yield_,bounds_error=True)
    def total(m):return float(interp([float(m)])[0])
    return distribution,maximum_energy,yields,total,(float(mass[0]),float(mass[-1]))
