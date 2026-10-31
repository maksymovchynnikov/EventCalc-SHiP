"""Cache the iteration-8 p>1 GeV response as merged accepted vertex cells."""
import numpy as np
from numba import njit

from ._na62_cutflow import ORIGIN,ZMAG,FIELD,PLANE,decisions,transport,ElectronFV
from eventcalc.sensitivity import PreparedAcceptance,Geometry,Selection


@njit(cache=True)
def accepted_cells(p,d,ids,q,w,o,mass,groups,vertices,nz):
    lower=[];upper=[];weights=[];selected=[];indices=[]
    dz=75./nz
    for i in range(len(p)):
        if p[i,2]<=0:continue
        tracks=0;charge=0.;supported=True
        for j in range(len(ids[i])):
            code=abs(ids[i,j])
            if code in (0,12,14,16,18,4900101,4900102):continue
            if code not in (211,321,22):supported=False
            if q[i,j]!=0:
                tracks+=1;charge+=q[i,j]
                if np.sqrt(np.dot(d[i,j,:3],d[i,j,:3]))<=1. or d[i,j,2]<=0:supported=False
        if not supported or tracks!=2 or charge!=0:continue
        beta=p[i,:2]/p[i,2];alpha=o[i,:2]-o[i,2]*beta
        a,b,sa,sb,ga,gb,ok=transport(alpha,beta,d[i],q[i])
        start=-1.
        for iz in range(nz):
            lo=105.+iz*dz;hi=lo+dz
            flags,ng,allg=decisions((lo+hi)/2,alpha,beta,d[i],ids[i],q[i],a,b,sa,sb,ga,gb,ok,vertices,1.)
            if flags[8] and start<0:start=lo
            if start>=0 and (not flags[8] or iz==nz-1):
                end=hi if flags[8] else lo
                start=max(start,o[i,2])
                if end<=start:
                    start=-1.
                    continue
                factor=mass/p[i,2]
                lower.append((start-o[i,2])*factor);upper.append((end-o[i,2])*factor)
                weights.append(w[i]);selected.append(groups[i]);indices.append(i);start=-1.
    return (np.asarray(lower),np.asarray(upper),np.asarray(weights),np.asarray(selected,dtype=np.int64),np.asarray(indices,dtype=np.int64))


def prepare(batch,ids,groups,nz=128):
    if not isinstance(nz,int) or nz<2:raise ValueError('At least two vertex cells required')
    arrays,mass=batch.arrays();p,d,v,w,o=arrays
    values=accepted_cells(p,d,np.asarray(ids,np.int64),np.asarray(batch.charges,float),w,o,mass,
        np.asarray(groups,np.int64),np.asarray(ElectronFV.digitized(centre_xy=(0,0)).vertices),nz)
    geometry=Geometry(z_min=105,z_max=180,x_in=2,x_out=2,y_in=2,y_out=2,aperture_z=PLANE,aperture_x=2.3,aperture_y=2.3,
        magnet_z_min=ZMAG[0],magnet_z_max=ZMAG[1],field_integral=FIELD,target_z=ORIGIN[2])
    selections=tuple(Selection(f'hadronic_group_{g}',min_particles=0,energy_min=0) for g in range(6))
    return PreparedAcceptance(*values[:4],selections,geometry,event_index=values[4])
