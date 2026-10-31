"""Hadronic NA62 truth-level cutflow with numerical vertex integration.

Not a reconstructed analysis: final-plane proxy, perfect track finding/PID,
isolated photons and no cluster merging or mass-window classification.
"""
from pathlib import Path
import sys
import numpy as np
from numba import njit

from ._na62_leptons import ORIGIN,ZMAG,FIELD,PLANE
from ._na62_primitives import ElectronFV
from eventcalc.sensitivity import dipole_displacement

STRAW=183.
LKR=241.495  # SensCalc card; nominal front plane, not a verified 2021 alignment.
STAGES=('cylinder_volume','restricted_fv','two_opposite_hadron_tracks','track_p_threshold',
        'track_aperture_proxy','separation_straw','separation_lkr','photons_reconstructed','hadronic_pointing')


@njit(cache=True)
def fv_pass(z,x,y,vertices):
    if z<=vertices[0,0] or z>=vertices[4,0]:return False
    l,r=(1,2) if z<vertices[2,0] else (3,4)
    bound=vertices[l,1]+(z-vertices[l,0])*(vertices[r,1]-vertices[l,1])/(vertices[r,0]-vertices[l,0])
    return x*x+y*y<bound*bound


@njit(cache=True)
def hadron_pointing(x,y,z,momentum):
    pt2=momentum[0]**2+momentum[1]**2
    if pt2<=0:return False
    dx=x-ORIGIN[0];dy=y-ORIGIN[1]
    t=-(dx*momentum[0]+dy*momentum[1])/pt2
    ztax=z+t*momentum[2]
    cda2=(dx+t*momentum[0])**2+(dy+t*momentum[1])**2
    return ((ztax-23.)/23.)**2+cda2/.04**2<1.


@njit(cache=True)
def decisions(z,alpha,beta,d,ids,q,a,b,sa,sb,ga,gb,transport_ok,vertices,pmin):
    """One event at one vertex; a/b final track, s STRAW, g LKr."""
    flags=np.zeros(9,np.bool_);xy=alpha+z*beta
    if np.dot(xy,xy)>=1.:return flags,0,False
    flags[0]=True
    if not fv_pass(z,xy[0],xy[1],vertices):return flags,0,False
    flags[1]=True
    tracks=np.zeros(2,np.int64);nc=0;supported=True
    for j in range(len(ids)):
        code=abs(ids[j])
        if code==0 or code in (12,14,16,18,4900101,4900102):continue
        if code not in (211,321,22):supported=False
        if q[j]!=0:
            if nc<2:tracks[nc]=j
            nc+=1
    if not supported or nc!=2 or q[tracks[0]]*q[tracks[1]]>=0:return flags,0,False
    flags[2]=True
    for j in tracks:
        if np.sqrt(np.dot(d[j,:3],d[j,:3]))<=pmin:return flags,0,False
    flags[3]=True
    for j in tracks:
        hit=a[j]+z*b[j];r2=np.dot(hit,hit)
        if not transport_ok[j] or r2<=.08**2 or r2>=1.15**2:return flags,0,False
    flags[4]=True
    delta=sa[tracks[0]]-sa[tracks[1]]+z*(sb[tracks[0]]-sb[tracks[1]])
    if np.dot(delta,delta)<.02**2:return flags,0,False
    flags[5]=True
    delta=ga[tracks[0]]-ga[tracks[1]]+z*(gb[tracks[0]]-gb[tracks[1]])
    if np.dot(delta,delta)<.2**2:return flags,0,False
    flags[6]=True
    reconstructed=d[tracks[0],:3]+d[tracks[1],:3]
    ngamma=0;all_photons=True
    for j in range(len(ids)):
        if ids[j]!=22:continue
        hit=ga[j]+z*gb[j];r2=np.dot(hit,hit)
        detected=d[j,3]>5. and transport_ok[j] and r2>.08**2 and r2<1.15**2
        if detected:reconstructed+=d[j,:3];ngamma+=1
        else:all_photons=False
    # This is reconstruction, not a veto on undetected photons.
    flags[7]=True
    flags[8]=hadron_pointing(xy[0],xy[1],z,reconstructed)
    return flags,ngamma,all_photons


@njit(cache=True)
def transport(alpha,beta,d,q):
    k=len(d);a=np.zeros((k,2));b=np.zeros((k,2));sa=a.copy();sb=b.copy();ga=a.copy();gb=b.copy();ok=np.zeros(k,np.bool_)
    for j in range(k):
        if d[j,2]<=0:continue
        slope=d[j,:2]/d[j,2]
        dx,dy,good=dipole_displacement(d[j,:3],q[j],*ZMAG,FIELD,PLANE)
        if not good:continue
        a[j]=alpha+PLANE*slope+np.array([dx,dy]);b[j]=beta-slope
        sa[j]=alpha+STRAW*slope;sb[j]=beta-slope
        dx,dy,good=dipole_displacement(d[j,:3],q[j],*ZMAG,FIELD,LKR)
        if not good:continue
        ga[j]=alpha+LKR*slope+np.array([dx,dy]);gb[j]=beta-slope;ok[j]=True
    return a,b,sa,sb,ga,gb,ok


@njit(cache=True)
def integrate(p,d,ids,q,w,o,mass,ctaus,groups,vertices,pmins,nz):
    # [threshold scenario, lifetime, stage, terminal-state group]
    result=np.zeros((len(pmins),len(ctaus),9,6))
    photons=np.zeros((len(pmins),len(ctaus),6))
    full_photon_control=np.zeros((len(pmins),len(ctaus)))
    dz=75./nz
    for i in range(len(p)):
        if p[i,2]<=0:continue
        beta=p[i,:2]/p[i,2];alpha=o[i,:2]-o[i,2]*beta
        a,b,sa,sb,ga,gb,ok=transport(alpha,beta,d[i],q[i])
        for iz in range(nz):
            lo=105.+iz*dz;hi=lo+dz;z=(lo+hi)/2
            prob=np.zeros(len(ctaus))
            for c in range(len(ctaus)):
                rate=mass/p[i,2]/ctaus[c]
                prob[c]=w[i]*np.exp(-(lo-o[i,2])*rate)*(-np.expm1(-dz*rate))
            for s in range(len(pmins)):
                flags,ng,allg=decisions(z,alpha,beta,d[i],ids[i],q[i],a,b,sa,sb,ga,gb,ok,vertices,pmins[s])
                for c in range(len(ctaus)):
                    for stage in range(9):
                        if flags[stage]:result[s,c,stage,groups[i]]+=prob[c]
                    if flags[8]:
                        photons[s,c,min(ng,5)]+=prob[c]
                        if allg:full_photon_control[s,c]+=prob[c]
    return result,photons,full_photon_control


def evaluate(batch,ids,groups,ctaus,pmins=(1.,10.),nz=64):
    arrays,mass=batch.arrays();p,d,v,w,o=arrays
    return integrate(p,d,np.asarray(ids,np.int64),np.asarray(batch.charges,float),w,o,mass,
        np.asarray(ctaus,float),np.asarray(groups,np.int64),np.asarray(ElectronFV.digitized(centre_xy=(0,0)).vertices),np.asarray(pmins,float),nz)
