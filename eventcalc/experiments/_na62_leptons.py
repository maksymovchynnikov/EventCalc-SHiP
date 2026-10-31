"""Analytic z intervals for the explicit NA62 prototype response.

Circle/linear-radius boundaries are quadratic along a straight parent ray.
No vertex Monte Carlo or lifetime-dependent selection sampling is used.
"""
from pathlib import Path
import sys
import numpy as np

from ._na62_primitives import ElectronFV,pointing,signal_region,dilepton_selection
from eventcalc.sensitivity import Geometry,Selection,PreparedAcceptance,dipole_displacement

NAMES=('volume_ee','selected_ee','detector_ee','volume_mumu','selected_mumu','detector_mumu')
ORIGIN=np.array([0.,-.022,23.07])
ZMAG=(196.35,197.65)
FIELD=.6928*(ZMAG[1]-ZMAG[0])
PLANE=243.85


def circle_polynomial(a,b,r0,r1=0.):
    """norm(a+b*z)^2 - (r0+r1*z)^2, ascending coefficients."""
    return np.stack([np.sum(a*a,axis=-1)-r0*r0,
                     2*(np.sum(a*b,axis=-1)-r0*r1),np.sum(b*b,axis=-1)-r1*r1],axis=-1)


def roots(c):
    """Two finite real roots or NaNs, including degenerate linear cases."""
    c=np.asarray(c,float);d,b,a=np.moveaxis(c,-1,0)
    out=np.full(c.shape[:-1]+(2,),np.nan)
    quad=a!=0
    disc=b*b-4*a*d
    valid=quad & (disc>=0)
    # Stable quadratic formula; repeated zero roots handled separately.
    q=-.5*(b+np.copysign(np.sqrt(np.maximum(disc,0)),b))
    np.divide(q,a,out=out[...,0],where=valid)
    np.divide(d,q,out=out[...,1],where=valid & (q!=0))
    repeat=valid & (q==0)
    out[...,1]=np.where(repeat,out[...,0],out[...,1])
    linear=~quad & (b!=0)
    np.divide(-d,b,out=out[...,0],where=linear)
    return out


def prepare_na62(batch,pdg):
    (p,d,v,w,o),mass=batch.arrays()
    ids=np.asarray(pdg);q=np.asarray(batch.charges,float)
    if ids.shape!=v.shape or q.shape!=v.shape: raise ValueError('PDG/charge arrays required')
    forward=p[:,2]>0
    beta=np.divide(p[:,:2],p[:,2,None],out=np.zeros_like(p[:,:2]),where=forward[:,None])
    alpha=o[:,:2]-o[:,2,None]*beta
    fv=ElectronFV.digitized(centre_xy=(0,0));vertices=np.asarray(fv.vertices)
    lines=[]
    for left,right in [(1,2),(3,4)]:
        r1=(vertices[right,1]-vertices[left,1])/(vertices[right,0]-vertices[left,0])
        lines.append((vertices[left,1]-r1*vertices[left,0],r1))
    base=[circle_polynomial(alpha,beta,1.)]
    base += [circle_polynomial(alpha,beta,r0,r1) for r0,r1 in lines]
    # Restrict full inclusive events to the explicitly supported leptonic channels.
    # The complete inclusive branching fraction is already represented in samples.
    pair=np.sum(np.where(v[...,None],d,0.),axis=1)
    channel_pass={c:dilepton_selection(d,ids,v,channel=c) for c in ['ee','mumu']}
    selected_any=channel_pass['ee']|channel_pass['mumu']
    if np.any(selected_any & (np.sum(np.where(~v,d[...,3],0.),axis=1)>1e-7)):
        raise ValueError('Prototype interval preparation supports fully visible DP pairs only; iDM needs pointing-boundary roots')
    # Complete events have already been validated. Only the selected visible
    # pair needs detector transport; do not allocate K-daughter interval tensors
    # for hadronic events which fail both leptonic channels.
    order=np.argsort(~v,axis=1,kind='stable')[:,:2]
    if order.shape[1]!=2: raise ValueError('At least two daughter slots required')
    dd=np.take_along_axis(d,order[...,None],axis=1)
    qq=np.take_along_axis(q,order,axis=1)
    vv=np.broadcast_to(selected_any[:,None],order.shape)
    slopes=np.zeros_like(dd[...,:2]);shifts=np.zeros_like(slopes);ok=np.zeros(vv.shape,bool)
    for i,j in zip(*np.nonzero(vv)):
        if dd[i,j,2]<=0: continue
        sx,sy,good=dipole_displacement(dd[i,j,:3],qq[i,j],*ZMAG,FIELD,PLANE)
        if good:
            slopes[i,j]=dd[i,j,:2]/dd[i,j,2];shifts[i,j]=[sx,sy];ok[i,j]=True
    a=alpha[:,None,:]+PLANE*slopes+shifts
    b=beta[:,None,:]-slopes
    detector_polys=[]
    for j in range(2):
        detector_polys.extend([circle_polynomial(a[:,j],b[:,j],1.15),circle_polynomial(a[:,j],b[:,j],.08)])
    all_roots=np.concatenate([roots(c) for c in base+detector_polys],axis=1)
    all_roots=np.where(np.isfinite(all_roots),all_roots,180.)
    lefts=[];rights=[];weights=[];selections=[];indices=[]
    for ci,channel in enumerate(['ee','mumu']):
        low=vertices[0,0] if channel=='ee' else 105.
        breaks=np.column_stack([np.full(len(p),low),np.full(len(p),180.),np.full(len(p),vertices[2,0]),all_roots])
        breaks=np.sort(np.clip(breaks,low,180.),axis=1)
        lo,hi=breaks[:,:-1],breaks[:,1:];mid=(lo+hi)/2
        xyz=np.empty(mid.shape+(3,));xyz[...,:2]=alpha[:,None,:]+mid[...,None]*beta[:,None,:];xyz[...,2]=mid
        vol=(hi>lo)&forward[:,None]&(np.sum(xyz[...,:2]**2,axis=-1)<1.)&(lo>=o[:,2,None])
        z,cda,valid=pointing(xyz.reshape(-1,3),np.repeat(pair[:,:3],mid.shape[1],axis=0),beam_xy=ORIGIN[:2])
        sr=signal_region(z,cda,channel=channel).reshape(mid.shape)
        selected=vol & channel_pass[channel][:,None] & sr
        if channel=='ee': selected &= fv(mid.ravel(),xyz[...,:2].reshape(-1,2)).reshape(mid.shape)
        hit=a[:,None,:,:]+mid[:,:,None,None]*b[:,None,:,:]
        r2=np.sum(hit*hit,axis=-1)
        detector=np.all(~vv[:,None,:] | (ok[:,None,:] & (r2>.08**2)&(r2<1.15**2)),axis=-1)
        masks=[vol,selected,selected&detector]
        for si,mask in enumerate(masks):
            ii,jj=np.nonzero(mask)
            factor=mass/p[ii,2]
            lefts.append((lo[ii,jj]-o[ii,2])*factor);rights.append((hi[ii,jj]-o[ii,2])*factor)
            weights.append(w[ii]);selections.append(np.full(len(ii),3*ci+si,dtype=np.int64));indices.append(ii.astype(np.int64))
    geom=Geometry(z_min=105,z_max=180,x_in=2,x_out=2,y_in=2,y_out=2,aperture_z=PLANE,aperture_x=2.3,aperture_y=2.3,
                  magnet_z_min=ZMAG[0],magnet_z_max=ZMAG[1],field_integral=FIELD,target_z=ORIGIN[2])
    return PreparedAcceptance(np.concatenate(lefts),np.concatenate(rights),np.concatenate(weights),np.concatenate(selections),
        tuple(Selection(n,min_particles=0,energy_min=0) for n in NAMES),geom,event_index=np.concatenate(indices))
