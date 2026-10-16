"""Production decay tensor and QRA polarization. Four-vectors are (px,py,pz,E)."""
import numpy as np
from numba import njit

SIGMA = np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]],complex)
GAMMA=np.zeros((4,4,4),complex)
GAMMA[0]=np.diag([1,1,-1,-1])
for i in range(3):
    GAMMA[i+1,:2,2:]=SIGMA[i]
    GAMMA[i+1,2:,:2]=-SIGMA[i]

@njit(cache=True)
def slash(v):
    return v[0]*GAMMA[0]-v[1]*GAMMA[1]-v[2]*GAMMA[2]-v[3]*GAMMA[3]

@njit(cache=True)
def spinors(px,pz,m):
    e=np.sqrt(px*px+pz*pz+m*m)
    a=np.zeros((4,2),np.complex128)
    a[0,0]=a[1,1]=np.sqrt(e+m)
    a[2,0]=pz/np.sqrt(e+m);a[2,1]=px/np.sqrt(e+m)
    a[3,0]=px/np.sqrt(e+m);a[3,1]=-pz/np.sqrt(e+m)
    return a

@njit(cache=True)
def qra_density_one(z,t,m,f1,f2):
    """Spin-summed QRA vertex with Dawson-corrected longitudinal emission.

    The high-energy limit is taken at P=10^4 max(m,mp,1) GeV. The
    physical outgoing longitudinal state is attached to its corrected
    emission amplitude; the common cross-section weight remains the
    analytic QRA kernel. Matrix basis is (in-plane T, normal T, L).
    """
    mp=.938272
    p=1e4*max(1.,m)
    pt=np.sqrt(t);kz=z*p;kn=np.sqrt(kz*kz+t);ek=np.sqrt(kn*kn+m*m)
    # The Pauli vertex uses the difference of the on-shell proton momenta.
    # Its energy differs from the vector energy within the QRA. This choice
    # is essential for reproducing w2 and the Dirac-Pauli interference.
    k=np.array([np.hypot(p,mp)-np.sqrt((1-z)**2*p*p+t+mp*mp),pt,0.,kz]);ks=slash(k)
    pol=np.array([[0.,kz/kn,0.,-pt/kn],[0.,0.,1.,0.],
                  [-m/(ek+kn),m*pt/(kn*(ek+kn)),0.,m*kz/(kn*(ek+kn))]])
    u=spinors(0.,p,mp);up=spinors(-pt,(1-z)*p,mp)
    ub=up.conj().T@GAMMA[0]
    a=np.empty((3,2,2),np.complex128)
    for i in range(3):
        es=slash(pol[i])
        vertex=f1*es+f2/(4*mp)*(es@ks-ks@es)
        a[i]=ub@vertex@u
    h=np.zeros((3,3),np.complex128)
    for i in range(3):
        for j in range(3):
            h[i,j]=np.sum(a[i]*a[j].conj())/2
    tr=(h[0,0]+h[1,1]+h[2,2]).real
    return h/tr,tr

@njit(cache=True)
def qra_projection(z,t,m,f1,f2,n):
    out=np.empty(len(z))
    for k in range(len(z)):
        h,_=qra_density_one(z[k],t[k],m[k],f1[k],f2[k])
        value=0.
        for i in range(3):
            for j in range(3):value+=n[k,i]*n[k,j]*h[i,j].real
        out[k]=value
    return out

def width(q,m1,m2,alpha_D=.1):
    q=np.asarray(q)
    plus=(m1+m2)**2/q**2;minus=(m2-m1)**2/q**2
    return alpha_D*q/3*np.sqrt(np.maximum((1-plus)*(1-minus),0))*(1-minus)*(1+plus/2)*(q>m1+m2)

def decay_weight(q,m1,m2,projection):
    a=(q*q-(m2-m1)**2)/2
    k2=np.maximum((q*q-(m1+m2)**2)*(q*q-(m2-m1)**2),0)/(4*q*q)
    return (a-2*k2*projection)/(a-2*k2/3)

def boost(p4,frame):
    """Active boost from the frame rest system; broadcast vectorized arrays."""
    b=frame[...,:3]/frame[...,3,None]
    b2=(b*b).sum(axis=-1);m=np.sqrt(np.maximum(frame[...,3]**2-(frame[...,:3]**2).sum(axis=-1),1e-28))
    g=frame[...,3]/m;dot=(p4[...,:3]*b).sum(axis=-1)
    fac=np.divide((g-1)*dot,b2,out=np.zeros_like(dot),where=b2>0)+g*p4[...,3]
    return np.concatenate([p4[...,:3]+fac[...,None]*b,(g*(p4[...,3]+dot))[...,None]],axis=-1)

def rest_axis(p4,frame):
    back=frame.copy();back[...,:3]*=-1
    q=boost(p4,back)[...,:3];n=np.linalg.norm(q,axis=-1)
    return np.divide(q,n[...,None],out=np.tile([0.,0.,1.],(len(q),1)),where=n[...,None]>0)

def unit_vectors(u,v):
    c=2*u-1;phi=2*np.pi*v;s=np.sqrt(np.maximum(1-c*c,0))
    return np.column_stack([s*np.cos(phi),s*np.sin(phi),c])

def daughter(q4,q,m1,m2,n):
    e=(q*q+m2*m2-m1*m1)/(2*q)
    k=np.sqrt(np.maximum(e*e-m2*m2,0))
    return boost(np.column_stack([k[:,None]*n,e]),q4)
