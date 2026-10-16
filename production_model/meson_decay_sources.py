"""Measured transition form factors and on-shell dark-photon decay kernels.

Functions return BR/epsilon^2, with radiative BRs from the local particle table.
Production of the parent is a separate input. No proton bremsstrahlung is added.
"""
import numpy as np

PARENTS={111:(.1349766,.98799),221:(.547862,.3931181),331:(.95778,.0219297),223:(.78265,.0834941)}
PROVENANCE={
 'pi0':dict(source='https://arxiv.org/abs/1612.08162',a=.0368,error=.0057,model='F(q2)=1+a*q2/m_pi0^2'),
 'eta':dict(source='https://arxiv.org/abs/1608.07898',slope=1.934,error=float(np.hypot(.067,.050)),model='NA60 pole; F(0)=1'),
 'eta_prime':dict(source='https://arxiv.org/html/2401.09136#S5',pole=.802,pole_error=float(np.hypot(.007,.008)),width=.113,width_error=float(np.hypot(.010,.002)),model='BESIII 2024 Eq.5; F(0)=1; parameter covariance unavailable'),
 'omega':dict(source='https://arxiv.org/abs/1608.07898',slope=2.223,error=float(np.hypot(.026,.037)),model='NA60 pole; integrated rate matches their alternative extraction, not the independent direct branching central value')}

def form_factor_squared(pid,m,variation=0,width_variation=0):
    q2=np.asarray(m,dtype=float)**2
    if pid==111:return (1+(.0368+variation*.0057)*q2/PARENTS[111][0]**2)**2
    if pid in (221,223):
        v=PROVENANCE['eta' if pid==221 else 'omega']
        return (1-(v['slope']+variation*v['error'])*q2)**-2
    if pid==331:
        v=PROVENANCE['eta_prime'];lam=v['pole']+variation*v['pole_error'];gam=v['width']+width_variation*v['width_error']
        return lam**2*(lam**2+gam**2)/((lam**2-q2)**2+lam**2*gam**2)
    raise ValueError(f'Unsupported parent {pid}')

def branching(pid,m,variation=0,width_variation=0):
    m=np.asarray(m,dtype=float)
    if np.any(m<0):raise ValueError("Mass must be nonnegative")
    mp,br=PARENTS[pid]
    if pid in (111,221,331):
        phase=np.maximum(1-m*m/mp**2,0)**3
        # Evaluate form factors only in physical phase space (avoid distant poles).
        safe=np.minimum(m,mp*(1-1e-12))
        return 2*br*phase*form_factor_squared(pid,safe,variation,width_variation)*(m<mp)
    daughter=PARENTS[111][0];d=mp*mp-daughter*daughter
    phase=np.maximum((1+m*m/d)**2-4*mp*mp*m*m/d**2,0)**1.5
    safe=np.minimum(m,(mp-daughter)*(1-1e-12))
    return br*phase*form_factor_squared(pid,safe,variation,width_variation)*(m<mp-daughter)

def sample_decay(pid,m,p4,rng):
    """Unpolarized two-body decay; p4 columns px,py,pz,E, output same.

    Each parent's invariant mass is retained, so omega width variations do not
    violate four-momentum conservation. The branching function uses pole masses.
    """
    p4=np.atleast_2d(np.asarray(p4,float));p=p4[:,:3];energy=p4[:,3]
    invariant=np.sqrt(np.maximum(energy**2-np.sum(p*p,axis=1),0))
    daughter=0. if pid in (111,221,331) else PARENTS[111][0]
    if np.any(invariant<m+daughter):raise ValueError('Parent below decay threshold')
    ev=(invariant**2+m*m-daughter*daughter)/(2*invariant)
    q=np.sqrt(np.maximum(ev*ev-m*m,0))
    ct=rng.uniform(-1,1,len(p4));ph=rng.uniform(0,2*np.pi,len(p4));st=np.sqrt(1-ct*ct)
    v=q[:,None]*np.column_stack([st*np.cos(ph),st*np.sin(ph),ct])
    beta=p/energy[:,None];b2=np.sum(beta*beta,axis=1);gam=energy/invariant
    dot=np.sum(beta*v,axis=1)
    coef=np.divide((gam-1)*dot,b2,out=np.zeros(len(p4)),where=b2>0)+gam*ev
    return np.column_stack([v+coef[:,None]*beta,gam*(ev+dot)])
