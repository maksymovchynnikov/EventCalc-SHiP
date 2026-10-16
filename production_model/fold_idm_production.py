"""Fold uncut mediator production into chi' flux, with a running mediator width.

Production probability per proton and epsilon^2. The dark masses are fixed
at each pole-mass benchmark throughout the virtual-mass integral.
"""
from .geometry import THETA_SHIP
from pathlib import Path
import argparse,json
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.stats import qmc
from .kinematics import transform
from .normalization import transition
from .idm_production_spin import width,decay_weight,daughter,boost,rest_axis,unit_vectors,qra_projection
from .emission import mixing_weights
from .meson_decay_sources import branching,PARENTS

R=Path(__file__).resolve().parent/'data';OUT=R/'results/idm_production';SRC=OUT/'sources'
EBINS=np.linspace(0,400,41);EMASS=[.1,1.1,5.,20.]
CHANNELS=['pi0','eta','eta_prime','omega_decay','fragmentation','brem','dy']
MGRID=np.unique(np.r_[np.geomspace(.03,20,91),np.linspace(.73,.83,17),np.linspace(.76266,.80266,41),np.linspace(.97,1.07,21),np.linspace(.99946,1.03946,61),.1,.3,.6,.775,.782,.78266,1.01946,1.1,1.5,2.,3.,5.,7.,10.,15.,20.])

def load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}

class Inputs:
    apply_dy_corrections=True
    apply_dy_transition=True
    def __init__(self, *, with_origins=False, data_root=None):
        self.root=R if data_root is None else Path(data_root)
        source=self.root/'results/idm_production/sources'
        output=self.root/'results/idm_production'
        self.origins={} if with_origins else None
        self.samples={};self.coef={};self.knots={};self.nuclear={}
        supplemental=load(output/"dy_nuclear_factors.npz") if self.apply_dy_corrections and (output/"dy_nuclear_factors.npz").exists() else {}
        self.config=json.loads((self.root/'results/fit/production_comparison_inputs.json').read_text())
        fit=json.loads((self.root/'results/fit/combined_cross_species.json').read_text())
        self.lam=next(x['lambda_p_GeV'] for x in fit['records'] if x['name']=='Shared compatibility refit' and x['choice']==fit['nominal'])
        self.set_formfactors('central')
        for cat in ['primary','cascade']:
          for kind,pids in [('fragmentation',[4900113,4900223,4900333]),('dy',[0])]:
            for pid in pids:
                paths=sorted(source.glob(f'{kind}_*_{pid}_{cat}.npz'),key=lambda p:float(p.name.split('_')[1]))
                mass=[];norm=[];pools=[]
                for p in paths:
                    a=load(p);mass.append(float(a['mass']));norm.append(float(a['norm']))
                    pools.append((a['rows'].astype(np.float32),a['sign'].astype(np.float32)))
                    if with_origins:self.origins.setdefault((kind,pid,cat),[]).append(load(self.root/'results/propagation/origins'/p.name)['origin'])
                    if kind=='dy' and self.apply_dy_corrections:self.nuclear[float(a['mass']),cat]=a.get('nuclear_factor',supplemental.get(p.stem))
                key=(kind,pid,cat);self.knots[key]=np.array(mass);self.coef[key]=np.array(norm);self.samples[key]=pools
          for pid in [111,221,331,223]:
            self.samples[pid,cat]=load(source/f'parent_{pid}_{cat}.npz')
            if with_origins:self.origins[pid,cat]=load(self.root/f'results/propagation/origins/parent_{pid}_{cat}.npz')['origin']
          self.samples['brem',cat]=load(source/f'brem_{cat}.npz')
          if with_origins:self.origins['brem',cat]=load(self.root/f'results/propagation/origins/brem_{cat}.npz')['origin']

    def set_formfactors(self,choice):
        filename,indices={'central':('FFproton_UAmodel.txt',(1,2)),
                          'lower':('FFproton_UAmodel_Uncertainty.txt',(1,3)),
                          'upper':('FFproton_UAmodel_Uncertainty.txt',(2,4))}[choice]
        ff=np.loadtxt(self.root/'reference_curves/proton_form_factors'/filename,dtype=str)
        self.ffx=ff[:,0].astype(float)
        self.ff=np.array([[complex(row[i].replace('+-','-')) for i in indices] for row in ff])

    def kernel(self,kind,pid,cat,q,u):
        key=kind,pid,cat;mass=self.knots[key];norm=self.coef[key];pools=self.samples[key]
        hi=np.clip(np.searchsorted(mass,q,side='right'),1,len(mass)-1);lo=hi-1
        frac=np.clip((q-mass[lo])/(mass[hi]-mass[lo]),0,1)
        interp=np.maximum(PchipInterpolator(mass,norm,extrapolate=False)(np.clip(q,mass[0],mass[-1])),0)
        if kind=='dy':
            # Same log-linear mass interpolation as the published DY flux.
            interp=(1-frac)*norm[lo]+frac*norm[hi]
            good=(norm[lo]>0)&(norm[hi]>0)
            interp[good]=np.exp((1-frac[good])*np.log(norm[lo[good]])+frac[good]*np.log(norm[hi[good]]))
            interp[q<mass[0]]=0
        above=q>mass[-1]
        if norm[-1]>0 and norm[-2]>0:
            interp[above]=norm[-1]*np.exp(np.log(norm[-1]/norm[-2])*(q[above]-mass[-1])/(mass[-1]-mass[-2]))
        else:interp[above]=0
        pr=np.divide(frac*norm[hi],(1-frac)*norm[lo]+frac*norm[hi],out=np.zeros_like(q),where=((1-frac)*norm[lo]+frac*norm[hi])>0)
        chosen=np.where(u[:,3]<pr,hi,lo);rows=np.zeros((len(q),10));rows[:,3:5]=.938;rows[:,8]=1;sign=np.zeros(len(q))
        if kind=='dy':
            self.current_dy_pool=chosen
            self.current_dy_row=np.zeros(len(q),dtype=np.int64)
        if self.origins is not None:self.current_origin=np.zeros((len(q),3))
        for k in np.unique(chosen):
            mask=chosen==k;pool,sg=pools[k]
            if not len(pool):continue
            ii=np.minimum((u[mask,2]*len(pool)).astype(int),len(pool)-1)
            rows[mask]=pool[ii];sign[mask]=sg[ii]
            if kind=='dy':self.current_dy_row[mask]=ii
            if self.origins is not None:self.current_origin[mask]=self.origins[key][k][ii]
            if kind=='dy' and self.apply_dy_corrections:
                factors=self.nuclear[float(mass[k]),cat]
                if factors is None or len(factors)!=len(pool):raise ValueError('Missing or incompatible tungsten normalization for DY source pool')
                sign[mask]*=factors[ii]
        p4,valid,beam,target=transform(rows,q)
        if kind=='dy' and self.apply_dy_corrections and self.apply_dy_transition:interp*=transition(q)
        return p4,interp*sign*valid,beam,target

    def formfactors(self,q):
        vals=[(np.interp(np.minimum(q,1.989),self.ffx,self.ff[:,i].real)+1j*np.interp(np.minimum(q,1.989),self.ffx,self.ff[:,i].imag))*np.minimum(1,1.989/q)**12 for i in range(2)]
        return vals

def proposal(M,m1,m2,u):
    """Truncated mixture resolves both the mediator and the narrow hadron poles."""
    low=(m1+m2)**2*(1+1e-12);high=27.5**2
    if low>=high:return np.full(len(u),m1+m2),np.zeros(len(u))
    centers=[(M*M,M*float(width(M,m1,m2)),.6)]
    poles=[(m,g) for m,g in [(.775,.149),(.78266,.00868),(1.01946,.004249)] if low<m*m<high]
    centers += [(m*m,m*g,.15/len(poles)) for m,g in poles]
    logweight=1-sum(x[2] for x in centers)
    intervals=[];cum=0.;s=np.zeros(len(u));pdf=np.zeros(len(u))
    for center,scale,w in centers:
        a=np.arctan((low-center)/scale);b=np.arctan((high-center)/scale)
        mask=(u[:,0]>=cum)&(u[:,0]<cum+w);s[mask]=center+scale*np.tan(a+(b-a)*u[mask,1]);cum+=w
        intervals.append((center,scale,w,a,b))
    mask=u[:,0]>=cum;s[mask]=low*np.exp(np.log(high/low)*u[mask,1])
    for center,scale,w,a,b in intervals:pdf+=w*scale/((s-center)**2+scale**2)/(b-a)
    pdf+=logweight/(s*np.log(high/low))
    q=np.sqrt(s);g=width(q,m1,m2)
    physical=q*g/(np.pi*((s-M*M)**2+s*g*g))
    return q,physical/pdf

def add_result(acc,q4,q,m1,m2,n,weight,projection,uncertain=False):
    out=daughter(q4,q,m1,m2,n)
    select=np.arctan2(np.linalg.norm(out[:,:2],axis=1),out[:,2])<THETA_SHIP
    w=weight*decay_weight(q,m1,m2,projection)
    central=np.where(select,w,0)
    if uncertain:
        low=np.where(select,weight*decay_weight(q,m1,m2,1.),0)
        high=np.where(select,weight*decay_weight(q,m1,m2,0.),0)
    else:low=high=central
    # Block means quantify integration conditional on the frozen source pool.
    vals=np.array([central.mean(),low.mean(),high.mean(),w.mean(),np.mean(weight*(np.arctan2(np.linalg.norm(q4[:,:2],axis=1),q4[:,2])<THETA_SHIP)),weight.mean()])
    acc['values']+=vals
    acc['blocks']+=central.reshape(16,-1).mean(axis=1)
    acc['hist']+=np.histogram(out[:,3],EBINS,weights=central/len(w))[0]

def evaluate(inp,M,delta,cat,power=15,seed=63111,finite=True,channels=None,consumer=None):
    emit=add_result if consumer is None else consumer
    m1=M/3;m2=m1*(1+delta);u=qmc.Sobol(12,scramble=True,seed=seed).random_base2(power)
    q,ws=proposal(M,m1,m2,u) if finite else (np.full(len(u),M),np.ones(len(u)))
    n=unit_vectors(u[:,4],u[:,5]);output=[];hist=[];blocks=[]
    for channel in (CHANNELS if channels is None else channels):
        a={'channel':channel,'values':np.zeros(6),'hist':np.zeros(len(EBINS)-1),'blocks':np.zeros(16)}
        if channel in ['pi0','eta','eta_prime','omega_decay']:
            pid={'pi0':111,'eta':221,'eta_prime':331,'omega_decay':223}[channel];data=inp.samples[pid,cat]
            if m1+m2>=PARENTS[pid][0]-(PARENTS[111][0] if pid==223 else 0):
                output.append(a['values']);hist.append(a['hist']);blocks.append(a['blocks']);continue
            parents=data['p4'];idx=np.minimum((u[:,2]*len(parents)).astype(int),len(parents)-1);parent=parents[idx]
            if inp.origins is not None:inp.current_origin=inp.origins[pid,cat][idx]
            pm=np.sqrt(np.maximum(parent[:,3]**2-(parent[:,:3]**2).sum(axis=1),0))
            md=PARENTS[111][0] if pid==223 else 0.;valid=q<pm-md
            ev=(pm*pm+q*q-md*md)/(2*pm);k=np.sqrt(np.maximum(ev*ev-q*q,0));direc=unit_vectors(u[:,6],u[:,7])
            # Invalid samples use a harmless physical four-vector, with zero weight.
            qs=np.minimum(q,pm-md-1e-8);ev=(pm*pm+qs*qs-md*md)/(2*pm);k=np.sqrt(np.maximum(ev*ev-qs*qs,0))
            q4=boost(np.column_stack([k[:,None]*direc,ev]),parent)
            axis=rest_axis(parent,q4);proj=(1-(n*axis).sum(axis=1)**2)/2
            weight=float(data['norm'])*branching(pid,q)*ws*valid
            # Keep a physical virtual mass in zero-weight entries to avoid overflow.
            q4[~valid]=np.column_stack([np.zeros(((~valid).sum(),3)),q[~valid]])
            emit(a,q4,q,m1,m2,n,weight,proj)
        elif channel=='fragmentation':
            mix=mixing_weights(q,inp.config['mixing_residue_scales'])
            for j,pid in enumerate([4900113,4900223,4900333]):
                q4,weight,_,_=inp.kernel(channel,pid,cat,q,u)
                emit(a,q4,q,m1,m2,n,weight*mix[:,j]*ws,1/3,uncertain=True)
        elif channel=='dy':
            q4,weight,beam,target=inp.kernel(channel,0,cat,q,u)
            axis=rest_axis(beam,q4)-rest_axis(target,q4);axis/=np.maximum(np.linalg.norm(axis,axis=1)[:,None],1e-100)
            proj=(1-(n*axis).sum(axis=1)**2)/2
            emit(a,q4,q,m1,m2,n,weight*ws,proj)
        else:
            data=inp.samples['brem',cat];ii=np.minimum((u[:,2]*len(data['p'])).astype(int),len(data['p'])-1)
            if inp.origins is not None:inp.current_origin=inp.origins['brem',cat][ii]
            pp=data['p'][ii];bn=data['n'][ii];sg=data['sigma'][ii];z=.1+.8*u[:,8];t=2*u[:,9];az=2*np.pi*u[:,10]
            e1=np.cross(bn,np.array([0.,1.,0.]));e1/=np.linalg.norm(e1,axis=1)[:,None];e2=np.cross(bn,e1)
            xyz=(z*pp)[:,None]*bn+np.sqrt(t)[:,None]*(np.cos(az)[:,None]*e1+np.sin(az)[:,None]*e2)
            energy=np.sqrt(q*q+(xyz*xyz).sum(axis=1));q4=np.column_stack([xyz,energy]);ein=np.hypot(pp,.9382720813)
            valid=((ein+.9382720813-energy)**2-((pp-z*pp)**2+t)>(2*.9382720813)**2)&(energy<ein)
            f1,f2=inp.formfactors(q)
            # brem_density interpolates its table, so provide the original fixed knots.
            ff={'mass_GeV':inp.ffx,'F1_real':inp.ff[:,0].real,'F1_imag':inp.ff[:,0].imag,'F2_real':inp.ff[:,1].real,'F2_imag':inp.ff[:,1].imag}
            # Direct kernel for the event-dependent q and high-mass continuation.
            from .emission import ALPHA,MP,MPI
            H=t+(1-z)*q*q+z*z*MP*MP;pref=ALPHA/(2*np.pi*H)
            w1=pref*((1+(1-z)**2)/z-z*(1-z)*(2*MP*MP+q*q)/H)
            w2=w1+ALPHA/(4*np.pi*MP*MP)*(1-(1-z)*(4*MP*MP+q*q)/H)*(1/z-.25*z*(4*MP*MP-q*q)/H)
            w12=pref*(2*z-z*(1-z)*3*q*q/H)
            sp=2*MP*pp*(1-z)+2*MP*MP;nsd=np.where(sp>(2*MP+MPI)**2,1.76+19.8*sp**.057,0)
            inp.current_virtuality=H/z;inp.current_pt2=t
            off=1/(1+(H/(z*inp.lam**2))**2)
            rate=np.maximum(abs(f1)**2*w1+abs(f2)**2*w2+np.real(f1*f2.conj())*w12,0)*off*off*nsd
            weight=ws*rate/sg*1.6*float(data['norm'])*valid*(H/(4*z*(1-z)**2*pp*pp)<.1)
            longitudinal=xyz/np.linalg.norm(xyz,axis=1)[:,None]
            tx=longitudinal*(bn*longitudinal).sum(axis=1)[:,None]-bn;tx/=np.maximum(np.linalg.norm(tx,axis=1)[:,None],1e-100);ty=np.cross(longitudinal,tx)
            nb=np.column_stack([(n*tx).sum(axis=1),(n*ty).sum(axis=1),(n*longitudinal).sum(axis=1)])
            projection=np.full(len(q),1/3);keep=weight>max(weight.max(initial=0)*1e-12,1e-100)
            if finite:projection[keep]=qra_projection(z[keep],t[keep],q[keep],f1[keep],f2[keep],nb[keep])
            emit(a,q4,q,m1,m2,n,weight,projection)
        output.append(a['values']);hist.append(a['hist']);blocks.append(a['blocks'])
    return np.array(output),np.array(hist),np.array(blocks)
