"""Inclusive electromagnetic continuum above the 5 GeV exHad boundary.

No extrapolation of exclusive hadron amplitudes. The uds continuum uses the
same perturbative R ratio as exHad; open charm is matched continuously to
its measured 5 GeV rate and evolves with massive-quark phase space and QCD.
Complete final states use Pythia timelike showers, fragmentation and decays.
This continuation is restricted to 5--8 GeV, below bottom production.
"""
import math
import numpy as np
import os
from pathlib import Path
XML=Path(os.environ.get("PYTHIA8DATA",os.environ.get("DP_PYTHIA_XML","")))
from exhad.inelastic_dark_matter import ALPHA,lepton_width
from .spectral_widths import spectral_point,GROUPS


def alpha_s(q):
    # Same four-loop continuation and Lambda as exHad's DeLiVeR input.
    nf=5.;b0=11-2*nf/3;b1=102-38*nf/3;b2=2857/2-5033*nf/18+325*nf*nf/54
    b3=100541-24423.3*nf+1625.4*nf*nf-27.493*nf**3
    L=math.log(q*q/.212**2);l=math.log(L)
    return 4*math.pi/(b0*L)*(1-b1*l/(b0*b0*L)+b1*b1/(b0**4*L*L)*(l*l-l-1+b2*b0/b1**2)+b1**3/(b0**6*L**3)*(-l**3+2.5*l*l+2*l-.5-3*b2*b0*l/b1**2+b3*b0*b0/(2*b1**3)))


def correction(q):
    a=alpha_s(q)/math.pi;nf=3.
    coefficients=[1.,1.9857-.1152*nf,-6.63694-1.20013*nf-.00518*nf*nf,-156.61+18.775*nf-.7974*nf*nf+.0215*nf**3]
    return 1+sum(c*a**(i+1) for i,c in enumerate(coefficients))


def widths(q):
    if not 5<=q<=8:raise ValueError('Continuum extension covers only 5--8 GeV')
    anchor=spectral_point(5.);mu=lepton_width(q,.1056583755);mu5=anchor[1]
    uds5=anchor[3:-1].sum();charm5=anchor[-1]
    uds=uds5*mu/mu5*correction(q)/correction(5.)
    def phase(m):
        r=(1.27/m)**2;return (1+2*r)*math.sqrt(1-4*r)
    charm=charm5*mu/mu5*phase(q)/phase(5.)*correction(q)/correction(5.)
    return np.array([lepton_width(q,.00051099895),mu,lepton_width(q,1.77686),uds*4/6,uds/6,uds/6,charm])


def generate(mass,count,seed):
    try:import pythia8mc as pythia8
    except ImportError:import pythia8
    p=pythia8.Pythia(str(XML),False)
    for setting in ['Print:quiet = on','ProcessLevel:all = off','PartonLevel:all = on','HadronLevel:all = on','Check:event = on','Check:epTolErr = 1e-8','ParticleDecays:limitTau0 = off','TimeShower:QEDshowerByQ = off','TimeShower:QEDshowerByL = off','Random:setSeed = on',f'Random:seed = {int(seed)%899999999+1}']:
        if not p.readString(setting):raise RuntimeError(setting)
    for pid in [13,211,321,130]:p.readString(f'{pid}:mayDecay = off')
    if not p.init():raise RuntimeError('Pythia continuum initialization failed')
    rate=widths(mass);rng=np.random.default_rng(seed);modes=rng.choice([11,13,15,2,1,3,4],count,p=rate/rate.sum());events=[];failures=0;showers=0
    for mode in modes:
        mode=int(mode);mq=p.particleData.m0(mode);energy=mass/2;momentum=math.sqrt(energy*energy-mq*mq)
        if mode in (11,13):
            events.append(np.array([[0,0,momentum,energy,mq,mode],[0,0,-momentum,energy,mq,-mode]],float));continue
        for attempt in range(32):
            p.event.reset();p.event[0].p(0.,0.,0.,mass);p.event[0].m(mass);p.event[0].status(-11)
            p.event.append(22,-23,0,0,2,3,0,0,0.,0.,0.,mass,mass)
            color=501 if mode<5 else 0
            p.event.append(mode,23,1,1,0,0,color,0,0.,0.,momentum,energy,mq)
            p.event.append(-mode,23,1,1,0,0,0,color,0.,0.,-momentum,energy,mq)
            if mode<5:
                p.event[2].scale(mass/2);p.event[3].scale(mass/2);showers+=p.forceTimeShower(2,3,mass/2)
            if not p.forceHadronLevel():failures+=1;continue
            rows=np.array([[v.px(),v.py(),v.pz(),v.e(),v.m(),v.id()] for v in p.event if v.isFinal()],float)
            if len(rows) and np.max(abs(rows[:,:4].sum(0)-[0,0,0,mass]))<2e-7:
                events.append(rows);break
            failures+=1
        else:raise RuntimeError(f'Failed continuum decay after 32 attempts at {mass} GeV')
    maximum=float(max(np.max(abs(e[:,:4].sum(0)-[0,0,0,mass])) for e in events))
    audit=dict(mass_GeV=mass,events=count,seed=seed,failed_attempts=failures,shower_branchings=showers,maximum_four_momentum_residual_GeV=maximum,mode_counts={str(k):int(sum(modes==k)) for k in [11,13,15,2,1,3,4]},partial_widths_GeV_per_epsilon2=rate.tolist())
    return events,float(rate.sum()),audit
