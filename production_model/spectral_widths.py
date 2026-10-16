"""Channel-resolved electromagnetic spectral measure and exact iDM spectral integral.

The same exHad light-hadron widths are used below and above its 1.7 GeV seam.
No exclusive state is added on top of an inclusive width that already owns it.
Support ends at 5 GeV, the validated exHad release endpoint (below requested 6).
The narrow hidden-charm resonances are outside this release; the smooth curves
must not be interpreted at J/psi or psi(2S). All widths are per epsilon^2.
"""
from functools import lru_cache
import json
import numpy as np
from scipy.integrate import trapezoid

from exhad.inelastic_dark_matter import ALPHA,HBARC,ME,lepton_width,catalog,differential_width
from exhad import exclusive
from exhad.dark_photon.closure import load_inputs
from exhad.dark_photon.em_response import EmResponseFit
from exhad.dark_photon.charm_rates import build_open_charm_owner
GROUPS=['ee','mumu','tautau','2pi','3pi','4pi','5pi','6pi','KK','KKpi','KK2pi','Pgamma','etaX','NN','other','charm']
LABELS=[r'$e^+e^-$',r'$\mu^+\mu^-$',r'$\tau^+\tau^-$',r'$\pi^+\pi^-$',r'$3\pi$',r'$4\pi$',r'$5\pi$',r'$6\pi$',r'$K\bar K$',r'$K\bar K\pi$',r'$K\bar K\,2\pi$',r'$P\gamma$',r'$\eta^{(\prime)}+X$',r'$N\bar N$',r'Other hadrons',r'Open charm']

def group_pdgs(ids):
 ids=[abs(int(i)) for i in ids];n=len(ids)
 if any(i in [221,331] for i in ids):return 'Pgamma' if 22 in ids and n==2 else 'etaX'
 if 22 in ids and n==2:return 'Pgamma'
 if all(i in [111,211] for i in ids) and 2<=n<=6:return str(n)+'pi'
 if all(i in [2212,2112] for i in ids):return 'NN'
 kaons=sum(i in [130,310,311,321] for i in ids);pions=sum(i in [111,211] for i in ids)
 if kaons==2 and kaons+pions==n:return ['KK','KKpi','KK2pi'][pions] if pions<3 else 'other'
 return 'other'

# Strong-decay expansion makes the pion/kaon classes consistent across 1.7 GeV.
# Eta, eta', pi0 and neutral kaons remain identified for these branching families.
# Tiny unlisted vector decays belong to Other, not to a renormalized principal mode.
VECTOR_DECAYS={223:[(.892,[211,-211,111]),(.0835,[111,22]),(.0153,[211,-211])],333:[(.492,[321,-321]),(.340,[130,310]),(.152,[211,-211,111]),(.013,[221,22])]}
def primary_fractions(ids):
 states=[(1.,list(ids))]
 for vector,branches in VECTOR_DECAYS.items():
  out=[]
  for w,state in states:
   if vector not in state:out.append((w,state));continue
   tail=state.copy();tail.remove(vector)
   for b,d in branches:out.append((w*b,tail+d))
   out.append((w*(1-sum(b for b,d in branches)),[0]))
  states=out
 result={}
 for w,state in states:
  g=group_pdgs(state);result[g]=result.get(g,0)+w
 return result

def key_group(key):
 ids=[]
 for item in key.split():
  p,n=item.split(':');ids.extend([int(p)]*int(n))
 return group_pdgs(ids)

@lru_cache(maxsize=1)
def authorities():return load_inputs(),EmResponseFit(None),build_open_charm_owner()

def spectral_point(q):
 if not 2*ME<q<=5.:raise ValueError('Spectral support: electron threshold < q <= 5 GeV')
 out=np.zeros(len(GROUPS));out[0]=lepton_width(q,ME);out[1]=lepton_width(q,.1056583755);out[2]=lepton_width(q,1.77686)
 def add(group,value):out[GROUPS.index(group)]+=value
 if q<1.7:
  # exHad enforces physical thresholds, including the held 1.699-1.700 seam.
  if q<exclusive.table_start('dark-photon'):return out
  rows=catalog();br=exclusive.rates('dark-photon',q);be=br[next(i for i,r in enumerate(rows) if r.label=='e-pair')]
  for row,b in zip(rows,br):
   if b<=0 or not row.has_hadrons:continue
   for group,fraction in primary_fractions(row.pdgs).items():add(group,out[0]*b/be*fraction)
 else:
  inputs,fit,charm=authorities();window=inputs.exclusive_edge if q<=2. else inputs.post_edge
  closure=window.closure_at(float(q));factor=4*np.pi*ALPHA
  for channel in closure.channels:
   groups={key_group(k) for k in channel.final_state_keys};group=next(iter(groups)) if len(groups)==1 else 'other'
   add(group,factor*channel.width_gev)
  if q<=2.:add('other',factor*closure.residual_width_gev)
  else:
   point=fit.rate_point_at(float(q),inclusive_width_gev=closure.inclusive_width_gev,external_width_gev=closure.external_width_gev)
   for r in fit.routes:
    groups={key_group(k) for k in r.final_state_keys};group=next(iter(groups)) if len(groups)==1 else 'other'
    add(group,factor*point.resolved_widths_gev[r.response_block_id])
   add('other',factor*point.residual_width_gev)
  if q>=charm.card.domain_gev[0]:add('charm',factor*charm.closure_at(q).inclusive_charm_width)
 return out

@lru_cache(maxsize=1)
def spectral_grid():
 q=np.unique(np.r_[np.geomspace(2*ME*(1+1e-9),.15,400),np.arange(.15,1.7,.0004),np.arange(1.7,5.,.002),[1.699,1.7,2.,3.72966,5.],np.arange(.775,.790,.00004),np.arange(1.010,1.030,.00003)])
 g=np.array([spectral_point(float(v)) for v in q]);return q,g

def transition_rates(ma,delta=.1,ratio=3.,alpha_D=.1):
 m1=ma/ratio;m2=m1*(1+delta);end=m2-m1
 if not 2*ME<end<=5.+1e-9:raise ValueError('q endpoint outside the spectral table')
 q,g=spectral_grid();mask=q<min(end,5.);x=np.r_[q[mask],end];gamma=np.vstack([g[mask],spectral_point(min(end,5.))])
 integrand=differential_width(x*x,m1,m2,ma,alpha_D,widths=gamma)*2*x[:,None]
 return trapezoid(integrand,x,axis=0)

