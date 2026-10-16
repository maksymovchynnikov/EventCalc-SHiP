"""Correlated production variations on the same collision and decay samples.

The proton virtuality scale stays at the fitted Foroughi-Abari value. Only the
tabulated timelike form-factor pairs, PDF members, QCD scales and continuum
transition are varied. No inclusive-flux rescaling replaces decay acceptance.
"""
from dataclasses import dataclass
from pathlib import Path
import copy
import hashlib
import json
import numpy as np
from .fold_idm_production import CHANNELS, evaluate
from .idm_production_spin import daughter, decay_weight
from .normalization import transition

FAMILIES={'CT18NLO':(59,1.6448536269514722),
          'MSHT20nlo_as118':(65,1.), 'NNPDF40_nlo_as_01180':(101,None)}
SCALES=((1.,1.),(.5,.5),(.5,1.),(1.,.5),(1.,2.),(2.,1.),(2.,2.))
FORM_FACTORS=('central','lower','upper')
OMITTED=('Alternative proton virtuality descriptions, higher vector fragmentation '
         'excitations, fragmentation polarization, nuclear/shower modeling, finite '
         'production-pool statistics and detector reconstruction.')


def validate_labels(labels):
    actual=[tuple(x) for x in labels]
    expected={(family,member,1.,1.) for family,(count,_) in FAMILIES.items()
              for member in range(count)}
    expected.update((family,0,*scale) for family in FAMILIES for scale in SCALES)
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError('Incomplete or duplicate PDF member / seven-scale labels')
    if actual[0]!=('CT18NLO',0,1.,1.):
        raise ValueError('The first PDF column must be the nominal CT18 member')


def pdf_envelope(values,labels):
    """Combine *integrated* yields, after summing primary and cascade sources.

CT18's 90% Hessian errors are converted to 68%; MSHT20 uses its 68% Hessian
errors and NNPDF uses the 16th/84th replica percentiles. PDF-family and seven
scale envelopes are combined as in the manuscript, without a joint CL claim.
"""
    validate_labels(labels)
    values=np.asarray(values,float)
    if values.shape[0]!=len(labels) or not np.all(np.isfinite(values)):
        raise ValueError('Expected finite integrated yields for every PDF label')
    lookup={tuple(label):i for i,label in enumerate(labels)}
    lows=[values[0]];highs=[values[0]]
    for family,(count,confidence) in FAMILIES.items():
        members=values[[lookup[family,j,1.,1.] for j in range(count)]]
        central=members[0]
        if confidence is None:
            lo,hi=np.quantile(members[1:],[.16,.84],axis=0)
        else:
            pairs=(members[1:]-central).reshape((-1,2)+central.shape)
            lo=central-np.sqrt(np.sum(np.maximum(-pairs.min(axis=1),0)**2,axis=0))/confidence
            hi=central+np.sqrt(np.sum(np.maximum(pairs.max(axis=1),0)**2,axis=0))/confidence
        scales=values[[lookup[family,0,*scale] for scale in SCALES]]
        lows.extend([lo,scales.min(axis=0)]);highs.extend([hi,scales.max(axis=0)])
    return np.maximum(0,np.min(lows,axis=0)),np.max(highs,axis=0)


class PDFWeights:
    """Load portable hard-event weights, bound to the exact production pools."""
    def __init__(self,inputs,path=None):
        self.path=Path(path) if path is not None else inputs.root/'uncertainties'
        manifest=self.path/'manifest.json'
        if not manifest.is_file():
            raise FileNotFoundError(f'Production uncertainty data are missing: {manifest}. '
                                    'Install the uncertainty package or explicitly request --central-only.')
        self.manifest=json.loads(manifest.read_text())
        for name,expected in self.manifest['sha256'].items():
            p=self.path/name
            if not p.is_file():raise FileNotFoundError(f'Missing uncertainty input: {p}')
            with p.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=expected:raise ValueError(f'Uncertainty input checksum mismatch: {p}')
        source=inputs.root/'results/idm_production/sources'
        if json.loads((self.path/'source_provenance.json').read_text())!=json.loads((source/'provenance.json').read_text()):
            raise ValueError('PDF identities do not match the production source pools')
        for name,expected in self.manifest['source_pool_sha256'].items():
            with (source/name).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=expected:raise ValueError(f'PDF/source-row identity mismatch: {name}')
        self.labels=json.loads((self.path/'labels.json').read_text());validate_labels(self.labels)
        self.blocks=[];self.stops=[];rows=0
        chunks=self.manifest.get('weight_chunks',[{'file':'ratios.npy'}])
        for chunk in chunks:
            weights=np.load(self.path/chunk['file'],mmap_mode='r',allow_pickle=False)
            if weights.ndim!=2 or weights.shape[1]!=len(self.labels) or not len(weights):
                raise ValueError('PDF weight columns do not match their labels')
            if chunk.get('start',rows)!=rows or chunk.get('stop',rows+len(weights))!=rows+len(weights):
                raise ValueError('PDF weight chunks are not contiguous')
            rows+=len(weights);self.blocks.append(weights);self.stops.append(rows)
        if not rows:raise ValueError('Empty PDF weight package')
        self.rows=rows
        self.ids={}
        for cat in ('primary','cascade'):
            key='dy',0,cat
            for k,mass in enumerate(inputs.knots[key]):
                ids=np.load(self.path/f'dy_{mass:g}_0_{cat}_ids.npy',allow_pickle=False)
                if ids.dtype.kind not in 'iu' or ids.ndim!=1 or len(ids)!=len(inputs.samples[key][k][0]):
                    raise ValueError(f'PDF row map does not match source pool {mass}, {cat}')
                if len(ids) and (ids.min()<0 or ids.max()>=self.rows):
                    raise ValueError('PDF row identity outside weight matrix')
                self.ids[cat,k]=ids

    def sampled_ids(self,inputs,cat):
        out=np.zeros(len(inputs.current_dy_pool),dtype=np.int64)
        for k in np.unique(inputs.current_dy_pool):
            take=inputs.current_dy_pool==k;pool=self.ids[cat,k]
            if len(pool):out[take]=pool[inputs.current_dy_row[take]]
        return out

    def take(self,ids):
        ids=np.asarray(ids)
        if ids.ndim!=1 or ids.dtype.kind not in 'iu' or (len(ids) and (ids.min()<0 or ids.max()>=self.rows)):
            raise ValueError('Invalid PDF hard-event identities')
        ratios=np.empty((len(ids),len(self.labels)))
        chosen=np.searchsorted(self.stops,ids,side='right')
        for k in np.unique(chosen):
            mask=chosen==k;start=0 if k==0 else self.stops[k-1]
            ratios[mask]=self.blocks[k][ids[mask]-start]
        if not np.all(np.isfinite(ratios)) or not np.all(ratios[:,0]==1.):
            raise ValueError('Invalid PDF ratios or non-unit nominal CT18 weights')
        return ratios


@dataclass
class VariationBatch:
    momenta: np.ndarray
    origins: np.ndarray
    weights: np.ndarray
    mass: float
    group: str
    reweights: np.ndarray


class ProductionVariations:
    """Sample each decay once; keep separate absolute production-weight columns.

DY weights include the nuclear factor once and omit the transition until the
variation columns are formed. This preserves contributions where a nominal
weight vanishes. For iDM, each form-factor choice also changes the mediator
polarization weight. Daughter kinematics are shared across all choices.
"""
    def __init__(self,production,weights_root=None):
        self.inputs=copy.copy(production.inputs)
        self.inputs.apply_dy_transition=False
        self.pdf=PDFWeights(self.inputs,weights_root)

    @staticmethod
    def endpoints(model,mass,delta):
        minimum=mass if model=='dp' else mass*(2+delta)/3
        return (2.,1.75,2.5) if minimum<2.5 else (2.,)

    def batches(self,mass,*,model='dp',delta=.1,power=15,seed=1,chunk_size=4096):
        if model not in ('dp','idm') or not np.isfinite(mass) or mass<=0:
            raise ValueError('Choose dp/idm and a finite positive mediator mass')
        if not np.isfinite(delta) or delta<=0 or not 4<=power<=24 or int(power)!=power or chunk_size<1:
            raise ValueError('Invalid splitting, integration power or batch size')
        ends=self.endpoints(model,mass,delta);inp=self.inputs
        for ic,cat in enumerate(('primary','cascade')):
            for channel in CHANNELS:
                samples=[]
                choices=FORM_FACTORS if channel=='brem' else ('central',)
                for choice in choices:
                    inp.set_formfactors(choice);records=[]
                    def consume(acc,q4,q,m1,m2,n,weight,projection,uncertain=False):
                        p=daughter(q4,q,m1,m2,n) if model=='idm' else q4
                        w=weight*decay_weight(q,m1,m2,projection) if model=='idm' else weight
                        ids=self.pdf.sampled_ids(inp,cat) if channel=='dy' else None
                        records.append((p,inp.current_origin.copy(),w,m2 if model=='idm' else mass,q,ids))
                    evaluate(inp,mass,delta,cat,power=int(power),seed=seed+ic*1721,
                             finite=model=='idm',channels=[channel],consumer=consume)
                    samples.append(records)
                for j,(p,origin,w,m,q,ids) in enumerate(samples[0]):
                    norm=len(w)
                    if channel=='brem':
                        varied=np.column_stack([records[j][2] for records in samples])
                        # Do not divide by nominal weights: a variation can populate a nominal zero.
                        w=np.max(np.abs(varied),axis=1)
                        reweights=np.divide(varied,w[:,None],out=np.zeros_like(varied),where=w[:,None]!=0)
                    take=np.flatnonzero(w!=0)
                    for left in range(0,len(take),chunk_size):
                        ii=take[left:left+chunk_size]
                        if channel=='dy':
                            ratios=self.pdf.take(ids[ii])
                            factors=np.stack([ratios*transition(q[ii],end)[:,None] for end in ends],axis=1)
                            factors=factors.reshape(len(ii),-1);group='dy'
                        elif channel=='brem':factors=reweights[ii];group='brem'
                        else:factors=np.ones((len(ii),1));group='fixed'
                        yield VariationBatch(p[ii],origin[ii],w[ii]/norm,m,group,factors)
