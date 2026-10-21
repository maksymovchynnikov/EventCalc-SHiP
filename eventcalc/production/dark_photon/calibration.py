"""NA50 normalization of proton-initiated Drell–Yan, with source identities."""
from pathlib import Path
import hashlib,json
import numpy as np

PROFILES=('central','lower','upper')
DATA=Path(__file__).resolve().parents[3]/'Distributions/Dark-photons/production/na50'


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class NA50Calibration:
    """Apply one fitted coefficient per PDF/scale hypothesis, coherently.

    Source pools remain uncalibrated. The compact proton identity mask is
    joined to their saved hard-event IDs; pion and other projectiles retain
    their original normalization. The same rule holds inside the iDM integral.
    """
    def __init__(self,inputs):
        md=json.loads((DATA/'manifest.json').read_text())
        for name,sha in md['sha256'].items():
            if digest(DATA/name)!=sha:raise ValueError('NA50 calibration checksum mismatch: '+name)
        self.fit=json.loads((DATA/'fit.json').read_text())
        with np.load(DATA/'proton_events.npz',allow_pickle=False) as z:self.proton=z['proton']
        if self.proton.dtype!=np.bool_ or self.proton.shape!=(md['count'],) or int(self.proton.sum())!=md['proton_count']:
            raise ValueError('Invalid NA50 proton identity mask')
        source=inputs.root/'results/idm_production/sources'
        uncertainty=inputs.root/'uncertainties'
        provenance=json.loads((source/'provenance.json').read_text())
        if provenance!=md['source_provenance'] or json.loads((uncertainty/'source_provenance.json').read_text())!=provenance:
            raise ValueError('NA50 proton identities belong to different production pools')
        package=json.loads((uncertainty/'manifest.json').read_text());self.ids={}
        for cat in ('primary','cascade'):
            key='dy',0,cat
            for k,mass in enumerate(inputs.knots[key]):
                name=f'dy_{mass:g}_0_{cat}_ids.npy';pool=f'dy_{mass:g}_0_{cat}.npz'
                if digest(uncertainty/name)!=package['sha256'][name] or digest(source/pool)!=package['source_pool_sha256'][pool]:
                    raise ValueError('NA50/source-row identity mismatch: '+pool)
                ids=np.load(uncertainty/name,allow_pickle=False)
                if ids.dtype.kind not in 'iu' or ids.shape!=(len(inputs.samples[key][k][0]),) or (len(ids) and (ids.min()<0 or ids.max()>=len(self.proton))):
                    raise ValueError('Invalid NA50 hard-event identities')
                self.ids[cat,k]=ids
        self.rows={tuple(row['label']):row for row in self.fit['calibrations']}
        self.central=float(self.rows['CT18NLO',0,1.,1.]['central'])
        self.metadata={'fit_sha256':md['sha256']['fit.json'],'proton_mask_sha256':md['original_mask_sha256'],
            'central_proton_DY_normalization':self.central,'profiles':list(PROFILES),
            'scope':self.fit['parameters'],'extrapolation':self.fit['extrapolation'],
            'covariance':self.fit['covariance'],'confidence':self.fit['confidence']}

    def coefficients(self,labels):
        if len(labels)!=len(self.rows) or {tuple(x) for x in labels}!=set(self.rows):
            raise ValueError('NA50 fit and PDF/scale labels differ')
        result=np.array([[self.rows[tuple(label)][profile] for label in labels] for profile in PROFILES])
        if np.any(~np.isfinite(result)) or np.any(result<=0):raise ValueError('Invalid NA50 coefficients')
        return result

    def sampled_protons(self,inputs,cat):
        out=np.zeros(len(inputs.current_dy_pool),dtype=bool)
        for k in np.unique(inputs.current_dy_pool):
            take=inputs.current_dy_pool==k;pool=self.ids[cat,k]
            if len(pool):out[take]=self.proton[pool[inputs.current_dy_row[take]]]
        return out
