import numpy as np
from eventcalc.sensitivity import charges_from_pdg
def pack(events):
    if not events:raise ValueError('Empty complete-decay pool')
    a=np.zeros((len(events),max(map(len,events)),6))
    for i,event in enumerate(events):a[i,:len(event)]=event
    ids=a[...,5].astype(np.int64)
    visible=~np.isin(np.abs(ids),[0,12,14,16,18,4900101,4900102])
    return a[...,:4],ids,visible,charges_from_pdg(ids,visible=visible)


