import numpy as np

def transition(q,end=2.):
    t=np.clip((np.asarray(q)-1.5)/(end-1.5),0,1)
    return t*t*(3-2*t)
