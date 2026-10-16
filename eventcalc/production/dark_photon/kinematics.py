import numpy as np

def transform(rows,m):
    rows=np.asarray(rows,dtype=np.float64)
    u,ct,p,ma,mb,rem=rows[:,:6].T;n=rows[:,6:9].copy();az=rows[:,9]
    # Compressed float32 directions must be normalized again: a tiny error
    # in |n| at high boost can otherwise exceed the light mediator mass^2.
    n/=np.linalg.norm(n,axis=1)[:,None]
    ein=np.hypot(p,ma);root=np.sqrt(ma*ma+mb*mb+2*mb*ein)
    q=u*np.sqrt(np.maximum((root*root-(m+rem)**2)*(root*root-(m-rem)**2),0))/(2*root)
    q=np.where(root>m+rem,q,0)
    qt=q*np.sqrt(np.maximum(1-ct*ct,0));qz=q*ct;ecm=np.hypot(m,q)
    gamma=(ein+mb)/root;beta=p/(ein+mb)
    e=gamma*(ecm+beta*qz);pz=gamma*(qz+beta*ecm)
    x=np.cross(n,np.array([0.,1.,0.]));x/=np.linalg.norm(x,axis=1)[:,None];y=np.cross(n,x)
    xyz=pz[:,None]*n+qt[:,None]*(np.cos(az)[:,None]*x+np.sin(az)[:,None]*y)
    beam=np.column_stack([p[:,None]*n,ein]);target=np.column_stack([np.zeros((len(rows),3)),mb])
    return np.column_stack([xyz,e]),root>m+rem,beam,target
