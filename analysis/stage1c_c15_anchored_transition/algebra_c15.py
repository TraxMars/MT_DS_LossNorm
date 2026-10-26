"""Validity-aware gradient budgets and optimization conflict quantities."""
import numpy as np

def norm(vector):return float(np.sqrt(np.einsum('i,i->',vector.reshape(-1),vector.reshape(-1),dtype=np.float64)))

def inner(a,b):return float(np.einsum('i,i->',a.reshape(-1),b.reshape(-1),dtype=np.float64))

def gram(vectors):
    return {a:{b:inner(x,y) for b,y in vectors.items()} for a,x in vectors.items()}

def combination_squared(g,weights):
    return max(0.,sum(va*vb*g[a][b] for a,va in weights.items() for b,vb in weights.items()))

def conflict(aa,tt,at,lam):
    if not np.isfinite([aa,tt,at,lam]).all():return dict(status='nonfinite')
    if aa<=1e-24 or tt<=1e-24:return dict(status='zero_or_below_threshold')
    a,t=np.sqrt(aa),np.sqrt(tt);cos=float(np.clip(at/(a*t),-1,1));joint=np.sqrt(max(0,aa+2*lam*at+lam*lam*tt))
    return dict(status='valid',dot=at,cosine=cos,anchor_norm=a,increment_norm=t,raw_norm_ratio=t/a,effective_increment_ratio=lam*t/a,
                joint_norm=joint,cancellation_fraction=1-joint/(a+lam*t),
                increment_orthogonal_fraction=float(np.sqrt(max(0,1-cos*cos))),
                predicted_anchor_rate=-(aa+lam*at),additional_anchor_rate=-lam*at,
                joint_may_increase_anchor=bool(aa+lam*at<0),critical_lambda=aa/(-at) if at<0 else None)

def calibrate(grams,budgets=(.1,.3)):
    ratios=[]
    for g in grams:
        a,t=g['anchor']['anchor'],g['transition']['transition']
        if not np.isfinite([a,t]).all() or min(a,t)<=1e-24:raise ValueError('Invalid discovery gradient norm')
        ratios.append(np.sqrt(t/a))
    if not ratios:raise ValueError('Empty calibration')
    lambdas={f'r{int(round(r*100)):02d}':r/max(ratios) for r in budgets}
    max_aux=0.
    for g in grams:
        if g['supervised']['supervised']<=1e-24:raise ValueError('Invalid supervised norm')
        candidates=[{'boundary':1.},{'anchor':1.}]
        for lam in lambdas.values():candidates.extend([{'anchor':1.,'transition':lam},{'anchor':1.,'permuted':lam}])
        max_aux=max(max_aux,max(np.sqrt(combination_squared(g,w)/g['supervised']['supervised']) for w in candidates))
    return dict(lambdas=lambdas,max_discovery_true_anchor_ratio=max(ratios),outer_beta=.10/max_aux,
                max_discovery_aux_sup_ratio=max_aux,outer_budget=.10,primary='r30')

def crossing(zplus,zminus):
    a,b=np.asarray(zplus),np.asarray(zminus);den=a-b
    valid=((a>0)&(b<=0))|((a<=0)&(b>0));valid &=abs(den)>1e-12
    where=np.full(a.shape,np.nan,dtype=float);where[valid]=a[valid]/den[valid]
    return where,valid
