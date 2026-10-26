"""Read-only reuse of complete C1/C0, with C15-only output guards."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
C1=PROJECT/'analysis/stage1c_c1_boundary_specificity'
sys.path.insert(0,str(C1))
from bridge import (C0,b,b_losses,STAGES,write_json,write_csv,fixed_rng,Capture,attenuate,NODE_SPECS,
                    adjacent_pairs,permuted_partners,PairSet,calibrate_epsilon,direction,dose_alpha,node_statistics,
                    Objectives,transition,prediction_metrics,surface6,split)
from measures import cosine_rows,rho,bootstrap
from objectives_c0 import sample,check_final
import torch

BASE=('boundary','anchor','transition','permuted','region','supervised')
FEATURE_BASE=BASE[:-1]
INTERVENTION_STAGES=('E3','D3','D2','D1')
DOSES=(.01,.03)
BUDGETS=(.10,.30)

def output_path(path):
    p=Path(path).resolve()
    if p==ROOT/'runs' or not p.is_relative_to(ROOT/'runs'):raise ValueError('C15 output must be an independent runs child')
    return p

def fixed_channels():
    import json
    old=json.loads((C1/'runs/discovery/FROZEN_PLAN.json').read_text());result={}
    for s in INTERVENTION_STAGES:
        choices=old['stages'][s]['channels'];eligible=old['stages'][s]['eligible'];n=len(eligible)
        result[s]=[c for c in (0,n//2,n-1) if c in choices]
        result[s]+=[c for c in choices if c not in result[s]][:3-len(result[s])]
        assert len(result[s])==3 and set(result[s])<=set(eligible)
    return result

def gradient_status(norms):
    import numpy as np
    n=np.asarray(norms);floor=max(1e-12,1e-6*float(np.nanmax(n,initial=0)))
    return np.where(~np.isfinite(n),'nonfinite',np.where(n==0,'exact_zero',np.where(n>floor,'valid','below_threshold')))

def joint_names(plan):
    names={'boundary':('boundary',0.),'anchor':('anchor',0.),'transition':('transition',0.),'permuted':('permuted',0.),'region':('region',0.)}
    for tag,lam in plan['lambdas'].items():
        names['true_'+tag]=('transition',lam);names['perm_'+tag]=('permuted',lam)
    return names

def combine_terms(base,plan):
    result={n:base[n] for n in FEATURE_BASE}
    for n,(source,lam) in joint_names(plan).items():
        if lam:result[n]=base['anchor']+lam*base[source]
    return result

def archived_aggregate(array,stage):
    """An unreachable head is NaN in A, but contributes no graph path to mean4.

    Never replace a reachable zero/nonfinite gradient with a valid signal.
    """
    import numpy as np
    a=np.asarray(array);heads=('D4','D3','D2','D1')
    reachable=heads if stage.startswith('E') else heads[heads.index(stage):]
    if a.shape[0]!=4:raise ValueError('Expected four archived heads')
    result=np.zeros_like(a[0])
    for i,head in enumerate(heads):
        if head in reachable:
            if not np.isfinite(a[i]).all():raise ValueError('Nonfinite REACHABLE archived gradient')
            result+=a[i]/4
        elif not np.isnan(a[i]).all():raise ValueError('Unexpected reachable status in archived head')
    return result
