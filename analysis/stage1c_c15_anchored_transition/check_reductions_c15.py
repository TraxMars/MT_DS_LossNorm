"""Validate reduction optimization against the completed real smoke statistics."""
import json
import numpy as np
import pandas as pd
from c15_common import ROOT,STAGES,FEATURE_BASE,b,write_json
from reductions_c15 import feature_statistics,feature_cosine

def run():
    p=ROOT/'runs/smoke';assert json.loads((p/'execution_results.json').read_text())['status']=='PASS'
    con=pd.read_csv(p/'contributions.csv');relations=pd.read_csv(p/'channel_relations.csv');records=[]
    # Same 2L x 2 crops; all 5 nodes and all targets, including noncontiguous D1.
    for index in sorted(con['index'].unique()):
        for s in STAGES:
            a=np.load(p/'raw/features'/f'L_{index}_{s}.npz');f=a['activation'];m=a['mask'];cache={n:a[n] for n in FEATURE_BASE}
            for n,g in cache.items():
                stats=feature_statistics(g,f,m);ref=con[(con['index']==index)&(con.stage==s)&(con.objective==n)].sort_values('channel')
                for metric in ('inner','abs_gxa','gradient_norm'):
                    expected=ref[metric].to_numpy();err=float(np.max(abs(stats[metric]-expected)));rel=float(np.linalg.norm(stats[metric]-expected)/max(np.linalg.norm(expected),1e-30))
                    assert err<=1e-12 or rel<=1e-9
                    records.append(dict(index=int(index),stage=s,objective=n,metric=metric,absolute_error=err,relative_error=rel))
                assert (stats['status']==ref.status.to_numpy()).all()
            for x,y in [('anchor','transition'),('boundary','transition'),('anchor','permuted')]:
                cos,an,bn,status,dot=feature_cosine(cache[x],cache[y]);ref=relations[(relations['index']==index)&(relations.stage==s)&(relations.coordinate_kind=='feature')&(relations.left==x)&(relations.right==y)].sort_values('channel')
                np.testing.assert_allclose(cos,ref.cosine.to_numpy(),rtol=1e-9,atol=1e-12,equal_nan=True);np.testing.assert_array_equal(status,ref.status)
            del cache,a
            print('equivalent',index,s,flush=True)
    source=json.loads((p/'source_at_run.json').read_text())['sources']
    result=dict(status='PASS',observations=4,nodes_checked=20,statistics_checks=len(records),max_relative_error=max(r['relative_error'] for r in records),
                old_core={n:source[n] for n in ('run_c15.py','c15_common.py','targets_c15.py','algebra_c15.py')},
                new_core={n:b.sha256(ROOT/n) for n in ('run_c15.py','c15_common.py','targets_c15.py','algebra_c15.py','reductions_c15.py')},records=records)
    write_json(ROOT/'REDUCTION_EQUIVALENCE.json',result)

if __name__=='__main__':run()
