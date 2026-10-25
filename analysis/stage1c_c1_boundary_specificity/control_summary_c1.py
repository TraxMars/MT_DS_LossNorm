"""Independent endpoint marginal/difficulty audit from archived final logits."""
import json
import numpy as np
from scipy.stats import wasserstein_distance
from bridge import ROOT,write_csv

def main():
    rows=[]
    for part in ('discovery','probe'):
        run=ROOT/'runs'/part
        for geo in json.loads((run/'geometry.json').read_text()):
            i=geo['index'];arrays=np.load(run/'raw'/f'L_{i}_prediction.npz');pairs=np.load(run/'raw'/f'L_{i}_pairs.npz')
            logits=arrays['logits'][0].astype(float);z=logits[1]-logits[0]
            for control,reference in [('permuted','true'),('permuted_1','true'),('permuted_2','true'),('far','true'),('far_matched','true_common')]:
                for side,sign in [('plus',-1),('minus',1)]:
                    coords=pairs[control+'_'+side];ref=pairs[reference+'_'+side]
                    if len(coords)<32:continue
                    a=z[tuple(ref.T)];c=z[tuple(coords.T)]
                    def metrics(v):
                        p=1/(1+np.exp(-np.clip(v,-700,700)))
                        return dict(logit=v,ce=np.logaddexp(0,sign*v),entropy=np.logaddexp(0,v)-p*v)
                    aa,cc=metrics(a),metrics(c)
                    row=dict(split=part,case=geo['case'],index=i,repeat=geo['repeat'],control=control,reference=reference,side=side,
                        reference_count=len(a),control_count=len(c),reference_error=float(((a<=0) if sign==-1 else (a>0)).mean()),
                        control_error=float(((c<=0) if sign==-1 else (c>0)).mean()),
                        exact_endpoint_multisets=np.array_equal(np.sort(np.ravel_multi_index(coords.T,z.shape)),np.sort(np.ravel_multi_index(ref.T,z.shape))))
                    for name in aa:
                        row[name+'_wasserstein']=float(wasserstein_distance(aa[name],cc[name]));row[name+'_reference_mean']=float(aa[name].mean());row[name+'_control_mean']=float(cc[name].mean())
                    if control=='far_matched':row['paired_logit_max_discrepancy']=float(abs(a-c).max())
                    rows.append(row)
    write_csv(ROOT/'endpoint_difficulty_matching.csv',rows);print('Endpoint geometry/difficulty audit:',len(rows),'rows')

if __name__=='__main__':main()
