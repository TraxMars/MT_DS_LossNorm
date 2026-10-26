"""Independent numeric and historical protection audit of completed C15."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from c15_common import ROOT,C1,STAGES,FEATURE_BASE,b,split,write_json
from algebra_c15 import norm,inner

def read(p):return json.loads(p.read_text())
def numpy_softplus(x):
    # Match torch.nn.functional.softplus(beta=1, threshold=20), including
    # its linear branch. Analytic logaddexp differs by ~2e-9 near 20.
    return np.where(x>20,x,np.logaddexp(0,x))
def run():
    checks={};runs={p:ROOT/'runs'/p for p in ('smoke','calibration','main')}
    checks['test_suites']=read(ROOT/'TEST_RESULTS.json')['status']=='PASS'
    checks['three_execution_gates']=all(read(p/'execution_results.json')['status']=='PASS' for p in runs.values())
    checks['state_rng_BN_grad_slots_Lonly']=all(all(read(p/'execution_results.json')['invariants'].values()) for p in runs.values())
    plan=read(runs['calibration']/'FROZEN_PLAN.json');checks['discovery_only_plan']=plan['discovery_cases']==split()['discovery'] and plan['probe_cases_not_opened_in_C15']
    checks['plan_hash_unchanged']=b.sha256(runs['calibration']/'FROZEN_PLAN.json')==read(runs['calibration']/'FROZEN_PLAN_SHA256.json')['sha256']==read(runs['main']/'source_at_run.json')['plan_sha256']
    eq=read(ROOT/'REDUCTION_EQUIVALENCE.json')
    checks['core_sources_match_runs']=all(b.sha256(ROOT/f)==read(runs[phase]/'source_at_run.json')['sources'][f] for phase in ('calibration','main') for f in ('run_c15.py','c15_common.py','targets_c15.py','algebra_c15.py','reductions_c15.py'))
    checks['smoke_reduction_equivalence']=eq['status']=='PASS' and all(b.sha256(runs['smoke']/'source_snapshot'/f)==eq['old_core'][f]==read(runs['smoke']/'source_at_run.json')['sources'][f] and b.sha256(ROOT/f)==eq['new_core'][f] for f in eq['old_core'])
    compatible=[r for p in runs.values() for r in read(p/'compatibility.json')]
    checks['A_C0_C1_compatibility']=all(r['absolute_error']<=1e-9 or r['relative_error']<=1e-4 for r in compatible)
    reconstruction=[r for phase in ('smoke','main') for r in read(runs[phase]/'reconstruction.json')]
    checks['direct_joint_LossNorm_autograd']=all(r['relative_error']<=1e-4 for r in reconstruction)
    matrix=pd.read_csv(ROOT/'summary/pair_geometry_and_marginals.csv')
    checks['true_pair_geometry_orientation']=len(matrix)==16 and (matrix.true_distance_mean==1).all()
    checks['permuted_endpoint_marginals']=all((matrix[c]==0).all() for c in ('bg_logit_wasserstein','bg_CE_wasserstein','bg_entropy_wasserstein'))
    contribution=pd.read_csv(runs['main']/'contributions.csv');errors=[];mask_errors=[];ce_errors=[]
    for index in sorted(contribution['index'].unique()):
        logits=np.load(runs['main']/'raw'/f'L_{index}_logits.npz');bands=logits['inner']|logits['outer'];y=logits['labels'][0]
        plus,minus=logits['plus'],logits['minus'];assert (y[tuple(plus.T)]==1).all() and (y[tuple(minus.T)]==0).all()
        z=logits['logits'][0,1].astype(float)-logits['logits'][0,0].astype(float)
        zp=z[tuple(plus.T)];zm=z[tuple(minus.T)];zq=z[tuple(logits['perm_minus'].T)]
        expected={'anchor':.5*(numpy_softplus(-zp)+numpy_softplus(zm)).mean(),'transition':numpy_softplus(1-zp+zm).mean(),'permuted':numpy_softplus(1-zp+zq).mean()}
        ce_errors.append(abs(expected['anchor']-.5*(np.logaddexp(0,-zp)+np.logaddexp(0,zm)).mean()))
        losses=pd.read_csv(runs['main']/'losses.csv');reference=losses[losses['index']==index].set_index('objective').value
        for n,v in expected.items():np.testing.assert_allclose(v,reference[n],rtol=1e-12,atol=1e-12)
        for s in STAGES:
            a=np.load(runs['main']/'raw/features'/f'L_{index}_{s}.npz');f=a['activation'];m=a['mask'];h,w,d=m.shape;fh,fw,fd=(n//q for n,q in zip(bands.shape,m.shape))
            expected_mask=bands.astype(float).reshape(h,fh,w,fw,d,fd).mean((1,3,5));mask_errors.append(float(abs(m-expected_mask).max()))
            for n in FEATURE_BASE:
                source=contribution[(contribution['index']==index)&(contribution.stage==s)&(contribution.objective==n)].sort_values('channel')
                observed=np.einsum('cijk,cijk,ijk->c',a[n],f,m,dtype=np.float64);error=norm(observed-source.inner.to_numpy())/max(norm(observed),1e-30);errors.append(error)
                np.testing.assert_allclose(observed,source.inner.to_numpy(),rtol=1e-9,atol=1e-12)
            del a
    checks['independent_raw_losses']=True;checks['feature_mask_native_average']=max(mask_errors)==0;checks['independent_signed_GxA']=max(errors)<1e-9
    checks['anchor_endpoint_CE_numeric_equivalence']=max(ce_errors)<2.1e-9
    for p in runs.values():
        norms=pd.read_csv(p/'parameter_norms.csv');diagonal=norms.assign(squared=norms.norm**2).groupby(['index','objective']).squared.sum()
        for item in read(p/'theta_grams.json'):
            for n,g in item['gram'].items():np.testing.assert_allclose(diagonal.loc[(item['index'],n)],g[n],rtol=1e-10,atol=1e-14)
    checks['all_parameter_gram_diagonal']=True
    interventions=pd.read_csv(runs['main']/'interventions.csv');valid=interventions.status=='valid'
    checks['fixed_intervention_count']=len(interventions)==384
    checks['energy_matching_valid']=np.allclose(interventions.loc[valid,'target_l2'],interventions.loc[valid,'actual_l2'],rtol=2e-5,atol=1e-5)
    checks['alpha_never_clipped']=interventions.loc[valid,'alpha'].between(0,.5).all()
    checks['no_E4_interventions']=set(interventions.stage)=={'E3','D3','D2','D1'}
    # Main outputs must use exactly the channels fixed before opening new probe data.
    checks['fixed_channels_no_probe_reselection']=all(set(rows.channel)==set(plan['channels'][s]) for s,rows in interventions.groupby('stage'))
    cal_con=pd.read_csv(runs['calibration']/'contributions.csv')
    for s,sources in plan['transfer_scores'].items():
        for c in plan['channels'][s]:
            base={n:cal_con[(cal_con.stage==s)&(cal_con.channel==c)&(cal_con.objective==n)].groupby('case').unit_response.mean().mean() for n in FEATURE_BASE}
            for name,values in sources.items():
                expected=base[name] if name in base else base['anchor']+plan['lambdas'][name.split('_')[1]]*base['transition' if name.startswith('true') else 'permuted']
                np.testing.assert_allclose(values[str(c)],expected,rtol=1e-12,atol=1e-14)
    checks['discovery_scores_reconstruct']=True
    protection=b.verify_protection(read(ROOT/'PROTECTION_BEFORE.json'));write_json(ROOT/'PROTECTION_AFTER.json',protection);checks['historical_sha256_1931']=protection['status']=='PASS' and protection['checked_files']==1931
    checks={k:bool(v) for k,v in checks.items()}
    result=dict(status='PASS' if all(checks.values()) else 'FAIL',checks=checks,compatibility_checks=len(compatible),direct_reconstruction_checks=len(reconstruction),
                max_feature_signed_GxA_relative_error=max(errors),max_mask_error=max(mask_errors),max_compatibility_relative_error=max(r['relative_error'] for r in compatible),
                max_anchor_endpoint_CE_difference=max(ce_errors),
                invalid_interventions=int((~valid).sum()),final_path_changed=int((interventions.loc[valid,'final_logit_l2']>0).sum()),
                history=protection,failed_smoke_preserved='runs/smoke_failed01/execution_results.json')
    write_json(ROOT/'VERIFICATION_RESULTS.json',result);assert result['status']=='PASS',result
    print(json.dumps(result,indent=2))

if __name__=='__main__':run()
