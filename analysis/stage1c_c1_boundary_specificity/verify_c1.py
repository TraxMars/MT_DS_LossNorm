"""Independent numpy replay and frozen plan/history integrity, no data opens."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from bridge import ROOT,STAGES,PRIMARY,b,write_json,split

def main():
    checks=[];max_inner=0.;max_pred=0.;loss_error=0.;comparisons=0
    for phase in ('discovery','probe'):
        out=ROOT/'runs'/phase;execution=json.loads((out/'execution_results.json').read_text())
        assert execution['status']=='PASS' and all(execution['invariants'].values())
        data=pd.read_csv(out/'contributions.csv');local=pd.read_csv(out/'local_attribution.csv');actual=pd.read_csv(out/'interventions.csv')
        losses=pd.read_csv(out/'losses.csv')
        geometries=json.loads((out/'geometry.json').read_text())
        for geo in geometries:
            index=geo['index'];pairs=np.load(out/'raw'/f'L_{index}_pairs.npz');prediction=np.load(out/'raw'/f'L_{index}_prediction.npz')
            logits=prediction['logits'].astype(float)[0];y=prediction['labels'][0]
            z=logits[1]-logits[0];n=len(pairs['true_plus'])
            assert n>=32
            reference=losses[losses['index']==index].set_index('objective')['raw_loss'].to_dict()
            region=np.logaddexp(0,z)-y*z
            loss_error=max(loss_error,abs(region.mean()-reference['region']))
            inner=prediction['mask_B']&(y==1);outer=prediction['mask_B']&(y==0)
            boundary=.5*(region[inner].mean()+region[outer].mean())
            loss_error=max(loss_error,abs(boundary-reference['boundary']))
            for name in ('true','permuted','far','permuted_1','permuted_2','far_matched','true_common'):
                p,q=pairs[name+'_plus'],pairs[name+'_minus']
                if len(p):assert (y[tuple(p.T)]==1).all() and (y[tuple(q.T)]==0).all()
                if name=='true':assert (np.abs(p-q).sum(1)==1).all()
                if name.startswith('permuted'):
                    np.testing.assert_array_equal(p,pairs['true_plus'])
                    np.testing.assert_array_equal(np.sort(np.ravel_multi_index(q.T,y.shape)),np.sort(np.ravel_multi_index(pairs['true_minus'].T,y.shape)))
                if name=='true':names=[('transition',1.),('transition_m0_5',.5),('transition_m2',2.)]
                else:names=[(name,1.)]
                for objective,m in names:
                    if objective not in reference:continue
                    value=np.logaddexp(0,m-z[tuple(p.T)]+z[tuple(q.T)]).mean()
                    loss_error=max(loss_error,abs(value-reference[objective]))
            assert prediction['mask_B'].sum()==prediction['mask_I'].sum()
            for label in (0,1):assert (prediction['mask_B']&(y==label)).sum()==(prediction['mask_I']&(y==label)).sum()
            assert not (prediction['mask_B']&prediction['mask_I']).any()
            for s in STAGES:
                archive=np.load(out/'raw/features'/f'L_{index}_{s}.npz');f=archive['activation'].astype(float)[0]
                for r in ('B','I'):
                    m=archive['mask_'+r].astype(float)[0,0]
                    factor=tuple(a//c for a,c in zip(y.shape,m.shape))
                    expected=prediction['mask_'+r].reshape(m.shape[0],factor[0],m.shape[1],factor[1],m.shape[2],factor[2]).mean((1,3,5))
                    np.testing.assert_array_equal(m,expected)
                    energy=np.sqrt(np.einsum('cijk,cijk->c',f*m,f*m))
                    for name in PRIMARY:
                        gradient=archive['g_'+name].astype(float)
                        inner=np.einsum('cijk,cijk->c',gradient,f*m)
                        table=data[(data['index']==index)&(data.stage==s)&(data.support==r)&(data.objective==name)].sort_values('channel')
                        error=float(abs(inner-table.inner.to_numpy()).max());max_inner=max(max_inner,error)
                        np.testing.assert_allclose(inner,table.inner,rtol=1e-9,atol=1e-11)
                        np.testing.assert_allclose(energy,table.activation_energy,rtol=1e-12,atol=1e-10)
                        sub=local[(local['index']==index)&(local.stage==s)&(local.support==r)&(local.objective==name)&(local.status=='valid')]
                        for row in sub.itertuples(index=False):
                            match=actual[(actual['index']==index)&(actual.stage==s)&(actual.support==r)&(actual.channel==row.channel)&(actual.dose==row.dose)].iloc[0]
                            pred=-float(match.alpha)*inner[row.channel];max_pred=max(max_pred,abs(pred-row.delta_hat));comparisons+=1
                            assert abs(pred-row.delta_hat)<1e-10
            del archive,f,logits,y,prediction,pairs
        assert loss_error<1e-10
        valid=actual[actual.status=='valid'];energy_error=abs(valid.actual_l2-valid.target_l2)/valid.target_l2
        assert energy_error.max()<2e-5
        # Pointwise head with disjoint native supports has structural zeros.
        d1i=valid[(valid.stage=='D1')&(valid.support=='I')];assert (d1i.boundary_delta.abs()<1e-12).all()
        assert (valid.final_logit_l2>0).any()
        access=json.loads((out/'hdf5_access_log.json').read_text());assert all(r['role']=='labeled' for r in access)
        assert set(r['case'] for r in access)==set(split()[phase])
        replays=json.loads((out/'compatibility.json').read_text())
        checks.append(dict(phase=phase,model_invariants=execution['invariants'],crops=len(geometries),interventions=len(actual),
            energy_max_relative_error=float(energy_error.max()),final_logits_changed_fraction=float((valid.final_logit_l2>0).mean()),
            parameter_and_C0_replays=len(replays),reference_max_relative_error=max(r['relative_error'] for r in replays),L_only_access=True))
    plan=ROOT/'runs/discovery/FROZEN_PLAN.json';expected=json.loads((ROOT/'runs/discovery/FROZEN_PLAN_SHA256.json').read_text())['sha256']
    assert b.sha256(plan)==expected
    probe=json.loads((ROOT/'runs/probe/source_at_run.json').read_text());assert probe['frozen_plan_sha256_before_probe']==expected
    discovery_source=json.loads((ROOT/'runs/discovery/source_at_run.json').read_text())
    for name in ('bridge.py','control_pairs.py','targets.py','measures.py','runner.py'):
        assert discovery_source['sources'][name]==probe['sources'][name]==b.sha256(ROOT/name)
    prelaunch=json.loads((ROOT/'PROBE_PRELAUNCH.json').read_text())
    sensitivity_path=ROOT/'runs/discovery/FROZEN_SENSITIVITY.json'
    assert b.sha256(sensitivity_path)==prelaunch['sensitivity_sha256']
    assert prelaunch['primary_plan_sha256']==expected and prelaunch['probe_directory_absent']
    assert sensitivity_path.stat().st_mtime_ns<(ROOT/'runs/probe/source_at_run.json').stat().st_mtime_ns
    plan_data=json.loads(plan.read_text());assert plan_data['discovery_cases']==split()['discovery'] and not plan_data['probe_cases_opened']
    # Reconstruct every saved discovery score from raw contribution tables,
    # using only the common validity indices prescribed by the plan.
    d=pd.read_csv(ROOT/'runs/discovery/contributions.csv');d=d[d.objective.isin(PRIMARY)]
    for s in STAGES:
        for c in plan_data['stages'][s]['eligible']:
            block=d[(d.stage==s)&(d.channel==c)]
            valid=block.groupby('index').apply(lambda v:bool(((v.gradient_status=='valid')&(v.max_dose_status=='valid')&v.unit_response.notna()).all()),include_groups=False)
            indices=valid[valid].index;assert len(indices)>=6
            for r in ('B','I'):
                for n in PRIMARY:
                    rows=block[(block.support==r)&(block.objective==n)&block['index'].isin(indices)]
                    score=float(rows.groupby('case').unit_response.mean().mean())
                    assert abs(score-plan_data['scores'][s][r][n][str(c)])<1e-10
    history=b.verify_protection(json.loads((ROOT/'PROTECTION_BEFORE.json').read_text()));assert history['status']=='PASS'
    tests=json.loads((ROOT/'TEST_RESULTS.json').read_text());assert tests['status']=='PASS'
    result=dict(status='PASS',numpy_replayed_primary_predictions=comparisons,max_numpy_inner_product_error=max_inner,max_numpy_prediction_error=max_pred,
        max_numpy_raw_loss_error=loss_error,phases=checks,history=history,frozen_discovery_scores_reconstructed=True,
        probe_plan_hash_verified=True,unit_tests=tests['unittest_total'],C2_C3_executed=False,optimizer_steps=0,checkpoint_writes=0)
    write_json(ROOT/'VERIFICATION_RESULTS.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
