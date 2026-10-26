"""Case-level C15 summaries from frozen outputs, without any model/data access."""
from itertools import combinations
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import wasserstein_distance
from c15_common import ROOT,C1,STAGES,FEATURE_BASE,INTERVENTION_STAGES,DOSES,split,write_json,write_csv,rho,bootstrap,direction,joint_names
from algebra_c15 import norm,inner

def read(path):return json.loads(path.read_text())
def cosine(a,b):
    den=norm(a)*norm(b)
    return inner(a,b)/den if den>1e-24 else None

def case_summaries(run,out):
    conflicts=pd.read_csv(run/'conflicts.csv');conflicts['partition']=np.where(conflicts.case.isin(split()['discovery']),'discovery','probe')
    columns=['cosine','dot','raw_norm_ratio','effective_increment_ratio','cancellation_fraction','increment_orthogonal_fraction','predicted_anchor_rate','additional_anchor_rate','boundary_rate','additional_boundary_rate']
    case=conflicts.groupby(['partition','case','coordinate_kind','stage','increment','budget'])[columns].mean().reset_index()
    case.to_csv(out/'gradient_case_means.csv',index=False)
    records=[];draws=np.random.default_rng(142).integers(0,4,(2000,4))
    for keys,rows in case.groupby(['partition','coordinate_kind','stage','increment','budget']):
        rows=rows.set_index('case').reindex(split()[keys[0]])
        for c in columns:
            stat=bootstrap(rows[c].values,draws);records.append(dict(zip(['partition','coordinate_kind','stage','increment','budget'],keys),metric=c,**stat))
    write_json(out/'gradient_case_bootstrap.json',records)
    channel=pd.read_csv(run/'channel_relations.csv');table=[]
    for keys,rows in channel.groupby(['case','stage','coordinate_kind','left','right']):
        valid=rows.status=='valid';r=dict(zip(['case','stage','coordinate_kind','left','right'],keys),observations=len(rows),valid_fraction=float(valid.mean()))
        r.update({k+'_fraction':float((rows.loc[valid,'dot_sign']==k).mean()) if valid.any() else None for k in ('positive','negative','near_orthogonal')})
        table.append(r)
    write_csv(out/'channel_dot_signs.csv',table)
    globals=[]
    for keys,rows in channel.groupby(['case','index','repeat','stage','coordinate_kind','left','right']):
        an=float(np.sqrt((rows.left_norm**2).sum()));bn=float(np.sqrt((rows.right_norm**2).sum()));dot=float(rows['dot'].sum())
        globals.append(dict(zip(['case','index','repeat','stage','coordinate_kind','left','right'],keys),left_norm=an,right_norm=bn,dot=dot,
                            cosine=dot/(an*bn) if min(an,bn)>1e-12 else None,status='valid' if min(an,bn)>1e-12 else 'below_threshold'))
    write_csv(out/'gradient_global_pairs.csv',globals)
    return case

def pairs(run,out):
    rows=[];groups=[];far=[]
    for oldpart in ('discovery','probe'):
        oldloss=pd.read_csv(C1/'runs'/oldpart/'losses.csv')
        for geo in read(C1/'runs'/oldpart/'geometry.json'):
            index=geo['index'];path=run/'raw'/f'L_{index}_logits.npz'
            if not path.exists():continue
            a=np.load(path);z=a['logits'][0,1].astype(float)-a['logits'][0,0].astype(float);label=a['labels'][0]
            plus,minus,perm=a['plus'],a['minus'],a['perm_minus'];zp=z[tuple(plus.T)];zm=z[tuple(minus.T)];zq=z[tuple(perm.T)]
            assert np.all(label[tuple(plus.T)]==1) and np.all(label[tuple(minus.T)]==0)
            np.testing.assert_array_equal(np.sort(np.ravel_multi_index(minus.T,z.shape)),np.sort(np.ravel_multi_index(perm.T,z.shape)))
            r=dict(case=geo['case'],index=index,repeat=geo['repeat'],partition=oldpart,count=len(plus),true_distance_mean=float(np.linalg.norm(plus-minus,axis=1).mean()),
                perm_distance_mean=float(np.linalg.norm(plus-perm,axis=1).mean()),perm_distance_median=float(np.median(np.linalg.norm(plus-perm,axis=1))),
                exact_same_pair_fraction=float((minus==perm).all(1).mean()),fg_unique=len(np.unique(plus,axis=0)),bg_unique=len(np.unique(minus,axis=0)),
                foreground_count=len(plus),background_count=len(minus),bg_logit_wasserstein=float(wasserstein_distance(zm,zq)),
                bg_CE_wasserstein=float(wasserstein_distance(np.logaddexp(0,zm),np.logaddexp(0,zq))),bg_entropy_wasserstein=float(wasserstein_distance(entropy(zm),entropy(zq))),
                true_gap_mean=float((zp-zm).mean()),true_gap_std=float((zp-zm).std()),perm_gap_mean=float((zp-zq).mean()),perm_gap_std=float((zp-zq).std()),
                true_loss=float(np.logaddexp(0,1-zp+zm).mean()),perm_loss=float(np.logaddexp(0,1-zp+zq).mean()),
                anchor_loss=float((.5*(np.logaddexp(0,-zp)+np.logaddexp(0,zm))).mean()),
                margin_satisfied_fraction=float((zp-zm>=1).mean()),margin_satisfied_wrong_fraction=float(((zp-zm>=1)&((zp<=0)|(zm>0))).mean()),
                wrong_given_satisfied=float(((zp<=0)|(zm>0))[zp-zm>=1].mean()) if (zp-zm>=1).any() else None)
            for objective in ('anchor','transition','permuted'):
                grad=a['gradient_'+objective];r[objective+'_endpoint_gnorm']=norm(grad)
                r[objective+'_plus_correct_gradient_fraction']=float((grad[tuple(plus.T)]<0).mean());r[objective+'_minus_correct_gradient_fraction']=float((grad[tuple(minus.T)]>0).mean())
            r['true_perm_logit_cosine']=cosine(a['gradient_transition'],a['gradient_permuted'])
            r['true_perm_relative_gradient_difference']=norm(a['gradient_transition']-a['gradient_permuted'])/max(norm(a['gradient_transition']),1e-30)
            rows.append(r)
            # Supplement only: old C1 far-control geometry/matching diagnostics.
            far.append(dict(case=geo['case'],index=index,partition=oldpart,source='unchanged C1 archive; not recalibrated',
                            matching=geo.get('far_matching',geo.get('matching',{})),pair_geometry=geo.get('pairs',{}).get('far',{}),
                            old_raw_losses={n:float(oldloss[(oldloss['index']==index)&(oldloss.objective==n)].raw_loss.iloc[0]) for n in ('far','far_matched','true_common')}))
    write_csv(out/'pair_geometry_and_marginals.csv',rows);write_json(out/'far_reference.json',far)
    write_csv(out/'endpoint_class_groups.csv',read(run/'endpoint_groups.json'))
    return pd.DataFrame(rows)

def entropy(z):
    p=expit(z);return -(p*np.log(np.maximum(p,1e-300))+(1-p)*np.log(np.maximum(1-p,1e-300)))

def stability(run,out,plan):
    table=[];cb=pd.read_csv(run/'contributions.csv')
    for part,cases in split().items():
        for s in STAGES:
            for objective in FEATURE_BASE:
                ps={};fs={}
                for case in cases:
                    indices=sorted(cb[cb.case==case]['index'].unique());pv=[];fv=[]
                    for index in indices:
                        pv.append(np.load(run/'raw/parameters'/f'L_{index}_{s}.npz')[objective])
                        rs=cb[(cb['index']==index)&(cb.stage==s)&(cb.objective==objective)].sort_values('channel');fv.append(rs.unit_response.to_numpy())
                    for kind,arr in [('parameter',pv),('spatially_reduced_feature_response',fv)]:
                        table.append(dict(partition=part,stage=s,objective=objective,coordinate_kind=kind,comparison='within_case_crops',case_a=case,case_b=case,cosine=cosine(arr[0],arr[1])))
                    ps[case]=np.mean(pv,axis=0);fs[case]=np.mean(fv,axis=0)
                for a,c in combinations(cases,2):
                    for kind,values in [('parameter',ps),('spatially_reduced_feature_response',fs)]:table.append(dict(partition=part,stage=s,objective=objective,coordinate_kind=kind,comparison='between_cases',case_a=a,case_b=c,cosine=cosine(values[a],values[c])))
    write_csv(out/'case_crop_stability.csv',table)

def attribution(run,out,plan):
    data=pd.read_csv(run/'attribution.csv');actual=pd.read_csv(run/'interventions.csv');loss=pd.read_csv(ROOT/'runs/calibration/losses.csv')
    # Discovery-only numerical floors, not probe-derived score calibration.
    eps={}
    for name,(source,lam) in joint_names(plan).items():
        source_values=loss[loss.objective==source].set_index('index').value
        values=source_values+lam*0
        if lam:values=loss[loss.objective=='anchor'].set_index('index').value+lam*source_values
        eps[name]=max(1e-9,1e-6*max(float(abs(values).max()),1e-3))
    case_records=[];raw_predictions=[]
    metrics=('boundary_delta','endpoint_error_delta','boundary_f1_delta','region_delta','dice_delta','changed_prediction_fraction')
    for case in split()['probe']:
        for stage in INTERVENTION_STAGES:
            for dose in DOSES:
                a=actual[(actual.case==case)&(actual.stage==stage)&(actual.dose==dose)].set_index(['index','channel'])
                for objective in joint_names(plan):
                    r=data[(data.case==case)&(data.stage==stage)&(data.dose==dose)&(data.objective==objective)].copy()
                    valid=(r.status=='valid');r=r[valid];channels=plan['channels'][stage]
                    if len(r)!=2*len(channels):raise ValueError('Incomplete probe interventions; do not silently remove channels')
                    for method in ('discovery_fixed','local_GT_attribution'):
                        p=[];threshold=[];truth={m:[] for m in metrics};boundary_eps=[]
                        for c in channels:
                            rows=r[r.channel==c].sort_values('index');ar=a.loc[[(i,c) for i in rows['index']]]
                            if method=='local_GT_attribution':pred=float(rows.delta_hat.mean());peps=float(rows.epsilon.mean())
                            else:pred=float(plan['transfer_scores'][stage][objective][str(c)]*ar.target_l2.mean());peps=eps[objective]
                            p.append(pred);threshold.append(peps);boundary_eps.append(float(ar.boundary_epsilon.mean()))
                            for metric in metrics:truth[metric].append(float(ar[metric].mean()))
                            raw_predictions.append(dict(case=case,stage=stage,dose=dose,objective=objective,method=method,channel=c,prediction=pred,prediction_epsilon=peps,
                                                        **{m:truth[m][-1] for m in metrics},boundary_epsilon=boundary_eps[-1]))
                        pdirection=[direction(v,e) for v,e in zip(p,threshold)];adirection=[direction(v,e) for v,e in zip(truth['boundary_delta'],boundary_eps)]
                        determinate=np.array([x in ('positive','negative') and y in ('positive','negative') for x,y in zip(pdirection,adirection)])
                        raw_signed,raw_status=rho(p,truth['boundary_delta']);raw_absolute,raw_astatus=rho(abs(np.array(p)),abs(np.array(truth['boundary_delta'])))
                        signed,status=rho(np.array(p)[determinate],np.array(truth['boundary_delta'])[determinate])
                        absolute,astatus=rho(abs(np.array(p)[determinate]),abs(np.array(truth['boundary_delta'])[determinate]))
                        record=dict(case=case,stage=stage,dose=dose,objective=objective,method=method,channels=len(p),signed_rho=signed,signed_status=status,abs_rho=absolute,abs_status=astatus,
                            determinate_channels=int(determinate.sum()),raw_signed_rho=raw_signed,raw_signed_status=raw_status,raw_abs_rho=raw_absolute,raw_abs_status=raw_astatus,
                            direction_coverage=float(determinate.mean()),actual_direction_determinate_fraction=float(np.mean([d in ('positive','negative') for d in adirection])),
                            predicted_direction_determinate_fraction=float(np.mean([d in ('positive','negative') for d in pdirection])),
                            direction_accuracy=float(np.mean(np.array(pdirection)[determinate]==np.array(adirection)[determinate])) if determinate.any() else None)
                        for metric in metrics:
                            value,status=rho(p,truth[metric]);record[metric+'_rho']=value;record[metric+'_rho_status']=status
                        case_records.append(record)
    write_csv(out/'attribution_probe_case.csv',case_records);write_csv(out/'attribution_probe_predictions.csv',raw_predictions)
    frame=pd.DataFrame(case_records);records=[];draws=np.random.default_rng(142).integers(0,4,(2000,4))
    for keys,rows in frame.groupby(['stage','dose','objective','method']):
        rows=rows.set_index('case').reindex(split()['probe'])
        for metric in ('signed_rho','abs_rho','raw_signed_rho','raw_abs_rho','direction_coverage','direction_accuracy','boundary_f1_delta_rho','endpoint_error_delta_rho','dice_delta_rho'):
            records.append(dict(zip(['stage','dose','objective','method'],keys),metric=metric,available_cases=int(rows[metric].notna().sum()),**bootstrap(rows[metric].to_numpy(),draws)))
    write_json(out/'attribution_probe_bootstrap.json',records)
    paired=[]
    for stage in INTERVENTION_STAGES:
        for dose in DOSES:
            group=frame[(frame.stage==stage)&(frame.dose==dose)&(frame.method=='discovery_fixed')]
            for a,c in [('true_r30','anchor'),('true_r30','perm_r30'),('anchor','boundary')]:
                for metric in ('signed_rho','abs_rho','direction_accuracy'):
                    x=group[group.objective==a].set_index('case').reindex(split()['probe'])[metric].to_numpy();y=group[group.objective==c].set_index('case').reindex(split()['probe'])[metric].to_numpy()
                    paired.append(dict(stage=stage,dose=dose,left=a,right=c,metric=metric,**bootstrap(x-y,draws)))
    write_json(out/'paired_method_differences.json',paired)
    return frame,actual

def plots(case,probe,actual,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.5))
    for kind,marker in [('feature','o'),('parameter','s')]:
        a=case[(case.coordinate_kind==kind)&(case.increment=='transition')&(case.budget=='r30')&(case.partition=='probe')]
        for i,s in enumerate(STAGES):axes[0].scatter(np.full(len(a[a.stage==s]),i)+(0 if kind=='feature' else .12),a[a.stage==s].cosine,marker=marker,label=kind if i==0 else None,alpha=.7)
    axes[0].set(xticks=range(5),xticklabels=STAGES,ylabel='cos(Anchor, True)',ylim=(-1.05,1.05));axes[0].legend(fontsize=8)
    names=['boundary','anchor','true_r30','perm_r30']
    for j,name in enumerate(names):
        a=probe[(probe.objective==name)&(probe.dose==.03)&(probe.method=='discovery_fixed')]
        values=[a[a.stage==s].signed_rho.mean() for s in INTERVENTION_STAGES]
        axes[1].plot(range(4),values,'o-',label=name)
    axes[1].set(xticks=range(4),xticklabels=INTERVENTION_STAGES,ylabel='Probe case-mean signed rho',ylim=(-1.05,1.05));axes[1].legend(fontsize=7)
    for dose,marker in [(.01,'o'),(.03,'x')]:
        a=actual[(actual.dose==dose)&actual.case.isin(split()['probe'])];axes[2].scatter(a.boundary_delta,a.boundary_f1_delta,s=15,marker=marker,label=str(dose),alpha=.6)
    axes[2].set(xlabel='Actual Boundary CE change',ylabel='Contour F1 change');axes[2].legend(title='dose',fontsize=8)
    fig.tight_layout();fig.savefig(out/'C15_diagnostics.png',dpi=180);fig.savefig(out/'C15_diagnostics.pdf');plt.close(fig)

def lossnorm_reference(run,out):
    rows=[];a=pd.read_csv(run/'lossnorm.csv');mt=pd.read_csv(run/'historical_MT_reference.csv')
    for (stage,objective),r in a[a.coordinate_kind=='parameter'].groupby(['stage','objective']):
        u=mt[mt.stage==stage]
        rows.append(dict(stage=stage,objective=objective,new_L_cases=8,archived_U_cases=2,archived_U_crops=4,
            raw_supervised_norm_mean=float(r.raw_sup_norm.mean()),raw_auxiliary_norm_mean=float(r.raw_aux_norm.mean()),
            weighted_aux_to_sup_raw_mean=float(r.aux_to_sup_raw_ratio.mean()),joint_normalized_norm_mean=float(r.joint_normalized_norm.mean()),
            normalized_aux_component_mean=float(r.joint_aux_component_norm.mean()),normalized_sup_component_mean=float(r.joint_sup_component_norm.mean()),
            original_MT_raw_norm_mean=float(u.raw_norm.mean()),original_MT_normalized_weighted_norm_mean=float(u.normalized_norm.mean()),
            original_MT_normalized_weighted_norm_min=float(u.normalized_norm.min()),original_MT_normalized_weighted_norm_max=float(u.normalized_norm.max()),
            scope='different archived U and new L crops; descriptive stage norms only, not new joint batch or train-mode validation'))
    write_csv(out/'lossnorm_MT_stage_reference.csv',rows)

def main():
    run=ROOT/'runs/main';assert read(run/'execution_results.json')['status']=='PASS';out=ROOT/'summary';out.mkdir(exist_ok=False)
    plan=read(ROOT/'runs/calibration/FROZEN_PLAN.json');case=case_summaries(run,out);pair=pairs(run,out);stability(run,out,plan);probe,actual=attribution(run,out,plan);plots(case,probe,actual,out)
    lnorm=pd.read_csv(run/'lossnorm.csv');lnorm.groupby(['coordinate_kind','stage','objective'])[['aux_to_sup_raw_ratio','joint_normalized_norm','independent_aux_to_joint_aux_ratio']].agg(['min','mean','max']).to_csv(out/'lossnorm_ranges.csv')
    lossnorm_reference(run,out)
    write_json(out/'SUMMARY.json',dict(status='PASS',cases=8,crops=16,physical_interventions=len(actual),valid_interventions=int((actual.status=='valid').sum()),
        lambda_r10=plan['lambdas']['r10'],lambda_r30=plan['lambdas']['r30'],outer_beta=plan['outer_beta'],
        wrong_given_satisfied=float(pair.wrong_given_satisfied.mean()),pair_geometry_table='pair_geometry_and_marginals.csv',inference='exploratory 4-case probe, no independent generalization claim'))
    print('C15 summaries PASS',flush=True)

if __name__=='__main__':main()
