"""Frozen L-only C15: smoke, discovery calibration, then one fixed 8L run."""
import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
from c15_common import (ROOT,C1,C0,b,b_losses,STAGES,BASE,FEATURE_BASE,DOSES,INTERVENTION_STAGES,write_json,write_csv,split,
                        Capture,attenuate,adjacent_pairs,permuted_partners,calibrate_epsilon,direction,dose_alpha,
                        prediction_metrics,fixed_channels,output_path,gradient_status,joint_names,combine_terms,cosine_rows,archived_aggregate)
from targets_c15 import Targets
from algebra_c15 import norm,inner,gram,calibrate,conflict,combination_squared,crossing
from objectives import normalize_loss


def base_context(student,x,y,label,dist,meta,obs,device):
    with Capture(student) as capture:outputs,features=student(x,deepsup=True,return_features=True)
    identity=capture.verify(features)
    bands,infos=b_losses.labeled_regions(label,dist,meta,2,142,obs['case'],obs['repeat'])
    assert infos['B']['status']=='valid'
    pairs=adjacent_pairs(y[0].cpu().numpy(),b.crop_native(np.ones(label.shape,bool),meta),seed_parts=(obs['case'],obs['repeat'],'L'))
    perm=permuted_partners(pairs,(obs['case'],obs['repeat'],'C1',0));assert pairs.valid
    part='discovery' if obs['case'] in split()['discovery'] else 'probe'
    old=np.load(C1/'runs'/part/'raw'/f'L_{obs["index"]}_pairs.npz')
    for name,p in [('true',pairs),('permuted',perm)]:
        np.testing.assert_array_equal(p.plus,old[name+'_plus']);np.testing.assert_array_equal(p.minus,old[name+'_minus'])
    assert np.all(y[0].cpu().numpy()[tuple(pairs.plus.T)]==1) and np.all(y[0].cpu().numpy()[tuple(pairs.minus.T)]==0)
    targets=Targets(pairs,perm,y,bands['B']);base=targets.base(outputs[-1])
    heads,sup,_=b.independent_losses(outputs,labels=y)
    np.testing.assert_allclose([float(t.detach()) for t in heads],obs['losses'],atol=1e-7,rtol=1e-5)
    base['supervised']=sup[None]
    reference=json.loads((C1/'runs'/part/'geometry.json').read_text());oldgeo=next(a for a in reference if a['index']==obs['index'])
    masks={s:b_losses.pool_native(bands['B'][0]|bands['B'][1],capture.nodes[s].shape[2:],device)[0].detach() for s in STAGES}
    energies={s:(capture.nodes[s].detach().double()*masks[s].double()).square().sum((0,2,3,4)).sqrt().cpu().numpy() for s in STAGES}
    q={s:oldgeo['coarse_masks'][s]['q'] for s in STAGES}
    return dict(outputs=outputs,features=capture.nodes,identity=identity,pairs=pairs,perm=perm,targets=targets,base=base,
                regions=bands['B'],masks=masks,energies=energies,q=q,old_geometry=oldgeo,old_part=part)


def extract(student,ctx,obs,mapping,out,tables,save_raw):
    key={k:obs[k] for k in ('case','index','repeat')};named=list(student.named_parameters());locations={n:i for i,(n,p) in enumerate(named)}
    features={s:{} for s in STAGES};parameters={s:{} for s in STAGES};vectors={};logit_grad={}
    for name in BASE:
        inputs=([ctx['features'][s] for s in STAGES]+[ctx['outputs'][-1]] if name!='supervised' else [])+[p for n,p in named]
        gs=torch.autograd.grad(ctx['base'][name][0],inputs,retain_graph=True,allow_unused=True)
        offset=6 if name!='supervised' else 0;pg=gs[offset:];pieces=[]
        for (pname,p),g in zip(named,pg):
            value=np.zeros(p.numel(),np.float32) if g is None else g.detach().cpu().numpy().reshape(-1)
            pieces.append(value)
            tables['parameter_norms'].append(dict(key,objective=name,parameter=pname,shape=list(p.shape),norm=norm(value),
                                                   status='unreachable' if g is None else 'exact_zero' if not np.any(value) else 'finite'))
        vectors[name]=np.concatenate(pieces)
        for s in STAGES:
            g=pg[locations[mapping[s].name]];assert g is not None and g.shape==mapping[s].parameter.shape
            parameters[s][name]=mapping[s].channel_vectors(g).detach().cpu().numpy()
        if name!='supervised':
            for s,g in zip(STAGES,gs[:5]):
                assert g is not None and g.shape==ctx['features'][s].shape
                features[s][name]=g.detach().cpu().numpy()[0]
            assert gs[5] is not None
            logit_grad[name]=gs[5].detach().cpu().numpy()[0,1]
        del gs,pg,inputs
    full_gram=gram(vectors);del vectors
    stats={};relations=[]
    for s in STAGES:
        f=ctx['features'][s].detach().cpu().numpy()[0].astype(float);m=ctx['masks'][s].cpu().numpy()[0,0].astype(float)
        stats[s]={}
        for name,g in features[s].items():
            gn=np.sqrt(np.einsum('cijk,cijk->c',g,g,dtype=np.float64));product=g.astype(float)*f*m
            stats[s][name]=dict(inner=product.sum((1,2,3)),abs_gxa=abs(product).sum((1,2,3)),gradient_norm=gn,status=gradient_status(gn))
            for c in range(len(gn)):
                tables['contributions'].append(dict(key,stage=s,objective=name,channel=c,inner=stats[s][name]['inner'][c],abs_gxa=stats[s][name]['abs_gxa'][c],
                    gradient_norm=gn[c],activation_energy=ctx['energies'][s][c],unit_response=-stats[s][name]['inner'][c]/ctx['energies'][s][c] if ctx['energies'][s][c]>1e-12 else None,status=stats[s][name]['status'][c]))
        for a,c in [('anchor','transition'),('boundary','transition'),('anchor','permuted')]:
            for kind,source in [('feature',features[s]),('parameter',parameters[s])]:
                x,y=source[a],source[c];cos,an,bn,status=cosine_rows(x,y)
                dot=np.einsum('ci,ci->c',x.reshape(len(x),-1),y.reshape(len(y),-1),dtype=np.float64)
                for channel,value in enumerate(cos):relations.append(dict(key,stage=s,coordinate_kind=kind,left=a,right=c,channel=channel,cosine=value,
                    dot=dot[channel],left_norm=an[channel],right_norm=bn[channel],status=status[channel],
                    dot_sign='undefined' if status[channel]!='valid' else 'positive' if value>1e-6 else 'negative' if value< -1e-6 else 'near_orthogonal'))
        if save_raw:
            np.savez(out/'raw/features'/f'L_{obs["index"]}_{s}.npz',activation=f.astype(np.float32),mask=m.astype(np.float32),**features[s])
            np.savez(out/'raw/parameters'/f'L_{obs["index"]}_{s}.npz',**parameters[s])
    tables['channel_relations'].extend(relations)
    tables['theta_grams'].append(dict(key,gram=full_gram,coordinate_kind='all_existing_parameters',parameter_count=len(named)))
    if save_raw:
        np.savez(out/'raw'/f'L_{obs["index"]}_logits.npz',logits=ctx['outputs'][-1].detach().cpu().numpy(),labels=ctx['targets'].original.labels.cpu().numpy(),
            plus=ctx['pairs'].plus,minus=ctx['pairs'].minus,perm_minus=ctx['perm'].minus,inner=ctx['regions'][0],outer=ctx['regions'][1],**{'gradient_'+n:g for n,g in logit_grad.items()})
    return dict(feature=features,parameter=parameters,all=full_gram,logit=logit_grad,stats=stats)


def relation_tables(ctx,data,obs,plan,tables):
    key={k:obs[k] for k in ('case','index','repeat')}
    domains=[('all_parameters','theta',data['all'])]
    for s in STAGES:
        domains.append(('parameter',s,gram(data['parameter'][s])))
        domains.append(('feature',s,gram(data['feature'][s])))
    for kind,s,g in domains:
        for source in ('transition','permuted'):
            for tag,lam in plan['lambdas'].items():
                r=conflict(g['anchor']['anchor'],g[source][source],g['anchor'][source],lam)
                r.update(key,coordinate_kind=kind,stage=s,increment=source,budget=tag,lambda_value=lam,
                         boundary_rate=-(g['boundary']['anchor']+lam*g['boundary'][source]),additional_boundary_rate=-lam*g['boundary'][source])
                tables['conflicts'].append(r)
    for s in STAGES:
        for name,(source,lam) in joint_names(plan).items():
            if not lam:continue
            a=data['stats'][s]['anchor'];t=data['stats'][s][source]
            for c in range(len(a['inner'])):
                energy=ctx['energies'][s][c];value=a['inner'][c]+lam*t['inner'][c]
                tables['joint_contributions'].append(dict(key,stage=s,objective=name,channel=c,unit_response=-value/energy if energy>1e-12 else None,
                    inner=value,source_status='valid' if a['status'][c]=='valid' and t['status'][c]=='valid' else 'source_invalid'))


def compatibility(data,ctx,obs,tables,full_feature=False):
    key={k:obs[k] for k in ('case','index','repeat')}
    oldpart=ctx['old_part'];oldloss=__import__('csv').DictReader((C1/'runs'/oldpart/'losses.csv').open())
    reference={r['objective']:float(r['raw_loss']) for r in oldloss if int(r['index'])==obs['index']}
    for name in ('boundary','transition','permuted','region'):
        np.testing.assert_allclose(float(ctx['base'][name][0].detach()),reference[name],rtol=1e-8,atol=1e-10)
        for s in STAGES:
            old=np.load(C1/'runs'/oldpart/'raw/parameters'/f'L_{obs["index"]}_{name}_{s}.npy')
            value=data['parameter'][s][name];error=norm(value-old);relative=error/max(norm(old),1e-30)
            assert error<=1e-9 or relative<=1e-4
            tables['compatibility'].append(dict(key,reference='C1_'+name,stage=s,relative_error=relative,absolute_error=error))
    for s in STAGES:
        old=archived_aggregate(np.load(b.A_RUN/'raw'/f'L_{s}.npy',mmap_mode='r')[obs['index']],s)
        error=norm(data['parameter'][s]['supervised']-old);relative=error/max(norm(old),1e-30)
        assert error<=1e-9 or relative<=1e-4
        tables['compatibility'].append(dict(key,reference='A_four_head_supervised',stage=s,relative_error=relative,absolute_error=error))
        if full_feature:
            old=np.load(C0/'runs/c0_la10_checked/raw/features'/f'L_{obs["index"]}_{s}.npz')['gradient'][0]
            error=norm(data['feature'][s]['transition']-old);relative=error/max(norm(old),1e-30)
            assert error<=1e-9 or relative<=1e-4
            tables['compatibility'].append(dict(key,reference='C0_feature_transition',stage=s,relative_error=relative,absolute_error=error))


def structural(ctx,data,obs,plan,tables):
    key={k:obs[k] for k in ('case','index','repeat')};p=ctx['pairs']
    z=(ctx['outputs'][-1][0,1].double()-ctx['outputs'][-1][0,0].double()).detach().cpu().numpy()
    zp,zm=z[tuple(p.plus.T)],z[tuple(p.minus.T)];oldcross,validcross=crossing(zp,zm)
    from scipy.special import expit
    for name,(source,lam) in joint_names(plan).items():
        gz=data['logit']['anchor']+lam*data['logit'][source] if lam else data['logit'][source]
        maximum=float(np.max(abs(gz)));changed=z-.05*gz/maximum if maximum>1e-12 else None
        newp=changed[tuple(p.plus.T)] if changed is not None else None;newm=changed[tuple(p.minus.T)] if changed is not None else None
        newcross,newvalid=crossing(newp,newm) if changed is not None else (None,None)
        # Single TRUE-pair derivatives before the mean/coalescing.  Permuted
        # endpoints are different pairs; their actual coalesced gradient is gz.
        da=-.5*expit(-zp);dm=.5*expit(zm);dt=expit(1-zp+zm)
        for fg,bg in ((True,True),(True,False),(False,True),(False,False)):
            keep=((zp>0)==fg)&((zm<=0)==bg);n=int(keep.sum());r=dict(key,objective=name,fg_correct=fg,bg_correct=bg,count=n,total=len(zp),fraction=n/len(zp))
            if n:
                a=.5*(np.logaddexp(0,-zp[keep])+np.logaddexp(0,zm[keep]));t=np.logaddexp(0,1-zp[keep]+zm[keep])
                r.update(anchor_mean=float(a.mean()),true_transition_mean=float(t.mean()),margin_satisfied_wrong_fraction=float(((zp-zm>=1)&((zp<=0)|(zm>0)))[keep].mean()),
                    objective_global_value=float(combine_terms(ctx['base'],plan)[name][0].detach()),
                    single_true_pair_anchor_dplus=float(da[keep].mean()),single_true_pair_anchor_dminus=float(dm[keep].mean()),
                    single_true_pair_transition_dplus=float(-dt[keep].mean()),single_true_pair_transition_dminus=float(dt[keep].mean()),
                    single_true_pair_joint_dplus=float((da-lam*dt)[keep].mean()) if lam and source=='transition' else None,
                    single_true_pair_joint_dminus=float((dm+lam*dt)[keep].mean()) if lam and source=='transition' else None,
                    joint_true_pair_loss_mean=float((a+lam*t).mean()) if lam and source=='transition' else None,
                    baseline_pair_error=float((((zp<=0)|(zm>0))[keep]).mean()),baseline_crossing_fraction=float(validcross[keep].mean()),
                    boundary_gradient_mean_plus=float(gz[tuple(p.plus[keep].T)].mean()),boundary_gradient_mean_minus=float(gz[tuple(p.minus[keep].T)].mean()),
                    fg_correct_descent_fraction=float((gz[tuple(p.plus[keep].T)]<0).mean()),bg_correct_descent_fraction=float((gz[tuple(p.minus[keep].T)]>0).mean()))
                if changed is not None:
                    common=keep&validcross&newvalid
                    r.update(logit_step_status='valid',after_pair_error=float(((newp<=0)|(newm>0))[keep].mean()),after_crossing_fraction=float(newvalid[keep].mean()),
                        crossing_error_before=float(abs(oldcross[common]-.5).mean()) if common.any() else None,crossing_error_after=float(abs(newcross[common]-.5).mean()) if common.any() else None,
                        common_crossings=int(common.sum()))
                else:r['logit_step_status']='below_threshold'
            tables['endpoint_groups'].append(r)


def weights(name,plan):
    if name in FEATURE_BASE:return {name:1.}
    source,lam=joint_names(plan)[name];return {'anchor':1.,source:lam}


def simulate_norm(ctx,data,obs,plan,tables):
    key={k:obs[k] for k in ('case','index','repeat')};ls=float(ctx['base']['supervised'][0].detach());beta=plan['outer_beta']
    terms=combine_terms({k:v for k,v in ctx['base'].items() if k!='supervised'},plan)
    domains=[('all_parameters','theta',data['all'])]+[('parameter',s,gram(data['parameter'][s])) for s in STAGES]
    for name in ('boundary','anchor','true_r10','true_r30','perm_r10','perm_r30','transition'):
        la=float(terms[name][0].detach());den=float(np.float32(np.float32(ls)+np.float32(beta)*np.float32(la)));aux=weights(name,plan)
        for kind,s,g in domains:
            raw_aux=combination_squared(g,aux);raw_sup=g['supervised']['supervised'];sa=sum(v*g['supervised'][a] for a,v in aux.items())
            combined=np.sqrt(max(0,raw_sup+2*beta*sa+beta*beta*raw_aux))/max(den,1e-8)
            tables['lossnorm'].append(dict(key,stage=s,coordinate_kind=kind,objective=name,beta=beta,raw_supervised_loss=ls,raw_aux_loss=la,joint_denominator=den,
                raw_sup_norm=np.sqrt(raw_sup),raw_aux_norm=np.sqrt(raw_aux),aux_to_sup_raw_ratio=beta*np.sqrt(raw_aux/max(raw_sup,1e-30)),
                original_sup_normalized_norm=np.sqrt(raw_sup)/max(ls,1e-8),joint_sup_component_norm=np.sqrt(raw_sup)/max(den,1e-8),
                joint_aux_component_norm=beta*np.sqrt(raw_aux)/max(den,1e-8),joint_normalized_norm=combined,
                independently_normalized_aux_norm=beta*np.sqrt(raw_aux)/max(la,1e-8),independent_aux_to_joint_aux_ratio=max(den,1e-8)/max(la,1e-8),
                norm_policy='original_FP32_joint_vs_separate_risk_reference'))


def direct_checks(ctx,data,mapping,obs,plan,tables):
    names=joint_names(plan);params=[mapping[s].parameter for s in STAGES];nodes=[ctx['features'][s] for s in STAGES]
    terms=combine_terms({k:v for k,v in ctx['base'].items() if k!='supervised'},plan)
    for name,(source,lam) in names.items():
        if not lam:continue
        actual=torch.autograd.grad(terms[name][0],nodes+params,retain_graph=True)
        for i,s in enumerate(STAGES):
            for kind,k,truth in [('feature',i,data['feature'][s]),('parameter',i+5,data['parameter'][s])]:
                value=actual[k].detach().cpu().numpy()[0] if kind=='feature' else mapping[s].channel_vectors(actual[k]).detach().cpu().numpy()
                expected=truth['anchor'].astype(float)+lam*truth[source].astype(float);relative=norm(value-expected)/max(norm(expected),1e-30)
                assert norm(value-expected)<=1e-9 or relative<=1e-4
                tables['reconstruction'].append(dict(index=obs['index'],stage=s,kind=kind,objective=name,relative_error=relative))
    for name in ('boundary','anchor','true_r10','true_r30','perm_r10','perm_r30'):
        joint=ctx['base']['supervised'][0]+plan['outer_beta']*terms[name][0].float()
        grads=torch.autograd.grad(normalize_loss(joint),params,retain_graph=True)
        for s,g in zip(STAGES,grads):
            expected=data['parameter'][s]['supervised'].astype(float)
            for source,value in weights(name,plan).items():expected=expected+plan['outer_beta']*value*data['parameter'][s][source].astype(float)
            expected/=max(float(joint.detach()),1e-8);actual=mapping[s].channel_vectors(g).detach().cpu().numpy();relative=norm(actual-expected)/max(norm(expected),1e-30)
            assert norm(actual-expected)<=1e-9 or relative<=1e-4
            tables['reconstruction'].append(dict(index=obs['index'],stage=s,kind='LossNorm_parameter',objective=name,relative_error=relative))


def intervene(student,x,y,ctx,data,obs,plan,tables):
    key={k:obs[k] for k in ('case','index','repeat')};terms=combine_terms({k:v for k,v in ctx['base'].items() if k!='supervised'},plan)
    base={n:float(v[0].detach()) for n,v in terms.items()};xb=x.expand(4,-1,-1,-1,-1).contiguous();nulls=defaultdict(list)
    with torch.no_grad():
        for s in INTERVENTION_STAGES:
            with attenuate(student,s,[None]*4,ctx['masks'][s],[0.]*4,ctx['features'][s]) as trace:logits=student(xb)
            assert trace['calls']==1 and max(trace['actual_l2'])==0
            for name,value in ctx['targets'](logits,plan).items():nulls[name].extend(value.cpu().tolist())
        eps={}
        for name in base:
            eps[name],record=calibrate_epsilon(base[name],nulls[name]);tables['noise'].append(dict(key,objective=name,epsilon=eps[name],**record))
        for s in INTERVENTION_STAGES:
            channels=plan['channels'][s]
            for dose in DOSES:
                budget=dose*ctx['q'][s];chosen=[dose_alpha(ctx['energies'][s][c],budget) for c in channels];alphas=[a if a is not None else 0. for a,status in chosen]
                with attenuate(student,s,channels+[None],ctx['masks'][s],alphas+[0.],ctx['features'][s]) as trace:logits=student(xb)
                assert trace['calls']==1
                measured=ctx['targets'](logits,plan);metrics=prediction_metrics(logits,y,*ctx['regions'])
                predicted=logits.argmax(1);p=ctx['pairs'];fg=predicted[:,p.plus[:,0],p.plus[:,1],p.plus[:,2]];bg=predicted[:,p.minus[:,0],p.minus[:,1],p.minus[:,2]]
                error=((fg!=1)|(bg!=0)).double().mean(1)
                for row,c in enumerate(channels):
                    alpha,status=chosen[row]
                    if status=='valid' and not np.isclose(trace['actual_l2'][row],budget,rtol=2e-5,atol=1e-5):status='energy_matching_failed'
                    common=dict(key,stage=s,channel=c,dose=dose,status=status,alpha=alpha,target_l2=budget,actual_l2=trace['actual_l2'][row],roundoff_l2=trace['subtraction_roundoff_l2'][row],q=ctx['q'][s])
                    if status=='valid':
                        common.update(boundary_delta=float(measured['boundary'][row]-measured['boundary'][3]),boundary_epsilon=eps['boundary'],
                            anchor_delta=float(measured['anchor'][row]-measured['anchor'][3]),endpoint_error_delta=float(error[row]-error[3]),endpoint_error_baseline=float(error[3]),
                            region_delta=float(measured['region'][row]-measured['region'][3]),changed_prediction_fraction=float((predicted[row]!=predicted[3]).double().mean()),
                            final_logit_l2=float((logits[row]-logits[3]).double().norm()))
                        for name,value in metrics.items():common[name+'_delta']=float(value[row]-value[3])
                    tables['interventions'].append(common)
                    for name,(source,lam) in joint_names(plan).items():
                        record=dict(key,stage=s,channel=c,dose=dose,status=status,objective=name,epsilon=eps[name],boundary_epsilon=eps['boundary'])
                        if status=='valid':
                            stat=data['stats'][s];value=stat['anchor']['inner'][c]+lam*stat[source]['inner'][c] if lam else stat[source]['inner'][c]
                            pred=-alpha*value;actual=float(measured[name][row]-measured[name][3])
                            record.update(delta_hat=pred,delta_actual=actual,boundary_delta=common['boundary_delta'],predicted_direction=direction(pred,eps[name]),
                                          actual_boundary_direction=direction(common['boundary_delta'],eps['boundary']),unit_response=-value/ctx['energies'][s][c])
                        tables['attribution'].append(record)


def historical_mt(out):
    obs=json.loads((b.A_RUN/'observations.json').read_text())['U'];ids=('3C2QTUNI0852XV7ZH4Q1','57SGAJMLCTCH92QUA0EE')
    selected=[o for o in obs if o['case'] in ids];assert len(selected)==4
    rows=[]
    for s in STAGES:
        values=np.load(b.A_RUN/'raw'/f'U_{s}.npy',mmap_mode='r')
        for o in selected:
            raw=archived_aggregate(values[o['index']],s);den=max(o['aggregate_loss'],1e-8)
            rows.append(dict(case=o['case'],index=o['index'],repeat=o['repeat'],stage=s,raw_loss=o['aggregate_loss'],raw_norm=norm(raw),
                normalized_norm=.3*norm(raw)/den,weight=.3,normalization_denominator=den,unreachable_head_status='archived NaN excluded from graph sum, denominator remains 4',source='Stage1A archived eval gradients only; no U data opened'))
    write_csv(out/'historical_MT_reference.csv',rows)


JSON_TABLES={'theta_grams','noise','node_identity','endpoint_groups','compatibility','reconstruction'}
def save(tables,out):
    for name,rows in tables.items():
        (write_json if name in JSON_TABLES else write_csv)(out/(name+('.json' if name in JSON_TABLES else '.csv')),rows)


def run(args,out):
    assert json.loads((ROOT/'TEST_RESULTS.json').read_text())['status']=='PASS'
    if args.phase!='smoke':assert json.loads((ROOT/'runs/smoke/execution_results.json').read_text())['status']=='PASS'
    started=time.monotonic();student,teacher,design=b.load_models(args.device);assert args.device.startswith('cuda')
    mapping=b.stage_parameters(student);snapshots=[b.state_snapshot(m) for m in (student,teacher)]
    states=(torch.get_rng_state().clone(),torch.cuda.get_rng_state(args.device).clone(),np.random.get_state(),random.getstate())
    torch.cuda.reset_peak_memory_stats(args.device)
    if args.phase=='main':
        path=ROOT/'runs/calibration/FROZEN_PLAN.json';plan=json.loads(path.read_text());assert b.sha256(path)==json.loads(path.with_name('FROZEN_PLAN_SHA256.json').read_text())['sha256'];plan_hash=b.sha256(path)
    else:plan=dict(lambdas={'r10':1.,'r30':1.},outer_beta=.01,channels=fixed_channels(),primary='r30');plan_hash=None
    selected=split()['discovery'][:2] if args.phase=='smoke' else split()['discovery'] if args.phase=='calibration' else split()['discovery']+split()['probe']
    tables=defaultdict(list);access=[];file_stats={};observations=json.loads((b.A_RUN/'observations.json').read_text())['L']
    for folder in ('features','parameters'):(out/'raw'/folder).mkdir(parents=True)
    write_json(out/'source_at_run.json',dict(phase=args.phase,sources={p.name:b.sha256(p) for p in ROOT.glob('*.py')},
        design_sha256=b.sha256(ROOT/'C15_DESIGN.md'),plan_sha256=plan_hash,cases=selected,device=args.device,gpu=torch.cuda.get_device_name(args.device),
        torch=torch.__version__,network_precision='FP32',diagnostic_precision='FP64',mode='eval',checkpoint_sha256=design['checkpoint_sha256'],C1_plan_sha256=b.sha256(C1/'runs/discovery/FROZEN_PLAN.json')))
    write_json(out/'parameter_mapping.json',{s:mapping[s].describe() for s in STAGES})
    first=True
    for case in selected:
        _,label,crops=b.load_case_and_crops('L',case,design,[o for o in observations if o['case']==case],access);dist=b_losses.distances(label)
        for r in access:
            p=Path(r['path']);file_stats[str(p)]=(p.stat().st_size,p.stat().st_mtime_ns)
        for obs,x,y,meta in crops:
            x,y=x.to(args.device),y.to(args.device);ctx=base_context(student,x,y,label,dist,meta,obs,args.device)
            tables['node_identity'].extend([dict(index=obs['index'],case=case,**r) for r in ctx['identity']])
            for name,value in ctx['base'].items():tables['losses'].append(dict(case=case,index=obs['index'],repeat=obs['repeat'],objective=name,value=float(value[0].detach())))
            data=extract(student,ctx,obs,mapping,out,tables,save_raw=args.phase!='calibration')
            compatibility(data,ctx,obs,tables,full_feature=args.phase=='smoke')
            if args.phase!='calibration':
                relation_tables(ctx,data,obs,plan,tables);structural(ctx,data,obs,plan,tables);simulate_norm(ctx,data,obs,plan,tables)
                if first:direct_checks(ctx,data,mapping,obs,plan,tables)
                intervene(student,x,y,ctx,data,obs,plan,tables)
            save(tables,out);write_json(out/'hdf5_access_log.json',access)
            print(f'C15 {args.phase} {case} crop{obs["repeat"]}; {time.monotonic()-started:.1f}s',flush=True)
            first=False;del ctx,data,x,y
    if args.phase=='calibration':
        final=calibrate([r['gram'] for r in tables['theta_grams']]);final.update(channels=fixed_channels(),discovery_cases=selected,probe_cases_not_opened_in_C15=True,
            C1_plan_sha256=b.sha256(C1/'runs/discovery/FROZEN_PLAN.json'),calibration_design_sha256=b.sha256(ROOT/'C15_DESIGN.md'))
        # Equal weight to cases, then to each case's two crops. No probe data.
        scores={}
        for s in INTERVENTION_STAGES:
            scores[s]={}
            for c in final['channels'][s]:
                base_scores={}
                for name in FEATURE_BASE:
                    values=[]
                    for case in selected:
                        rows=[r for r in tables['contributions'] if r['stage']==s and r['channel']==c and r['objective']==name and r['case']==case]
                        assert len(rows)==2 and all(r['status']=='valid' and r['unit_response'] is not None for r in rows)
                        values.append(float(np.mean([r['unit_response'] for r in rows])))
                    base_scores[name]=float(np.mean(values))
                for name,(source,lam) in joint_names(final).items():
                    scores[s].setdefault(name,{})[str(c)]=base_scores['anchor']+lam*base_scores[source] if lam else base_scores[source]
        final['transfer_scores']=scores
        final['transfer_score_definition']='equal-case discovery mean signed response per unit L2; no probe gradients or ranking'
        write_json(out/'FROZEN_PLAN.json',final);write_json(out/'FROZEN_PLAN_SHA256.json',dict(sha256=b.sha256(out/'FROZEN_PLAN.json')))
    historical_mt(out)
    invariants=dict(student_state_unchanged=b.unchanged(student,snapshots[0]),teacher_state_unchanged=b.unchanged(teacher,snapshots[1]),teacher_frozen=all(not p.requires_grad for p in teacher.parameters()),
        all_grad_slots_none=all(p.grad is None for m in (student,teacher) for p in m.parameters()),both_eval=not student.training and not teacher.training,
        torch_rng_unchanged=torch.equal(states[0],torch.get_rng_state()),cuda_rng_unchanged=torch.equal(states[1],torch.cuda.get_rng_state(args.device)),
        numpy_rng_unchanged=all(np.array_equal(a,c) for a,c in zip(states[2],np.random.get_state())),python_rng_unchanged=states[3]==random.getstate(),
        no_leftover_hooks=all(not m._forward_hooks and not m._forward_pre_hooks for m in student.modules()),only_L_data_opened=all(a['role']=='labeled' for a in access),
        data_stats_unchanged=all((Path(p).stat().st_size,Path(p).stat().st_mtime_ns)==v for p,v in file_stats.items()),
        frozen_plan_unchanged=args.phase!='main' or b.sha256(ROOT/'runs/calibration/FROZEN_PLAN.json')==plan_hash)
    assert all(invariants.values()),invariants
    result=dict(status='PASS',phase=args.phase,cases=len(selected),crops=2*len(selected),interventions=len(tables['interventions']),invariants=invariants,
        elapsed_seconds=time.monotonic()-started,peak_allocated_mib=torch.cuda.max_memory_allocated(args.device)/2**20,optimizer_steps=0,checkpoint_writes=0,formal_training=False,C2_C3_executed=False)
    write_json(out/'execution_results.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('smoke','calibration','main'),required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:1')
    args=p.parse_args();out=output_path(args.output);out.mkdir(parents=True,exist_ok=False)
    try:print(json.dumps(run(args,out),indent=2))
    except BaseException:
        write_json(out/'execution_results.json',dict(status='FAIL',phase=args.phase,traceback=traceback.format_exc()));raise
