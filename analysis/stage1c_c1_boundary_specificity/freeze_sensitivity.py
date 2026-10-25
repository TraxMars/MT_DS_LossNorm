"""Freeze predeclared margin sensitivity slopes before any probe access."""
import json
from datetime import datetime,timezone
import pandas as pd
from bridge import ROOT,STAGES,PRIMARY,b,write_json

def main():
    run=ROOT/'runs/discovery';assert not (ROOT/'runs/probe').exists()
    assert json.loads((run/'execution_results.json').read_text())['status']=='PASS'
    plan=json.loads((run/'FROZEN_PLAN.json').read_text());d=pd.read_csv(run/'contributions.csv')
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),probe_directory_absent=True,
                primary_plan_sha256=b.sha256(run/'FROZEN_PLAN.json'),scores={})
    for s in STAGES:
        result['scores'][s]={r:{n:{} for n in ('transition_m0_5','transition_m2')} for r in ('B','I')}
        for c in plan['stages'][s]['eligible']:
            block=d[(d.stage==s)&(d.channel==c)]
            primary=block[block.objective.isin(PRIMARY)]
            valid=primary.groupby('index').apply(lambda v:bool(((v.gradient_status=='valid')&(v.max_dose_status=='valid')&v.unit_response.notna()).all()),include_groups=False)
            indices=valid[valid].index
            for r in ('B','I'):
                for n in ('transition_m0_5','transition_m2'):
                    rows=block[(block.support==r)&(block.objective==n)&block['index'].isin(indices)]
                    assert rows.unit_response.notna().all()
                    result['scores'][s][r][n][str(c)]=float(rows.groupby('case').unit_response.mean().mean())
    path=run/'FROZEN_SENSITIVITY.json';assert not path.exists();write_json(path,result)
    write_json(ROOT/'PROBE_PRELAUNCH.json',dict(created_utc=result['created_utc'],primary_plan_sha256=result['primary_plan_sha256'],
                                            sensitivity_sha256=b.sha256(path),probe_directory_absent=True))
    print('Margin 0.5/2 sensitivity scores frozen; original candidate plan unchanged')

if __name__=='__main__':main()
