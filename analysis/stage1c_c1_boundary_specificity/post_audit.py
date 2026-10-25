"""Verify the low-signal reporting fix without changing raw or primary data."""
import json
from datetime import datetime,timezone
import pandas as pd
from bridge import ROOT,b,write_json

def main():
    reference=pd.read_csv(ROOT/'TRANSFER_REFERENCE_BEFORE_VALIDITY_AUDIT.csv')
    current=pd.read_csv(ROOT/'attribution_by_case.csv');transfer=current[current['mode']!='local_GT_gradient']
    keys=['split','case','stage','support','dose','objective','mode']
    old=reference.sort_values(keys).reset_index(drop=True);new=transfer[list(reference.columns)].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(old,new,check_exact=False,rtol=1e-12,atol=1e-14)
    invalid=current[(current['mode']=='local_GT_gradient')&(current.objective=='far')&(current.source_reliable_channels==0)]
    assert invalid.signed_spearman.isna().all() and invalid.direction_accuracy.isna().all()
    badcases=sorted(invalid[invalid.split=='probe'].case.unique());assert len(badcases)==3
    assert current[(current['mode']=='discovery_B_transfer')&(current.support=='B')&(current.stage=='D1')&(current.objective=='transition')].direction_accuracy.eq(0).all()
    tests=json.loads((ROOT/'TEST_RESULTS.json').read_text());assert tests['status']=='PASS' and tests['unittest_total']==70
    raw=json.loads((ROOT/'VERIFICATION_RESULTS.json').read_text());assert raw['status']=='PASS'
    history=b.verify_protection(json.loads((ROOT/'PROTECTION_BEFORE.json').read_text()));assert history['status']=='PASS'
    result=dict(status='PASS',created_utc=datetime.now(timezone.utc).isoformat(),unchanged_transfer_records=len(old),
                primary_transfer_statistics_unchanged=True,unreliable_local_far_cases=badcases,undefined_local_far_summary_rows=len(invalid),
                final_unit_tests=tests['unittest_total'],history=history,raw_replay_report_sha256=b.sha256(ROOT/'VERIFICATION_RESULTS.json'))
    write_json(ROOT/'POST_AUDIT_VERIFICATION.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
