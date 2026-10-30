"""Require all 12 months; keep missing races visible in full-year output."""
from collections import Counter,defaultdict
import gzip,json
from pathlib import Path


def main():
    root=Path('annual_results');root.mkdir(exist_ok=True)
    months=[json.loads(p.read_text()) for p in sorted(Path('monthly_results').glob('**/summary.json'))]
    assert len(months)==12 and {m['month'] for m in months}=={f'2023-{i:02d}' for i in range(1,13)}
    statuses=Counter();totals=Counter();families=defaultdict(Counter)
    for m in months:
        assert m['all_records_accounted'];statuses.update(m['statuses']);totals.update(m['totals'])
        for k,v in m['by_family'].items():families[k].update(v)
    def enrich(c):
        d=dict(c);s=c['investment_yen'];n=c['settled_races']
        return {**d,'net_yen':c['returned_yen']-s,'roi_percent':100*c['returned_yen']/s if s else None,
            'hit_rate_percent':100*c['hits']/n if n else None,'average_points':c['points']/n if n else None}
    missing=sum(v for k,v in statuses.items() if k not in ('processed','not_low_confidence','excluded_girls'))
    summary={'period':['2023-01-01','2023-12-31'],'policy_id':'URA_MANUAL_FORMATIONS_V1_20261030',
        'months':months,'universe':sum(m['universe'] for m in months),
        'recorded':sum(m['recorded'] for m in months),'statuses':dict(statuses),
        'totals':enrich(totals),'by_family':{k:enrich(v) for k,v in families.items()},
        'unresolved_input_races':missing,'data_complete':missing==0 and totals['unresolved_selected_races']==0,
        'decision_context':'RETROSPECTIVE_FINAL_ODDS_PROXY',
        'legacy_formations_used':False,'base_ranker_2023_in_sample':True,
        'probability_coefficients_trained_after_2023':True,
        'not_unknown_period_test':True,'not_purchase_time_odds_test':True}
    (root/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    ids=set()
    with gzip.open(root/'all_races.jsonl.gz','wt') as out:
        for p in sorted(Path('monthly_results').glob('**/races.jsonl.gz')):
            with gzip.open(p,'rt') as f:
                for line in f:
                    row=json.loads(line);assert row['race_id'] not in ids
                    ids.add(row['race_id']);out.write(line)
    assert len(ids)==summary['universe']==summary['recorded']
    print('URA_FULL_2023_AGGREGATED',len(ids),'records; results saved')


if __name__=='__main__':main()
