"""ALL 2023 races, frozen trained URA + new textbook whole-set policy.

Retrospective FINAL-ODDS proxy only. No fabricated pre-start timestamps, no
production-input relaxation, no retraining, no legacy shape imports.
The base ranker was trained on 2023-24; URA coefficients were learned later.
This run is NOT out-of-sample or a chronological 2023 trading experiment.
"""
import argparse
from collections import Counter,defaultdict
from datetime import date,timedelta
import gzip,hashlib,itertools,json,math,re,sqlite3
from pathlib import Path
import numpy as np
from ura.core import prepare,ordered_probabilities
from ura.manual_formations import (POLICY_ID,FIXED_MODEL_SHA,FIXED_URA_BLOB,
    load_probability_model,parse_lines,parse_book,build_formations,
    compare_priced_formations,public_shape)


def netid(source):
    if not re.fullmatch(r'\d{16}',source):raise ValueError('Unexpected archive ID')
    d=date.fromisoformat(source[2:6]+'-'+source[6:8]+'-'+source[8:10])+timedelta(days=int(source[10:12])-1)
    return d.strftime('%Y%m%d')+source[:2]+f'{int(source[12:]):02d}'


def forecast(row,book,model):
    """A final-price research decision, deliberately distinct from live evaluate_race."""
    riders,x,gap,underdog=prepare([{'bike_number':r['bike'],'score':r['score']} for r in row['ranked']])
    if abs(gap-float(row['score_gap']))>.002:raise ValueError('Rank/gap mismatch')
    out={'policy_id':POLICY_ID,'decision_context':'RETROSPECTIVE_FINAL_ODDS_PROXY',
         'counts_as_live_purchase':False,'automatic_purchase':False,'decision':'NO_BET',
         'tickets':[],'points':0,'stake_yen':0,'source_gap':gap,'selected':None}
    if gap>=1:return {**out,'reason':'NOT_LOW_CONFIDENCE'}
    active=[int(r['bike_number']) for r in riders]
    lines=parse_lines(row['line_formation'],active)
    lower=[active[i] for i in range(len(active)) if underdog[i]]
    shapes=build_formations(active,lines,lower)
    odds=parse_book(book,active)
    allp=ordered_probabilities(x,np.asarray(model['parameters'],float))
    probs={tuple(active[i] for i in t):v for t,v in allp.items()}
    evaluated,qualified,best=compare_priced_formations(shapes,probs,odds)
    out.update(lower_rank_bikes=lower,candidate_count=len(shapes),price_qualified_count=len(qualified))
    if best is None:return {**out,'reason':'NO_WHOLE_FORMATION_PRICE_EDGE'}
    metrics,shape=best
    selected=public_shape(shape,probs,odds)
    return {**out,'decision':'BET','reason':'WHOLE_FORMATION_PRICE_EDGE',
            'selected':selected,'tickets':selected['tickets'],'points':metrics['points'],
            'stake_yen':metrics['investment_yen']}


def actual_trifecta(rows):
    pos={i:[r['bike_number'] for r in rows if r['finish_position']==i] for i in (1,2,3)}
    return tuple(pos[i][0] for i in (1,2,3)) if all(len(x)==1 for x in pos.values()) else None


def settle(decision,results,pay,book):
    if decision['decision']!='BET':return {'status':'no_purchase','investment_yen':0,'returned_yen':0}
    actual=actual_trifecta(results)
    combo=tuple(map(int,re.findall(r'\d+',str(pay.get('sanrentan_combo','')))))
    dividend=pay.get('sanrentan_payout')
    if actual is None or combo!=actual or dividend is None or not math.isfinite(float(dividend)) or float(dividend)<100:
        return {'status':'unverified_outcome_or_payout','investment_yen':None,'returned_yen':None}
    win='-'.join(map(str,actual))
    if win not in book or abs(float(book[win])*100-float(dividend))>.01:
        return {'status':'final_quote_payout_mismatch','investment_yen':None,'returned_yen':None,
                'actual':win,'payout_per_100_yen':dividend,'quoted_final_odds':book.get(win)}
    hit=win in decision['tickets'];stake=decision['stake_yen'];paid=int(dividend) if hit else 0
    return {'status':'settled','actual':win,'payout_per_100_yen':dividend,
            'hit':hit,'investment_yen':stake,'returned_yen':paid,'net_yen':paid-stake}


def main():
    p=argparse.ArgumentParser();p.add_argument('--month',required=True)
    p.add_argument('--forecasts',required=True,type=Path);p.add_argument('--odds',required=True,type=Path)
    p.add_argument('--db',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();assert re.fullmatch(r'2023-(0[1-9]|1[012])',a.month)
    a.out.mkdir(parents=True,exist_ok=True)
    model=load_probability_model()
    with gzip.open(a.forecasts,'rt') as f:source={r['race_id']:r for r in map(json.loads,f)}
    assert len(source)==26838,'Pinned source forecast universe changed'
    con=sqlite3.connect('file:'+str(a.db)+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
    meta={r['race_id']:dict(r) for r in con.execute('SELECT * FROM races WHERE date LIKE ?',(a.month+'%',))}
    quotes=defaultdict(list)
    with gzip.open(a.odds,'rt') as f:
        for obj in map(json.loads,f):
            rid=netid(str(obj['race_id']))
            if rid[:6]==a.month.replace('-',''):quotes[rid].append(obj)
    ids=sorted(set(meta)|set(quotes)|{rid for rid in source if rid[:6]==a.month.replace('-','')})
    audit={'policy_id':POLICY_ID,'month':a.month,'scope':'ALL_RACES_UNION_DB_FORECAST_AND_ODDS_CALENDAR',
        'source_ranker_sha256':FIXED_MODEL_SHA,'probability_model_git_blob':FIXED_URA_BLOB,
        'legacy_formations_used':False,'source_ranker_2023_in_sample':True,
        'probability_coefficients_trained_after_2023':True,
        'new_training':False,'decision_context':'RETROSPECTIVE_FINAL_ODDS_PROXY',
        'production_prestart_checks_relaxed':False,
        'conditions':'LOW gap<1; whole shape includes lower-rated rider; composite>=3.0; modeled ROI>1; maximize modeled net profit; 100 yen uniform',
        'input_sha256':{k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in [('forecasts',a.forecasts),('odds',a.odds)]}}
    (a.out/'policy.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    status=Counter();totals=Counter();families=defaultdict(Counter)
    f=(a.out/'races.jsonl').open('w')
    def record(row):
        status[row['status']]+=1
        d=row.get('decision',{});s=row.get('settlement',{})
        if d.get('decision')=='BET':totals['selected_races']+=1
        if s.get('status')=='settled':
            fam=families[d['selected']['family']]
            for c in (totals,fam):
                c['settled_races']+=1;c['hits']+=int(s['hit']);c['points']+=d['points']
                c['investment_yen']+=s['investment_yen'];c['returned_yen']+=s['returned_yen']
        elif d.get('decision')=='BET':totals['unresolved_selected_races']+=1
        f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
    print('URA_2023_MONTH_START',a.month,len(ids),flush=True)
    for i,rid in enumerate(ids):
        row={'race_id':rid,'date':f'{rid[:4]}-{rid[4:6]}-{rid[6:8]}','status':'pending'}
        m=meta.get(rid);old=source.get(rid)
        if m is None:record({**row,'status':'missing_race_metadata'});continue
        entries=[dict(r) for r in con.execute('SELECT bike_number,class FROM entries WHERE race_id=?',(rid,))]
        classes=[str(e['class'] or '') for e in entries]
        if any(c.upper().startswith(('L','Ｌ')) for c in classes) or 'ガール' in str(m.get('race_name','')):
            record({**row,'status':'excluded_girls'});continue
        if not old or old['status']!='processed':
            record({**row,'status':'missing_compatible_frozen_score_snapshot','source_status':old.get('status') if old else None});continue
        if len(entries)!=m['rider_count'] or set(e['bike_number'] for e in entries)!=set(r['bike'] for r in old['ranked']):
            record({**row,'status':'entry_score_roster_mismatch'});continue
        if not classes or any(not c or c in ('0','0.0') for c in classes):
            record({**row,'status':'unverified_race_class'});continue
        if m['grade'] not in {'F1','F2','G1','G2','G3','GP'}:
            record({**row,'status':'unknown_grade'});continue
        # LOW gate precedes odds requirement, matching current live policy.
        if float(old['ranked'][0]['score'])-float(old['ranked'][1]['score'])>=1:
            record({**row,'status':'not_low_confidence'});continue
        if len(quotes.get(rid,[]))!=1:
            record({**row,'status':'missing_or_duplicate_odds_record'});continue
        book=quotes[rid][0].get('odds',{}).get('3連単',{})
        try:d=forecast(old,book,model)
        except ValueError as exc:
            record({**row,'status':'invalid_formation_or_price_input','reason':str(exc)});continue
        # Outcomes queried only after the complete formation decision is fixed.
        results=[dict(r) for r in con.execute('SELECT bike_number,finish_position FROM race_results WHERE race_id=?',(rid,))]
        payrow=con.execute('SELECT * FROM race_payouts WHERE race_id=?',(rid,)).fetchone()
        settlement=settle(d,results,dict(payrow) if payrow else {},book)
        record({**row,'status':'processed','grade':m['grade'],'ranked':old['ranked'],
                'line_formation':old['line_formation'],'decision':d,'settlement':settlement})
        if i%100==0:print('URA_MONTH_PROGRESS',a.month,i+1,len(ids),dict(status),flush=True)
    f.close();con.close()
    report={'month':a.month,'universe':len(ids),'recorded':sum(status.values()),'statuses':dict(status),
        'totals':dict(totals),'by_family':{k:dict(v) for k,v in families.items()},
        'all_records_accounted':sum(status.values())==len(ids),'simulation_context':'RETROSPECTIVE_FINAL_ODDS_PROXY'}
    assert report['all_records_accounted']
    (a.out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print('URA_2023_MONTH_DONE',a.month,'recorded',report['recorded'],flush=True)


if __name__=='__main__':main()
