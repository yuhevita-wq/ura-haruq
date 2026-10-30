"""URA manual formation policy: scenario -> whole set -> price -> decision.

No V1/V2/V3 formation catalog, rank-bin rule, six-point restriction, 2221 gate,
or 7.1-point spread filter is used. Frozen learned probability model is retained.
Manual source: libfile_30aa4d12f5e08191990a4239a0fd7780, pp9-21,29-35.
Numeric ROI/composite thresholds are policy choices, not claims in the manual.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib,itertools,json,math,re
from pathlib import Path
import numpy as np
from ura.core import prepare,ordered_probabilities,low_rank_start

POLICY_ID='URA_MANUAL_FORMATIONS_V1_20261030'
FIXED_MODEL_SHA='e70690f4538b77e927a042e7ae2db312f8a0a4f16bf72def37e7ccad3568c193'
FIXED_URA_BLOB='5a353643cfe8196ab695521d479237294c3f9b66'
DEFAULT_MODEL=Path(__file__).resolve().parents[1]/'artifacts/ura_ai_v1.json'
MANUAL='libfile_30aa4d12f5e08191990a4239a0fd7780'
STAKE=100


def load_probability_model(path=DEFAULT_MODEL):
    raw=Path(path).read_bytes()
    blob=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
    if blob!=FIXED_URA_BLOB:raise ValueError('Frozen trained URA probability model mismatch')
    model=json.loads(raw)
    if not model.get('training_completed') or not model.get('trained_on_real_results') or model.get('source_frozen_model_sha256')!=FIXED_MODEL_SHA:
        raise ValueError('Trained probability model provenance mismatch')
    return model


def expand(fields,active):
    if len(fields)!=3:raise ValueError('Three finish-position fields required')
    fields=tuple(tuple(sorted(set(x))) for x in fields)
    if any(not f or not set(f)<=set(active) for f in fields):raise ValueError('Empty, absent or unknown rider in formation')
    return frozenset(t for t in itertools.product(*fields) if len(set(t))==3)


def parse_lines(value,active):
    if isinstance(value,str):
        if not re.fullmatch(r'[1-9](?:-[1-9])*(?:/[1-9](?:-[1-9])*)*',value):raise ValueError('Line order must be explicit, e.g. 1-2-3/4-5/6-7')
        lines=[list(map(int,x.split('-'))) for x in value.split('/')]
    else:lines=[[int(x) for x in group] for group in value]
    flat=[b for group in lines for b in group]
    if len(flat)!=len(set(flat)) or set(flat)!=set(active) or any(not x for x in lines):
        raise ValueError('Complete ordered lineup must match every active rider exactly once')
    return lines


def build_formations(active,lines,lower):
    """Generate whole structural scenarios; never cut individual negative-EV tickets.

    Lower-rated rider is an anchor in first/second/third, or one of a fixed pair.
    Candidate support pools follow complete line relationships, not predicted
    rank=finishing-order. All members are potential relations, not proven flow.
    """
    candidates=[];seen=set();lower=set(lower)
    roles={b:{'line':i+1,'position':j+1} for i,line in enumerate(lines) for j,b in enumerate(line)}
    def add(blocks,family,context):
        blocks=[b for b in blocks if b['expanded']]
        tickets=frozenset(t for b in blocks for t in b['expanded'])
        if len(tickets)<2 or tickets in seen:return
        # Reject the whole shape if it violates URA's lower-rated-rider premise.
        # Pruning individual tickets would change the taught shape.
        if not all(set(t)&lower for t in tickets):return
        seen.add(tickets)
        candidates.append({'family':family,'context':context,'blocks':blocks,
            'tickets':tickets,'points':len(tickets)})
    def block(fields,why,pages):
        normalized=[sorted(set(x)) for x in fields]
        return {'fields':normalized,'notation':'-'.join(''.join(map(str,x)) for x in normalized),
                'reason':why,'manual_pages':pages,'expanded':expand(normalized,active)}
    heads=[line[0] for line in lines]
    for u in sorted(lower):
        own=next(line for line in lines if u in line)
        for li,line in enumerate(lines):
            core=[b for b in line if b!=u]
            if not core:continue
            pools=[core,sorted(set(core+own)-{u}),sorted(set(core+own+heads)-{u})]
            for pool in pools:
                if not pool:continue
                context={'lower_anchor':u,'anchor_role':roles[u],'support_line':li+1,
                         'support_line_order':line,'uncertainty_pool':pool}
                first=block([[u],pool,pool],'低評価選手が勝つ仮説。相手の2・3着順を残す',[9,10])
                second=block([core,[u],pool],'低評価選手が2着へ入る仮説。前後の候補を分ける',[16])
                third=block([core,pool,[u]],'低評価選手が3着へ残る仮説。上位は連携・異線候補から組む',[17,19])
                for b,f in [(first,'first_fixed'),(second,'second_fixed'),(third,'third_fixed')]:add([b],f,context)
                # Keep scenario blocks distinct; do not merge their candidate columns.
                for aa,bb in itertools.combinations([first,second,third],2):add([aa,bb],'multiple_scenarios',context)
                add([first,second,third],'multiple_scenarios',context)
                for v in core:
                    pair=block([[u,v],[u,v],pool],'低評価選手と連携相手・異線相手のワンツー表裏',[11,14,20])
                    sandwich=block([[v],pool,[u]],'相手が勝ち、低評価選手は3着へ残す',[12,15])
                    add([pair],'pair_reverse',context)
                    add([sandwich],'sandwich',context)
                    add([block([[u],pool,[v]],'低評価選手が勝ち、相手を3着へ置く',[15])],'sandwich',context)
                    add([block([[u],[v],pool],'低評価選手が勝つ順序まで限定する',[13])],'ordered_pair',context)
                    add([block([[v],[u],pool],'連携・異線相手が勝ち、低評価選手が2着',[13])],'ordered_pair',context)
                    add([pair,sandwich],'pair_plus_sandwich',context)
                for v,w in itertools.combinations(core,2):
                    add([block([[u,v,w]]*3,'低評価選手を含む3人の着順を残す',[21])],'three_rider_box',context)
    return candidates


def evaluate_set(tickets,probabilities,odds):
    n=len(tickets)
    if not n:raise ValueError('Empty purchase set')
    mass=math.fsum(probabilities[t] for t in tickets)
    stake=n*STAKE
    expected=STAKE*math.fsum(probabilities[t]*odds[t] for t in tickets)
    return {'probability_mass':mass,'points':n,'investment_yen':stake,
        'estimated_return_yen':expected,'estimated_net_yen':expected-stake,
        'estimated_roi':expected/stake,
        'composite_odds':1/math.fsum(1/odds[t] for t in tickets),
        'minimum_hit_return_ratio':min(odds[t] for t in tickets)/n}


def public_shape(shape,probabilities,odds=None):
    tickets=shape['tickets']
    result={k:v for k,v in shape.items() if k not in ('tickets','blocks','metrics')}
    result['blocks']=[{k:v for k,v in b.items() if k!='expanded'} for b in shape['blocks']]
    result['tickets']=['-'.join(map(str,t)) for t in sorted(tickets)]
    result['probability_mass']=math.fsum(probabilities[t] for t in tickets) if probabilities is not None else None
    if odds is not None:result['metrics']=evaluate_set(tickets,probabilities,odds)
    result['block_contributions']=[]
    for i,b in enumerate(shape['blocks']):
        other=frozenset(t for j,x in enumerate(shape['blocks']) if j!=i for t in x['expanded'])
        extra=b['expanded']-other
        result['block_contributions'].append({'notation':b['notation'],'unique_additional_points':len(extra),
            'additional_cost_yen':len(extra)*STAKE,
            'additional_probability_mass':math.fsum(probabilities[t] for t in extra) if probabilities is not None else None,
            'additional_estimated_net_yen':STAKE*math.fsum(probabilities[t]*odds[t]-1 for t in extra) if odds is not None else None})
    return result


def compare_priced_formations(shapes,probabilities,odds):
    """Shared live/replay selection. No inputs here come from realized results."""
    evaluated=[(evaluate_set(c['tickets'],probabilities,odds),c) for c in shapes]
    qualifying=[(m,c) for m,c in evaluated if m['composite_odds']>=3.0 and m['estimated_roi']>1.0]
    selected=max(qualifying,key=lambda v:(v[0]['estimated_net_yen'],v[0]['probability_mass'],-v[0]['points'])) if qualifying else None
    return evaluated,qualifying,selected


def parse_book(raw,active):
    expected=set(itertools.permutations(active,3));book={}
    for key,value in raw.items():
        if not re.fullmatch(r'[1-9]-[1-9]-[1-9]',str(key)):raise ValueError('Odds keys must be exact ordered triples, e.g. 1-2-3')
        t=tuple(map(int,key.split('-')));v=float(value)
        if t not in expected or t in book or not math.isfinite(v) or v<=1:raise ValueError('Invalid or duplicated odds')
        book[t]=v
    if set(book)!=expected:raise ValueError('All ordered-triple odds required; partial book cannot select a formation')
    return book


def timestamp(value):
    d=datetime.fromisoformat(str(value).replace('Z','+00:00'))
    if d.tzinfo is None or d.utcoffset() is None:raise ValueError('Timezone-aware timestamps required')
    return d


def evaluate_race(source,model=None):
    if not isinstance(source,dict) or not source.get('race_id'):raise ValueError('race_id required')
    out={'race_id':str(source['race_id']),'policy_id':POLICY_ID,'manual_source':MANUAL,
        'decision':'NO_BET','reason':None,'tickets':[],'points':0,'stake_yen':0,
        'yen_per_ticket':STAKE,'bet_type':'3連単','selected':None,
        'legacy_formations_used':False,'automatic_purchase':False,
        'validation_status':'UNVALIDATED_NEW_FORMATION_POLICY',
        'selection_basis':'whole_formation_probability_price_and_cost'}
    if source.get('grade') not in {'F1','F2','G1','G2','G3','GP'}:
        return {**out,'reason':'MISSING_OR_UNSUPPORTED_GRADE'}
    if not source.get('race_class') or any(s in str(source['race_class']) for s in ('ガール','L級','Ｌ級','L1')):
        return {**out,'reason':'GIRLS_OR_UNVERIFIED_RACE_CLASS'}
    if 'ranked' not in source:
        return {**out,'reason':'FULL_SCORE_SNAPSHOT_REQUIRED_NO_LEGACY_RANK_FALLBACK'}
    riders,x,gap,underdog=prepare([{'bike_number':int(r['bike']),'score':float(r['score'])} for r in source['ranked']])
    active=[int(r['bike_number']) for r in riders]
    if any(b<1 or b>9 for b in active):raise ValueError('Bike numbers must be 1-9')
    if not math.isfinite(float(source.get('confidence',float('nan')))) or abs(gap-float(source['confidence']))>.002:raise ValueError('Saved scores and confidence mismatch')
    if gap>=1:return {**out,'reason':'NOT_LOW_CONFIDENCE','confidence_gap':gap}
    lines=parse_lines(source.get('formation',source.get('line_formation',[])),active)
    lower=[active[i] for i in range(len(active)) if underdog[i]]
    shapes=build_formations(active,lines,lower)
    probability_source='unchanged_trained_URA_ordered_probabilities'
    if source.get('model_sha256')==FIXED_MODEL_SHA:
        model=load_probability_model() if model is None else model
        if model != load_probability_model():raise ValueError('Trained URA parameters differ from pinned artifact')
        p=ordered_probabilities(x,np.asarray(model['parameters'],float))
        probs={tuple(active[i] for i in triple):v for triple,v in p.items()}
    elif source.get('model_role')=='INDEPENDENT_CHRONOLOGICAL_2026':
        if not source.get('model_sha256'):raise ValueError('Independent model fingerprint required')
        probability_source='independent_compatible_joint_snapshot'
        if not source.get('trifecta_probabilities'):
            return {**out,'confidence_gap':gap,'lower_rank_bikes':lower,
                'candidate_count':len(shapes),'reason':'MISSING_COMPATIBLE_INDEPENDENT_PROBABILITIES',
                'candidates':[public_shape(c,None) for c in shapes]}
        provenance=source.get('probability_provenance',{})
        rank_hash=hashlib.sha256(json.dumps([{'bike':int(r['bike_number']),'score':float(r['score'])} for r in riders],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if (provenance.get('race_id')!=str(source['race_id'])
                or provenance.get('ranked_scores_sha256')!=rank_hash
                or provenance.get('source_model_sha256')!=source['model_sha256']
                or not provenance.get('probability_model_id')
                or provenance.get('trained_on_results') is not True
                or provenance.get('odds_used_as_teacher') is not False):
            raise ValueError('Independent probability provenance mismatch; never port original-model coefficients')
        probs={tuple(map(int,k.split('-'))):float(v) for k,v in source['trifecta_probabilities'].items()}
        if (set(probs)!=set(itertools.permutations(active,3)) or
                any(not math.isfinite(v) or v<0 or v>1 for v in probs.values()) or
                abs(math.fsum(probs.values())-1)>1e-8):
            raise ValueError('Complete normalized independent joint probability distribution required')
    else:
        raise ValueError('Unknown probability input model; no cross-model fallback')
    out.update(confidence_gap=gap,lower_rank_bikes=lower,line_formation=lines,
        candidate_count=len(shapes),probability_model_git_blob=FIXED_URA_BLOB if source['model_sha256']==FIXED_MODEL_SHA else None,
        probability_method=probability_source,
        probability_limit='New formation-level probabilities and ROI are not forward validated; model lacks line interaction')
    if not source.get('trifecta_odds'):
        return {**out,'reason':'MISSING_ODDS','candidates':[public_shape(c,probs) for c in shapes]}
    odds=parse_book(source['trifecta_odds'],active)
    if not all(source.get(k) for k in ('start_at','prediction_at','lineup_at','odds_observed_at')):
        return {**out,'reason':'MISSING_PRESTART_TIMESTAMPS'}
    start=timestamp(source['start_at'])
    if any(timestamp(source[k])>=start for k in ('prediction_at','lineup_at','odds_observed_at')):
        return {**out,'reason':'INPUT_NOT_CAPTURED_BEFORE_START'}
    if source.get('odds_kind')!='pre_start_snapshot':
        return {**out,'reason':'FINAL_OR_UNVERIFIED_ODDS_NOT_PURCHASE_TIME_PRICES'}
    # No individual p*odds filter: evaluate each complete shape/union unchanged.
    evaluated,qualifying,best=compare_priced_formations(shapes,probs,odds)
    audit_keys=('race_id','grade','race_class','ranked','model_sha256','model_role','confidence','formation','line_formation','trifecta_odds','trifecta_probabilities','probability_provenance','start_at','prediction_at','lineup_at','odds_observed_at','odds_kind')
    out['decision_inputs_sha256']=hashlib.sha256(json.dumps({k:source.get(k) for k in audit_keys},sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    out['price_qualified_count']=len(qualifying)
    out['candidates']=[public_shape(c,probs,odds) for m,c in sorted(evaluated,key=lambda v:(-v[0]['estimated_net_yen'],v[0]['points']))]
    if not qualifying:return {**out,'reason':'NO_WHOLE_FORMATION_PRICE_EDGE'}
    m,chosen=best
    selected=public_shape(chosen,probs,odds)
    return {**out,'decision':'BET','reason':'WHOLE_FORMATION_PRICE_EDGE',
        'selected':selected,'tickets':selected['tickets'],'points':m['points'],
        'stake_yen':m['investment_yen'],'odds_observed_at':source['odds_observed_at'],
        'decision_note':'Model-based selection only; profitability has not been validated on unknown periods'}


def main():
    p=argparse.ArgumentParser(description='URA manual formations: replace all legacy purchase shapes')
    p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path)
    args=p.parse_args();payload=json.loads(args.input.read_text())
    if not isinstance(payload,dict) or not isinstance(payload.get('races'),list):raise ValueError('Expected {"races":[...]}')
    model=load_probability_model()
    result={'policy_id':POLICY_ID,'legacy_formations_used':False,
            'races':[evaluate_race(row,model) for row in payload['races']]}
    rendered=json.dumps(result,ensure_ascii=False,indent=2)+'\n'
    if args.out:args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(rendered)
    print(rendered,end='')


if __name__=='__main__':main()
