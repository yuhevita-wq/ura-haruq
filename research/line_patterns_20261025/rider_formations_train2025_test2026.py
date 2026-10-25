"""Research-only 2025 rider-informed trifecta formation / 2026 forward backtest.

Two strictly separated commands:
  fit   : Jan-Sep 2025 learn rider + rank + line-position tendencies,
          Oct-Dec 2025 select rule by *actual* official trifecta ROI,
          refit on full 2025 and SHA-seal the policy.
  score : Read sealed 2025 policy, score 2026 months without re-fitting,
          write full per-race purchase/return evidence.

Everyone is eligible: non-girls S, A1/A2, A3; n=5..9 supported by
original frozen HaruQ. Rank5-7 exposure is used as *information*, NOT a
qualification rule. Neither line patterns nor rider appearance counts
restrict candidate tickets. All three finishing positions are optimized.

Final odds are not used here (availability/time stamp issue); returns
based on official settled 3連単 payout for 100JPY each. Research-only.
"""
import argparse, csv, json, math, hashlib, itertools, re
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

TYPES=("S級","A級1・2班","チャレンジA3")
RIDER_WEIGHTS=(0.0,0.7,1.5)
ROLE_WEIGHTS=(0.0,0.6)
RARITY_WEIGHTS=(0.0,0.35,0.7)
COHESION_WEIGHTS=(0.0,0.3)
POINTS=(4,8,16,32)
TICKET_YEN=100
FIT_END="2025-10-01"
PAYOUT_CAP_YEN=100000

def file_sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda:f.read(2**20),b""):h.update(chunk)
    return h.hexdigest()

def races_from_csv(path):
    d={}
    with open(path,encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            rid=row["race_id"]
            r=d.setdefault(rid,{"id":rid,"day":row["date"],"cat":row["category"],
                 "n":int(row["rider_count"]),"low":int(row["is_low"]),"entrants":[],"actual":{}})
            if r["day"]!=row["date"] or r["cat"]!=row["category"]:
                raise RuntimeError("Conflicting race group")
            try:
                pos=int(row["finish_position"]) if row["finish_position"] not in ("",None) else 0
            except ValueError:pos=0
            car=int(row["bike"])
            r["entrants"].append({"id":row["rider_id"],"bike":car,
                  "rank":int(row["ura_rank"]),"role":str(row["line_role"] or ""),
                  "line":str(row["line_group"] or "")})
            if pos in (1,2,3):
                if pos in r["actual"]:raise RuntimeError("Duplicate top 3 "+rid)
                r["actual"][pos]=car
    for rid,r in d.items():
        if r["cat"] not in TYPES:continue
        n=len(r["entrants"])
        if n<5 or n>9 or r["n"]!=n:
            raise RuntimeError("Incomplete original predictor lineup "+rid)
        if {x["rank"] for x in r["entrants"]}!=set(range(1,n+1)):
            raise RuntimeError("Bad frozen ranking "+rid)
        if set(r["actual"])!={1,2,3}:raise RuntimeError("Incomplete race top3 "+rid)
    return [r for r in d.values() if r["cat"] in TYPES]

def load_payouts(path):
    out={}
    with open(path,encoding="utf-8-sig",newline="") as f:
        for x in csv.DictReader(f):
            rid=x["race_id"]
            if rid in out:raise RuntimeError("Duplicate official payout "+rid)
            amount=x.get("sanrentan_payout","")
            ordered=x.get("sanrentan_combo","")
            parts=re.findall(r"[1-9]",str(ordered))
            out[rid]=({"amount":int(float(amount)),"ordered":tuple(map(int,parts))}
                      if amount and len(parts)==3 else None)
    return out

def audit_races(races,payouts):
    counts=Counter();usable=[]
    for r in races:
        counts["race_from_frozen_rank_archive"]+=1
        if r["cat"] not in TYPES:raise RuntimeError("Unexpected grade")
        p=payouts.get(r["id"])
        if p is None or p["amount"]<=0:
            counts["missing_payout"]+=1;continue
        actual=tuple(r["actual"][pos] for pos in (1,2,3))
        if p["ordered"]!=actual:
            counts["actual_payout_mismatch"]+=1;continue
        r["payout"]=p["amount"]
        r["winner"]=actual
        usable.append(r);counts["official_settlement_matched"]+=1
        counts["eligible_"+r["cat"]]+=1
    return usable,dict(counts)

def fit_model(races):
    starts=Counter();wins=Counter()
    role_starts=Counter();role_wins=Counter()
    rider_starts=Counter();rider_obs=Counter();rider_expected=Counter()
    for r in races:
        cat,n=r["cat"],r["n"]
        for x in r["entrants"]:
            k=(cat,n,x["rank"])
            starts[k]+=1
            role=(cat,n,x["rank"],x["role"])
            role_starts[role]+=1
            for pos in (1,2,3):
                if r["actual"][pos]==x["bike"]:
                    wins[(*k,pos)]+=1
                    role_wins[(*role,pos)]+=1
                rid=(cat,x["id"],pos)
                rider_starts[(cat,x["id"])]+=int(pos==1)
                rider_obs[rid]+=int(r["actual"][pos]==x["bike"])
    # Prior rank curves (category+n+rank) are pooled over *training only*.
    base={}
    for cat,n,rank in starts:
        for pos in (1,2,3):
            p=(wins[(cat,n,rank,pos)]+5/n)/(starts[(cat,n,rank)]+5)
            base[f"{cat}|{n}|{rank}|{pos}"]=float(p)
    role_lift={}
    for cat,n,rank,role in role_starts:
        for pos in (1,2,3):
            b=base[f"{cat}|{n}|{rank}|{pos}"]
            p=(role_wins[(cat,n,rank,role,pos)]+32*b)/(role_starts[(cat,n,rank,role)]+32)
            role_lift[f"{cat}|{n}|{rank}|{role}|{pos}"]=float(max(-1.,min(1.,math.log(p/b))))
    for r in races:
        for x in r["entrants"]:
            for pos in (1,2,3):
                rid=(r["cat"],x["id"],pos)
                rider_expected[rid]+=base[f"{r['cat']}|{r['n']}|{x['rank']}|{pos}"]
    rider_lift={}
    for (cat,rid,pos),obs in rider_obs.items():
        expected=rider_expected[(cat,rid,pos)]
        alpha=15
        # Shrink observed vs expected with 15 effective entries at typical
        # baseline probability; do NOT require minimum sample starts.
        n=rider_starts[(cat,rid)]
        typical=expected/n if n else 0.
        numerator=obs+alpha*typical
        denominator=expected+alpha*typical
        lift=math.log(numerator/denominator) if denominator>0 else 0.
        rider_lift[f"{cat}|{rid}|{pos}"]=round(max(-1.3,min(1.3,lift)),7)
    return {"base":base,"role_lift":role_lift,"rider_lift":rider_lift,
            "rider_ids_with_2025_observations":len(rider_starts),
            "source_races":len(races),"trained_until":max(r["day"] for r in races)}

def race_scores(r,model):
    es=sorted(r["entrants"],key=lambda e:e["bike"])
    n=len(es);cat=r["cat"]
    scores=np.zeros((3,n));player=np.zeros((3,n));roles=np.zeros((3,n))
    for z,e in enumerate(es):
        for pos in (1,2,3):
            b=model["base"].get(f"{cat}|{n}|{e['rank']}|{pos}",1/n)
            scores[pos-1,z]=math.log(max(1e-8,b))
            player[pos-1,z]=model["rider_lift"].get(f"{cat}|{e['id']}|{pos}",0.)
            roles[pos-1,z]=model["role_lift"].get(f"{cat}|{n}|{e['rank']}|{e['role']}|{pos}",0.)
    idx=np.asarray(list(itertools.permutations(range(n),3)),dtype=np.int32)
    ii,jj,kk=idx[:,0],idx[:,1],idx[:,2]
    base=scores[0,ii]+scores[1,jj]+scores[2,kk]
    rider=player[0,ii]+player[1,jj]+player[2,kk]
    role=roles[0,ii]+roles[1,jj]+roles[2,kk]
    # Soft incentives are only grid-tuned with 2025 cash returns.
    # No tickets forbidden by odds, weak rider or line structure.
    rarity=np.asarray([
        (0.35*(es[i]["rank"]-1)+0.7*(es[j]["rank"]-1)+(es[k]["rank"]-1))/n
        for i,j,k in idx],float)
    group=[e["line"] for e in es]
    cohesion=np.asarray([
        sum(group[p] not in ("","0","nan","None") and group[p]==group[q]
            for p,q in ((i,j),(j,k),(i,k))) for i,j,k in idx],float)
    return (np.array([e["bike"] for e in es]),idx,
            base,rider,role,rarity,cohesion)

def configurations():
    return [(pr,ro,rar,co) for pr in RIDER_WEIGHTS for ro in ROLE_WEIGHTS
            for rar in RARITY_WEIGHTS for co in COHESION_WEIGHTS]

def evaluate_grid(races,model):
    confs=configurations();points=POINTS
    returns=np.zeros((len(confs),len(points)),np.float64)
    capped=np.zeros_like(returns);hits=np.zeros_like(returns)
    amounts=np.zeros_like(returns)
    biggest=np.zeros_like(returns)
    for r in races:
        bikes,idx,base,rider,role,rarity,cohesion=race_scores(r,model)
        actual=np.asarray(r["winner"],int)
        winnerix=np.where(np.all(bikes[idx]==actual,axis=1))[0]
        if len(winnerix)!=1:raise RuntimeError("Winner not in permutations")
        target=int(winnerix[0])
        for c,(pr,ro,rar,co) in enumerate(confs):
            predicted=base+pr*rider+ro*role+rar*rarity+co*cohesion
            # Sorting stable ensures repeated runs identical.
            order=np.argsort(-predicted,kind="stable")
            location=int(np.where(order==target)[0][0])+1
            for p,K in enumerate(points):
                num=min(K,len(order))
                amounts[c,p]+=num*TICKET_YEN
                if location<=num:
                    hits[c,p]+=1
                    returns[c,p]+=r["payout"]
                    capped[c,p]+=min(r["payout"],PAYOUT_CAP_YEN)
                    biggest[c,p]=max(biggest[c,p],r["payout"])
    results=[]
    for c,(pr,ro,rar,co) in enumerate(confs):
        for p,K in enumerate(points):
            stake=amounts[c,p];paid=returns[c,p];cap=capped[c,p]
            roi=100*paid/stake if stake else 0
            cap_roi=100*cap/stake if stake else 0
            # 2025 score rewards documented sustainable cash performance
            # (not just a single colossal winner). Raw ROI always reported.
            quality=.5*(roi+cap_roi)
            results.append({"player_weight":pr,"role_weight":ro,
                "rarity_weight":rar,"line_weight":co,"tickets_per_race":K,
                "races":len(races),"stake_yen":int(stake),
                "return_yen":int(paid),"capped_return_yen":int(cap),
                "max_one_win_yen":int(biggest[c,p]),"hits":int(hits[c,p]),
                "ROI_percent":round(roi,4),"capped_ROI_percent":round(cap_roi,4),
                "selection_quality":round(quality,4)})
    results.sort(key=lambda x:(-x["selection_quality"],-x["hits"],x["tickets_per_race"]))
    return results

def writecsv(path,records):
    if not records:raise RuntimeError("Empty csv write")
    with open(path,"w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,list(records[0]));w.writeheader();w.writerows(records)

def fit(args):
    from db.schema import get_connection
    import sqlite3
    allr=races_from_csv(args.exposures)
    if len({x["id"] for x in allr})<15000:raise RuntimeError("2025 complete original multi-class training source too small")
    payouts=load_payouts(args.payouts)
    eligible,audit=audit_races(allr,payouts)
    train=[r for r in eligible if r["day"]<FIT_END]
    valid=[r for r in eligible if r["day"]>=FIT_END]
    if len(train)<8000 or len(valid)<1500:raise RuntimeError("Insufficient valid 2025 time split")
    mod=fit_model(train)
    results=evaluate_grid(valid,mod)
    selected=results[0]
    print("2025_SELECTED_PARAMS",json.dumps(selected,ensure_ascii=False),flush=True)
    print("2025_NO_PLAYER_CONTROL",json.dumps(
        [r for r in results if r["player_weight"]==0][:4],ensure_ascii=False),flush=True)
    fullmodel=fit_model(eligible)
    result={"research_only":True,"purchase_approved":False,
        "format":"2025-player-informed-dynamic-trifecta-formations-v1",
        "training_year":"2025","hyperparam_train":"2025-01-01..2025-09-30",
        "hyperparam_select":"2025-10-01..2025-12-31",
        "2026_results_consulted_during_fitting":False,
        "strategy":{k:selected[k] for k in
            ("player_weight","role_weight","rarity_weight","line_weight","tickets_per_race")},
        "rider_model":fullmodel,"validation_metrics":selected,
        "train_source_sha256":file_sha(args.exposures),
        "payout_source_sha256":file_sha(args.payouts),
        "prior_year_archive_audit":audit,
        "2025_time_split_races":{"train":len(train),"validation":len(valid)},
        "max_points":max(POINTS),
        "non_girls_classes":TYPES,"supported_fields":[5,6,7,8,9],
        "no_7car_or_LOW_or_3car_line_filter":True,
        "place_combo_method":"Generate ALL legal ordered 3riders; score by observed 2025 place-rank baseline + shrinked personal slot residual + soft line-role/pattern + softness for rare ranks; buy top K tickets 100 yen each, K selected in 2025 Q4 only",
        "odds":"No final odds used for candidate selection; full official settlement payoff for ROI",
        "limitations":["Base ranker fitted before test selection but may have seen some of 2026 in an earlier URA pipeline; not guaranteed pristine model holdout.",
        "2025 fitted personal outcome residuals risk regression to mean; control w_player=0 must be compared.",
        "2025 Q4 strategy search is retrospective and selection bias remains.",
        "Because candidate selection does not read live quote, this is not expected-value-based market pricing.",
        "No online betting or modification of existing HaruQ."]
    }
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    out.joinpath("SEALED_2025_RIDER_FORMATION_POLICY.json").write_text(
        json.dumps(result,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n",
        encoding="utf8")
    writecsv(out/"2025_Q4_FORMATION_POLICY_SEARCH_ALL.csv",results)
    out.joinpath("2025_seal_audit.json").write_text(json.dumps({
        "frozen_policy_SHA256":file_sha(out/"SEALED_2025_RIDER_FORMATION_POLICY.json"),
        "2025_source_races":len(allr),"2025_settlement_eligible":len(eligible),
        "train":len(train),"validation":len(valid),
        "no_2026_used":True,"chosen":selected},indent=2,ensure_ascii=False)+"\n")
    print("2025_FORMATION_SEALED",json.dumps({"digest":file_sha(out/"SEALED_2025_RIDER_FORMATION_POLICY.json"),"policy":selected,"n_2025":len(eligible)},ensure_ascii=False),flush=True)

def score(args):
    conf=json.loads(Path(args.policy).read_text(encoding="utf8"))
    if not (conf["training_year"]=="2025" and
            conf["2026_results_consulted_during_fitting"] is False):
        raise RuntimeError("Policy not sealed without 2026")
    model=conf["rider_model"];s=conf["strategy"]
    allr=races_from_csv(args.exposures)
    if any(r["day"][:4]!="2026" for r in allr):
        raise RuntimeError("Forward test must be ONLY 2026!")
    payouts=load_payouts(args.payouts)
    selected,audit=audit_races(allr,payouts)
    detail=[];outs=Counter();cat=defaultdict(Counter)
    for r in selected:
        bikes,idx,base,rider,role,rarity,cohesion=race_scores(r,model)
        vals=base+s["player_weight"]*rider+s["role_weight"]*role+(
              s["rarity_weight"]*rarity+s["line_weight"]*cohesion)
        best=np.argsort(-vals,kind="stable")[:s["tickets_per_race"]]
        combos=[tuple(int(v) for v in bikes[idx[z]]) for z in best]
        won=r["winner"] in combos
        stake=len(combos)*100
        paid=r["payout"] if won else 0
        line={"race_id":r["id"],"date":r["day"],"category":r["cat"],
            "car_count":r["n"],"is_LOW":r["low"],
            "ticket_count":len(combos),
            "tickets":";".join("-".join(map(str,x)) for x in combos),
            "actual_combo":"-".join(map(str,r["winner"])),
            "hit":int(won),"stake_yen":stake,"return_yen":paid,
            "official_payout_per_100_yen":r["payout"],
            "player_list_2025_coverage":sum(f"{r['cat']}|{e['id']}|1" in model["rider_lift"] for e in r["entrants"]),
            "policy_sha256":file_sha(args.policy)}
        detail.append(line)
        for bucket in ("ALL",r["cat"],r["cat"]+"|"+("LOW" if r["low"] else "HIGH"),str(r["n"])+"car"):
            z=cat[bucket];z["races"]+=1;z["stake"]+=stake;z["paid"]+=paid;z["hits"]+=int(won)
    for k,z in cat.items():
        z["roi_pct"]=round(100*z["paid"]/z["stake"],4) if z["stake"] else None
    audit.update({"policy_sha256":file_sha(args.policy),"2026_source_races":len(allr),
        "2026_payout_evaluable":len(detail),"by_group":dict(cat),
        "purchase_approved":False,"model_retrained_with_2026":False})
    out=Path(args.out);out.mkdir(exist_ok=True,parents=True)
    if detail:writecsv(out/"per_race_ticket_ledger.csv",detail)
    out.joinpath("score_audit.json").write_text(json.dumps(audit,indent=2,ensure_ascii=False)+"\n")
    print("2026_RIDER_FORMATION_SCORE",json.dumps(audit,ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest="mode",required=True)
    for name in ("fit","score"):
        x=s.add_parser(name);x.add_argument("--exposures",required=True)
        x.add_argument("--payouts",required=True);x.add_argument("--out",required=True)
        if name=="score":x.add_argument("--policy",required=True)
    a=p.parse_args()
    if a.mode=="fit":fit(a)
    else:score(a)
if __name__=="__main__":main()
