"""Independent multi-period cash audit of FIXED formation models, no re-fitting.

Evaluation on April/May 2026 (unused for formation discovery or formation fitting).
Odds are FINAL archived prices, not verified pre-race quotations.
All strategies are PREDECLARED, applied to identical eligible 7-9-rider races.
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

from db.schema import get_connection
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker
from ura.roi_holdout import (odds_race_id,actual_and_payout,feature_snapshot,
                             frozen_artifact,FROZEN_HARUQ_MODEL_SHA256)
from ura.formation_selector import load_fixed
from ura.formation_marginal import predict_marginals,frontier,MODEL_NAME
from ura.core import prepare

BEGIN="2026-04-01"
END_EXCLUSIVE="2026-06-01"
STAKE=100
STRATEGIES=(
 "type1","type1_2","all6",
 "old_one","old_top2","old_top3",
 "new_top2","new_top3","new_1_add_density","new_1_add_hit",
 "new_max16","new_max24",
)

def archive_odds(path,first=BEGIN,last=END_EXCLUSIVE):
    books={}; ambiguous=set(); count=Counter()
    with gzip.open(path,"rt",encoding="utf-8") as fp:
        for raw in fp:
            o=json.loads(raw)
            try:rid=odds_race_id(o["race_id"])
            except (ValueError,KeyError,TypeError):
                count["bad_id"]+=1;continue
            date=rid[:8]
            if not (first.replace("-","")<=date<last.replace("-","")):continue
            tri=o.get("odds",{}).get("3連単")
            if not isinstance(tri,dict) or not tri:
                count["no_odds"]+=1;continue
            count["source_records"]+=1
            if rid in ambiguous:
                count["ignored_ambiguous"]+=1;continue
            if rid in books:
                if tri==books[rid]["odds"]:
                    count["identical_duplicate"]+=1
                else:
                    books.pop(rid);ambiguous.add(rid);count["conflicting_duplicate"]+=1
                continue
            books[rid]={"odds":tri,"source_race_id":o["race_id"]}
    count["ambiguous_races"]=len(ambiguous)
    count["distinct_races"]=len(books)
    return books,dict(count)

def simple_bootstrap(rows,rep=3000,seed=20261024):
    if not rows:return None
    a=np.array([(z["stake"],z["returned"]) for z in rows],dtype=np.float64)
    gen=np.random.default_rng(seed)
    n=len(rows); vals=[]
    for _ in range(rep):
        b=a[gen.integers(n,size=n)].sum(axis=0)
        if b[0]>0: vals.append(100*b[1]/b[0])
    return [round(float(x),2) for x in np.quantile(vals,[.025,.975])] if vals else None

def choice_map(prediction):
    shapes=prediction["shapes"]
    baseline=sorted(shapes,key=lambda t:(
        -shapes[t]["estimated_capture_probability"]/shapes[t]["points"],t))
    q=prediction["probabilities_by_outcome"]
    alternative=sorted(shapes,key=lambda t:(-q[str(t)]/shapes[t]["points"],t))
    by_capture=sorted((z for z in shapes if z!=1),key=lambda t:(-q[str(t)],t))
    by_density=sorted((z for z in shapes if z!=1),key=lambda t:(
        -q[str(t)]/shapes[t]["points"],t))
    strategies={
        "type1":(1,),
        "type1_2":(1,2),
        "all6":tuple(range(1,7)),
        "old_one":tuple(baseline[:1]),
        "old_top2":tuple(baseline[:2]),
        "old_top3":tuple(baseline[:3]),
        "new_top2":tuple(alternative[:2]),
        "new_top3":tuple(alternative[:3]),
        "new_1_add_density":(1,by_density[0]),
        "new_1_add_hit":(1,by_capture[0]),
    }
    for key,budget in (("new_max16",16),("new_max24",24)):
        options=[p for p in frontier(prediction,max_types=6) if p["points"]<=budget]
        best=max(options,key=lambda p:(p["expected_capture_probability"],-p["points"],tuple(-x for x in p["types"])))
        strategies[key]=tuple(best["types"])
    return strategies

def evaluate(artifact,results,seed=20261024):
    grouped=defaultdict(list)
    by_month=defaultdict(list)
    for z in results:
        grouped[z["strategy"]].append(z)
        by_month[(z["strategy"],z["month"])].append(z)
    def summarize(v):
        invested=sum(x["stake"] for x in v)
        paid=sum(x["returned"] for x in v)
        hits=sum(x["hit"] for x in v)
        return {"races":len(v),"tickets":sum(x["points"] for x in v),
                "hits":hits,"stake_yen":invested,"paid_yen":paid,
                "net_yen":paid-invested,
                "roi_percent":round(100*paid/invested,4) if invested else None,
                "race_bootstrap_roi_95pct":simple_bootstrap(v,seed=seed)}
    return {"by_strategy":{k:summarize(grouped[k]) for k in STRATEGIES},
            "monthly":{m:{k:summarize(by_month[(k,m)]) for k in STRATEGIES}
                       for m in sorted({x["month"] for x in results})}}

def run(model_path,odds_path,out_dir):
    base=load_fixed("artifacts/ura_ai_v1.json")
    frozen_artifact("artifacts/ura_ai_v1.json")
    shape=json.loads(Path("artifacts/ura_shape_selector_v1.json").read_text())
    marginal=json.loads(Path("artifacts/ura_marginal_v1.json").read_text())
    if marginal["name"]!=MODEL_NAME or marginal["fit_races"]!=219 or marginal["source_ura_not_retrained"] is not True:
        raise RuntimeError("Marginal model fingerprints do not match frozen research")
    if hashlib.sha256(Path(model_path).read_bytes()).hexdigest()!=FROZEN_HARUQ_MODEL_SHA256:
        raise RuntimeError("Frozen original model bytes changed")
    source_model=LGBMRanker();source_model.load(str(model_path))
    builder=FeatureBuilder()
    books,coverage=archive_odds(odds_path)
    counts=Counter()
    rows=[]
    evidence=[]
    with get_connection() as conn:
        for rid,book in sorted(books.items()):
            counts["odds_race_id"]+=1
            race=conn.execute("SELECT date,rider_count,race_name,grade FROM races WHERE race_id=?",(rid,)).fetchone()
            if race is None:
                counts["no_original_race"]+=1;continue
            date=str(race["date"])[:10]
            if date.replace("-","")!=rid[:8]:
                counts["race_date_disagrees"]+=1;continue
            if "ガール" in str(race["race_name"] or ""):
                counts["girls_excluded"]+=1;continue
            official=actual_and_payout(conn,rid)
            if not official:
                counts["invalid_results"]+=1;continue
            winning,payout,entrants=official
            n=len(entrants)
            if n not in (7,8,9):
                counts["unsupported_field_size"]+=1;continue
            if int(race["rider_count"])!=n:
                counts["rider_count_mismatch"]+=1;continue
            if payout is None:
                counts["missing_official_payout"]+=1;continue
            prices=book["odds"]
            if len(prices)!=n*(n-1)*(n-2):
                counts["partial_trifecta_book"]+=1;continue
            if set(prices)!={"-".join(map(str,t)) for t in itertools.permutations(sorted(entrants),3)}:
                counts["different_odds_roster"]+=1;continue
            try:
                scores=feature_snapshot(builder,source_model,conn,rid,date)
                if {int(x["bike_number"]) for x in scores}!=entrants:
                    counts["feature_rider_mismatch"]+=1;continue
                ordered=sorted(scores,key=lambda x:(-x["score"],int(x["bike_number"])))
                gap=float(ordered[0]["score"]-ordered[1]["score"])
                if gap>=1.:
                    counts["not_low"]+=1;continue
                snapshot={"race_id":rid,"date":date,
                          "model_sha256":FROZEN_HARUQ_MODEL_SHA256,
                          "ranked":[{"bike":int(z["bike_number"]),"score":float(z["score"])} for z in scores],
                          "confidence":gap}
                prediction=predict_marginals(snapshot,base,shape,marginal)
                if prediction["decision"]!="RESEARCH":
                    counts["model_rejected"]+=1;continue
                choices=choice_map(prediction)
            except Exception as exc:
                counts["feature_or_inference_exception"]+=1
                if counts["feature_or_inference_exception"]<=8:
                    print("SKIP_EXCEPTION",rid,type(exc).__name__,str(exc)[:220],flush=True)
                continue
            counts["matched_eligible_low_races"]+=1
            month=date[:7]
            forms=prediction["shapes"]
            unique_win_types=[i for i in range(1,7) if winning in forms[i]["tickets"]]
            if len(unique_win_types)>1:
                raise RuntimeError("Overlapping 7-9 riders shapes: invalid marginal outcomes")
            # Auditable pre-outcome prediction ranks for learning NEW formation families.
            # Never infer ranking from the winner or odds; winner is used only as label.
            urariders,urax,_,_=prepare(scores)
            utilities=urax@np.asarray(base["parameters"],float)
            priority=sorted(range(n),key=lambda i:(-utilities[i],int(urariders[i]["bike_number"])))
            priority_bikes=[int(urariders[i]["bike_number"]) for i in priority]
            ranks_by_bike={b:i+1 for i,b in enumerate(priority_bikes)}
            actual_ranks=[ranks_by_bike[int(v)] for v in winning.split("-")]
            if len(set(actual_ranks))!=3:
                raise RuntimeError("Incorrect actual rank reconstruction")
            evidence.append({"race_id":rid,"date":date,"month":month,
                        "winner":winning,"payout":payout,"n":n,
                        "gap":round(gap,5),"actual_outcome_type":unique_win_types[0] if unique_win_types else 0,
                        "all6_points":sum(forms[i]["points"] for i in range(1,7)),
                        "marginal_p_none":round(prediction["probabilities_by_outcome"]["0"],8),
                        "ura_probability_rank_bikes":"-".join(map(str,priority_bikes)),
                        "actual_rank_positions":"-".join(map(str,actual_ranks)),
                        "ura_rank_utilities":";".join(f"{utilities[i]:.12g}" for i in priority)})
            for strategy,types in choices.items():
                tickets=set().union(*(set(forms[i]["tickets"]) for i in types)) if types else set()
                if sum(forms[i]["points"] for i in types)!=len(tickets):
                    raise RuntimeError("Unexpected ticket duplicate; cannot claim full stake")
                invested=STAKE*len(tickets)
                was_hit=int(winning in tickets)
                rows.append({"race_id":rid,"date":date,"month":month,
                    "strategy":strategy,"selected_types":"+".join(map(str,types)),
                    "points":len(tickets),"stake":invested,
                    "returned":int(payout)*was_hit,"hit":was_hit,
                    "actual_combo":winning,"official_payout":payout,
                    "field_size":n,
                    "predicted_capture_probability":sum(prediction["probabilities_by_outcome"][str(t)] for t in types)})
            if counts["matched_eligible_low_races"]%100==0:
                print("EVALUATION_PROGRESS",json.dumps({"matched":counts["matched_eligible_low_races"],
                          "inspected":counts["odds_race_id"],"audit":dict(counts)},ensure_ascii=False),flush=True)
    if not rows:
        raise RuntimeError("No fully matched eligible races, cannot report return")
    details=evaluate(None,rows)
    report={"system":"URA_FORMATION_MULTIPERIOD_FROZEN_ROI_V1",
       "eligible_date_start":BEGIN,"eligible_date_end_exclusive":END_EXCLUSIVE,
       "all_models_frozen":True,"no_training_or_selection_on_evaluation_set":True,
       "frozen_base_sha256":FROZEN_HARUQ_MODEL_SHA256,
       "frozen_ura_blob_sha": "5a353643cfe8196ab695521d479237294c3f9b66",
       "shape_artifact_sha":hashlib.sha256(Path("artifacts/ura_shape_selector_v1.json").read_bytes()).hexdigest(),
       "marginal_artifact_sha":hashlib.sha256(Path("artifacts/ura_marginal_v1.json").read_bytes()).hexdigest(),
       "candidate_strategies_fixed_before_result_access":list(STRATEGIES),
       "odds_type":"historical final 3連単, NOT verified pre-race executable",
       "hypothesis":"April-May 2026 are outside formation model fitting; these preceded June training and were seen in original 2026 holdout/validation analyses.",
       "archive_counts":coverage,"join_quality":dict(counts),
       "results":details,
       "prediction_inputs_time_audit":"NOT PASSED: original race features may use final entries odds/popularity and weather, current rider class; cannot claim pre-start parity.",
       "purchase_approved":False}
    root=Path(out_dir);root.mkdir(parents=True,exist_ok=True)
    (root/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    for name,items in (("per_strategy_race.csv",rows),("eligible_races.csv",evidence)):
        with (root/name).open("w",newline="",encoding="utf-8") as fp:
            w=csv.DictWriter(fp,fieldnames=list(items[0]));w.writeheader();w.writerows(items)
    compact={k:{n:vv for n,vv in obj.items() if n!="race_bootstrap_roi_95pct"}
             for k,obj in details["by_strategy"].items()}
    print("MULTIPERIOD_FROZEN_RESULTS",json.dumps({"races":counts["matched_eligible_low_races"],
                 "months":sorted(details["monthly"]),
                 "portfolio":compact,"audit":dict(counts)},ensure_ascii=False),flush=True)
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",type=Path,required=True)
    ap.add_argument("--odds",type=Path,required=True)
    ap.add_argument("--out",type=Path,default=Path("ura_external_odds_validation"))
    x=ap.parse_args()
    run(x.model,x.odds,x.out)

if __name__=="__main__":main()
