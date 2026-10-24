"""V3: record frozen-URA pre-outcome rankings with real official outcomes from 2023 onward.

Read-only original model and SQLite history; year-sharded in GitHub Actions.
No training, type selection or 2025 outcome inspection occurs in this extractor.
Full archived trifecta odds are required; final odds are not pre-race quotes.
"""
import argparse
import csv
import hashlib
import json
import itertools
from collections import Counter
from pathlib import Path

import numpy as np
from db.schema import get_connection
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker
from ura.core import prepare,low_rank_start
from ura.roi_holdout import (odds_race_id,actual_and_payout,
                            feature_snapshot,FROZEN_HARUQ_MODEL_SHA256,
                            frozen_artifact)
from ura.formation_selector import load_fixed
from ura.formation_external_audit import archive_odds

def extract(year,model_path,odds_path,out_dir):
    if year not in (2023,2024,2025):
        raise ValueError("This research extraction is fixed to 2023-2025")
    base=load_fixed("artifacts/ura_ai_v1.json")
    frozen_artifact("artifacts/ura_ai_v1.json")
    if hashlib.sha256(Path(model_path).read_bytes()).hexdigest()!=FROZEN_HARUQ_MODEL_SHA256:
        raise RuntimeError("Original HaruQ model changed")
    ranker=LGBMRanker();ranker.load(str(model_path))
    builder=FeatureBuilder()
    first=f"{year}-01-01";last=f"{year+1}-01-01"
    books,archive_audit=archive_odds(odds_path,first,last)
    counts=Counter()
    rows=[]
    with get_connection() as conn:
        for race_id,book in sorted(books.items()):
            counts["archived_races"]+=1
            race=conn.execute("SELECT date,rider_count,race_name,grade FROM races WHERE race_id=?",(race_id,)).fetchone()
            if race is None:
                counts["not_in_history"]+=1;continue
            day=str(race["date"])[:10]
            if day.replace("-","")!=race_id[:8]:
                counts["date_mismatch"]+=1;continue
            if "ガール" in str(race["race_name"] or ""):
                counts["girls_excluded"]+=1;continue
            result=actual_and_payout(conn,race_id)
            if result is None:
                counts["missing_or_invalid_result"]+=1;continue
            combo,payout,entrants=result
            n=len(entrants)
            if n not in (7,8,9):
                counts["unsupported_field_size"]+=1;continue
            if n!=int(race["rider_count"]):
                counts["rider_count_mismatch"]+=1;continue
            if payout is None:
                counts["missing_official_payout"]+=1;continue
            odds=book["odds"]
            if len(odds)!=n*(n-1)*(n-2):
                counts["incomplete_final_odds_book"]+=1;continue
            if set(odds)!={"-".join(map(str,t)) for t in itertools.permutations(sorted(entrants),3)}:
                counts["odds_roster_mismatch"]+=1;continue
            if combo not in odds:
                counts["winning_combo_price_missing"]+=1;continue
            if abs(float(odds[combo])*100-float(payout))>1.5:
                counts["settlement_odds_disagreement"]+=1;continue
            try:
                source=feature_snapshot(builder,ranker,conn,race_id,day)
                if {int(v["bike_number"]) for v in source}!=entrants:
                    counts["feature_roster_mismatch"]+=1;continue
                riders,feat,gap,_=prepare(source)
                if gap>=1.0:
                    counts["high_confidence_excluded"]+=1;continue
                utilities=feat@np.array(base["parameters"],float)
                priority=sorted(range(n),key=lambda i:(-utilities[i],int(riders[i]["bike_number"])))
                bike_order=[int(riders[i]["bike_number"]) for i in priority]
                ranks={bike:i+1 for i,bike in enumerate(bike_order)}
                actual=tuple(ranks[int(b)] for b in combo.split("-"))
                if len(set(actual))!=3:raise ValueError("Repeated actual rank")
            except Exception as e:
                counts["feature_inference_failure"]+=1
                if counts["feature_inference_failure"]<=5:
                    print("V3_FEATURE_SKIPPED",race_id,type(e).__name__,str(e)[:180],flush=True)
                continue
            counts["matched_low_7to9"]=len(rows)+1
            rows.append({"race_id":race_id,"date":day,"year":year,"n":n,
                        "grade":str(race["grade"] or ""),"gap":round(gap,8),
                        "ranked_bikes":"-".join(map(str,bike_order)),
                        "actual_bikes":combo,
                        "actual_ranks":"-".join(map(str,actual)),
                        "payout_yen":int(payout),
                        "final_winning_odds":float(odds[combo])})
            if len(rows)%500==0:
                print("V3_EXTRACT_PROGRESS",json.dumps({"year":year,"matched":len(rows),
                    "scanned":counts["archived_races"]},ensure_ascii=False),flush=True)
    root=Path(out_dir);root.mkdir(parents=True,exist_ok=True)
    if rows:
        with (root/f"year_{year}.csv").open("w",newline="",encoding="utf8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={"year":year,"status":"complete" if rows else "no_eligible_races",
            "race_rows":len(rows),"archive":archive_audit,"audit":dict(counts),
            "original_model_sha256":FROZEN_HARUQ_MODEL_SHA256,
            "ura_blob_sha1":"5a353643cfe8196ab695521d479237294c3f9b66",
            "used_for_model_training":False,
            "notes":["Original HaruQ model may itself have trained on this past year.",
                     "Race-day historical features not proven as-of prestart.",
                     "Only complete final trifecta odds with official settlement consistency included."]}
    (root/f"audit_{year}.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    print("V3_EXTRACT_RESULT",json.dumps(report,ensure_ascii=False),flush=True)
    if not rows:raise RuntimeError(f"No eligible data to learn in {year}")
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--year",type=int,required=True)
    ap.add_argument("--model",type=Path,required=True)
    ap.add_argument("--odds",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    extract(args.year,args.model,args.odds,args.out)

if __name__=="__main__":
    main()
