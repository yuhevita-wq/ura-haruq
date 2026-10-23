#!/usr/bin/env python3
"""URA-only prediction from a saved HaruQ score JSON and trained URA AI artifact."""
import argparse
import csv
import json
from pathlib import Path

from ura.core import select_candidates
from ura.train import FROZEN_MODEL_SHA256

def load_odds(path,race_id):
    if path is None:return {}
    out={}
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        for r in csv.DictReader(stream):
            if str(r.get("race_id",""))!=str(race_id):continue
            ticket=str(r["ticket"]).replace("=","-").strip()
            price=float(r["odds"])
            if price>1 and ticket not in out:
                out[ticket]=price
    return out

def analyze(source,trained,odds=None):
    if not trained.get("training_completed") or not trained.get("trained_on_real_results"):
        raise ValueError("An actually trained URA model is required")
    if trained.get("source_frozen_model_sha256")!=FROZEN_MODEL_SHA256:
        raise ValueError("URA model frozen-source SHA mismatch")
    if source.get("model_sha256")!=FROZEN_MODEL_SHA256:
        raise ValueError("HaruQ snapshot source model SHA mismatch")
    if source.get("grade") in ("GIRLS","ガールズ"):
        raise ValueError("Girls races are not eligible")
    ranked=[{"bike_number":int(x["bike"]),"score":float(x["score"])}
            for x in source["ranked"]]
    result=select_candidates(ranked,trained["parameters"])
    # Respect a LOW classification derived only from source original model scores.
    if abs(result["confidence_gap"]-float(source["confidence"]))>0.002:
        raise ValueError("Source confidence and displayed ranks are inconsistent")
    prices=load_odds(odds,source["race_id"])
    for candidate in result["candidates"]:
        if candidate["ticket"] in prices:
            price=prices[candidate["ticket"]]
            candidate["odds"]=price
            candidate["estimated_roi_per_yen"]=candidate["probability"]*price
            candidate["estimated_ev_per_yen"]=candidate["probability"]*price-1.
    return {
       "system":"URA_HARUQ_V1",
       "race_id":source["race_id"],
       "race_date":source.get("date"),
       "velodrome":source.get("velodrome"),
       "grade":source.get("grade"),
       "source_model_sha256":FROZEN_MODEL_SHA256,
       "source_ranking_unchanged":True,
       "trained_ura_parameters":trained["parameters"],
       **result,
       "odds_observed":len(prices),
       "automatic_purchase":False,
       "purchase_amount_yen":0,
       "caveat":"Odds snapshots must be timestamped pre-start to backtest expected value. No ROI guarantee."
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prediction",type=Path,required=True,
                   help="Saved HaruQ JSON, copied READ-ONLY")
    p.add_argument("--trained",type=Path,required=True,
                   help="ura_ai_v1.json from successful independent trainer")
    p.add_argument("--odds",type=Path,help="Optional odds CSV: race_id,ticket,odds")
    p.add_argument("--out",type=Path,default=Path("ura_prediction.json"))
    args=p.parse_args()
    prediction=json.loads(args.prediction.read_text(encoding="utf-8"))
    model=json.loads(args.trained.read_text(encoding="utf-8"))
    result=analyze(prediction,model,args.odds)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"race_id":result["race_id"],
       "confidence":result["confidence_gap"],
       "research_candidates":result["candidate_count"],
       "result":str(args.out)},ensure_ascii=False))

if __name__=="__main__":main()
