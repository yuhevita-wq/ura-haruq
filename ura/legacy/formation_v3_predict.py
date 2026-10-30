"""Inference from frozen URA and newly trained 2023-founded V3 rank formations.

Never modifies original predictor. Produces every qualifying 3連単 ticket; no bets.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from ura.core import prepare,ordered_probabilities,low_rank_start
from ura.legacy.formation_selector import load_fixed,FIXED_HARUQ_SHA256

EXPECTED_ARTIFACT="URA_FORMATION_V3_2023_TRAINED_UNDERDOG_COMPLIANT"
EXPECTED_TRAINED_SHA="8bebb7d7ea6caff71b56ef4eeb0e80908621a61fd6404f07350f7641972850e9"
EXPECTED_URA_BLOB="5a353643cfe8196ab695521d479237294c3f9b66"

def bin_rank(rank,n):
    return 1+7*(rank-1)//n

def research_prediction(source,ura,v3):
    if source.get("model_sha256")!=FIXED_HARUQ_SHA256:
        raise ValueError("Original source model fingerprint mismatch")
    if (v3.get("system")!=EXPECTED_ARTIFACT or
        v3.get("full_source_model_v3_json_sha256")!=EXPECTED_TRAINED_SHA or
        v3.get("frozen_ura_git_blob")!=EXPECTED_URA_BLOB or
        v3.get("base_model_sha256")!=FIXED_HARUQ_SHA256 or
        v3.get("purchase_approved") is not False):
        raise ValueError("V3 compliant research artifact fingerprint mismatch")
    if ura.get("source_frozen_model_sha256")!=FIXED_HARUQ_SHA256:
        raise ValueError("Old URA model mismatched")
    prepared=[{"bike_number":int(v["bike"]),"score":float(v["score"])} for v in source["ranked"]]
    riders,x,gap,_=prepare(prepared)
    n=len(riders)
    if abs(float(source["confidence"])-gap)>0.002:
        raise ValueError("Original source gap mismatch")
    if gap>=1. or n not in (7,8,9):
        return {"race_id":source.get("race_id"),"decision":"NOT_ELIGIBLE",
                "tickets":[],"points":0,"research_only":True,"purchase_approved":False}
    score=x@np.asarray(ura["parameters"],float)
    priority=sorted(range(n),key=lambda i:(-score[i],int(riders[i]["bike_number"])))
    bikes=[int(riders[i]["bike_number"]) for i in priority]
    probs=ordered_probabilities(x,np.asarray(ura["parameters"],float))
    price_probs={(int(riders[i]["bike_number"]),int(riders[j]["bike_number"]),int(riders[k]["bike_number"])):p
                 for (i,j,k),p in probs.items()}
    union=set();forms=[]
    for shape in v3["new_types"]:
        bounds=shape["rank_bin_bounds"]
        if len(bounds)!=3 or any(len(v)!=2 or v[0]<1 or v[1]>7 or v[0]>v[1] for v in bounds):
            raise ValueError("Invalid saved learned shape")
        selected=[]
        for ranks in itertools.permutations(range(1,n+1),3):
            if max(ranks)<low_rank_start(n):
                continue
            if all(lo<=bin_rank(pos,n)<=hi for pos,(lo,hi) in zip(ranks,bounds)):
                selected.append(ranks)
        tickets=["-".join(map(str,(bikes[a-1],bikes[b-1],bikes[c-1]))) for a,b,c in selected]
        dup=union.intersection(tickets)
        if dup:raise RuntimeError("V3 learned disjoint formations unexpectedly overlap")
        union.update(tickets)
        forms.append({"id":shape["id"],"rider_rank_bin_ranges":bounds,
                      "points":len(tickets),"tickets":tickets,
                      "sum_first_stage_probability":round(sum(price_probs[tuple(map(int,t.split("-")))] for t in tickets),8)})
    return {"system":"URA_HARUQ_V3_2023_FORMATION_RESEARCH",
            "race_id":source.get("race_id"),"date":source.get("date"),"source_gap":gap,
            "ura_probability_rank_bikes":bikes,"field_size":n,
            "frozen_first_stage":True,
            "formations":forms,"tickets":sorted(union),"points":len(union),
            "each_ticket_has_lower_rank_rider":True,
            "decision":"RESEARCH_ONLY",
            "research_only":True,"purchase_approved":False,
            "warning":"Retrospective research; V3 2025 return was 87.82%. No pre-start odds and feature equivalence verified."}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prediction",type=Path,required=True)
    ap.add_argument("--trained-ura",type=Path,default=Path("artifacts/ura_ai_v1.json"))
    ap.add_argument("--trained-v3",type=Path,default=Path("artifacts/ura_formation_v3_2023_compliant.json"))
    ap.add_argument("--out",type=Path,default=Path("ura_v3_research.json"))
    args=ap.parse_args()
    out=research_prediction(json.loads(args.prediction.read_text()),
         load_fixed(args.trained_ura),json.loads(args.trained_v3.read_text()))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(out,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"race_id":out["race_id"],"decision":out["decision"],
                       "points":out["points"],"approved":False},ensure_ascii=False))
if __name__=="__main__":
    main()
