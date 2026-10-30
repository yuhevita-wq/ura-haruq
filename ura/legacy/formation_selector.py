"""Independent stage-two formation selector for frozen URA-HARUQ v1.

Selects ONE research formation using archived fitted capture probabilities.
Does not modify or re-train stage one, and never authorizes bets.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
from scipy.special import expit
from ura.core import prepare, ordered_probabilities, low_rank_start

FIXED_URA_BLOB = "5a353643cfe8196ab695521d479237294c3f9b66"
FIXED_HARUQ_SHA256 = "e70690f4538b77e927a042e7ae2db312f8a0a4f16bf72def37e7ccad3568c193"
CATALOG = (
    (1,"下位逆転",("tail","top1","mid2_5")),
    (2,"本命3着回帰",("tail","mid2_5","top1")),
    (3,"下位2着割り込み",("rank2_4","tail","top1")),
    (4,"下位3着侵入",("top1_2","rank3_4","tail")),
    (5,"大穴の2・3着同居",("rank3_5","tail_last2","last")),
    (6,"本命頭・下位2着",("top1","tail","mid2_5")),
)
def rank_sets(n):
    if not 5<=n<=9: raise ValueError("Require 5 to 9 riders")
    tail=low_rank_start(n)
    return {"tail":tuple(range(tail,n+1)),
            "top1":(1,),"mid2_5":tuple(range(2,min(5,n)+1)),
            "rank2_4":(2,3,4),"top1_2":(1,2),"rank3_4":(3,4),
            "rank3_5":tuple(range(3,min(5,n)+1)),
            "tail_last2":tuple(range(max(tail,n-1),n+1)),"last":(n,)}
def rank_tickets(n,type_id):
    if type_id not in range(1,7):raise ValueError("Unknown catalog type")
    _,_,spec=CATALOG[type_id-1]
    sets=rank_sets(n)
    lo=low_rank_start(n)
    return tuple((a,b,c) for a,b,c in itertools.product(*(sets[x] for x in spec))
                 if len({a,b,c})==3 and max(a,b,c)>=lo)
def logit_features(p,gap,n):
    p=float(np.clip(p,1e-6,1-1e-6))
    return [math.log(p/(1-p)),float(gap),float((n-7)/2)]
def load_fixed(path):
    raw=Path(path).read_bytes()
    blob=hashlib.sha1(f"blob {len(raw)}\0".encode()+raw).hexdigest()
    if blob!=FIXED_URA_BLOB:
        raise ValueError("First-stage coefficients were changed")
    model=json.loads(raw)
    if not model.get("training_completed") or model.get("source_frozen_model_sha256")!=FIXED_HARUQ_SHA256:
        raise ValueError("Invalid original frozen URA artifact")
    return model
def choose(source,ura,shape):
    if source.get("model_sha256")!=FIXED_HARUQ_SHA256:
        raise ValueError("HaruQ model checksum mismatch")
    if shape.get("stage_two_version")!="URA_SHAPE_SELECTOR_V1" or \
       shape.get("frozen_ura_git_blob")!=FIXED_URA_BLOB or \
       shape.get("base_model_sha256")!=FIXED_HARUQ_SHA256:
        raise ValueError("Second-stage model does not match frozen first stage")
    ranked=[{"bike_number":int(t["bike"]),"score":float(t["score"])} for t in source["ranked"]]
    riders,features,gap,_=prepare(ranked)
    if abs(gap-float(source["confidence"]))>0.002:
        raise ValueError("Source gap mismatch")
    if gap>=1.:
        return {"race_id":source.get("race_id"),"decision":"NOT_ELIGIBLE","selected_type":None,
                "research_only":True,"purchase_approved":False}
    theta=np.asarray(ura["parameters"],dtype=float)
    utilities=features@theta
    priority=sorted(range(len(riders)),key=lambda i:(-utilities[i],int(riders[i]["bike_number"])))
    bikes=[int(riders[i]["bike_number"]) for i in priority]
    likelihood=ordered_probabilities(features,theta)
    per_combo={(int(riders[i]["bike_number"]),int(riders[j]["bike_number"]),int(riders[k]["bike_number"])):p
               for (i,j,k),p in likelihood.items()}
    shapes=[]
    for tid,name,spec in CATALOG:
        ranked_tickets=rank_tickets(len(bikes),tid)
        tickets=[tuple(bikes[a-1] for a in ranks) for ranks in ranked_tickets]
        probability=sum(per_combo[t] for t in tickets)
        x=np.asarray(logit_features(probability,gap,len(bikes)))
        fitted=shape["type_models"][str(tid)]
        p=float(expit(float(fitted["intercept"])+x@np.asarray(fitted["weights"])))
        shapes.append({"type_id":tid,"name":name,"rank_positions":list(spec),
                       "points":len(tickets),
                       "estimated_capture_probability":p,
                       "estimated_captures_per_ticket":p/len(tickets),
                       "unadjusted_first_stage_probability":float(probability),
                       "tickets":["-".join(map(str,t)) for t in tickets]})
    shapes.sort(key=lambda t:(-t["estimated_captures_per_ticket"],t["type_id"]))
    return {"race_id":source.get("race_id"),"date":source.get("date"),
            "system":"URA_HARUQ_TWO_STAGE_RESEARCH_V1",
            "gap":gap,"ura_probability_order_bikes":bikes,
            "selected_type":shapes[0],"other_types":shapes[1:],
            "research_only":True,"purchase_approved":False,
            "warning":"Only capture-per-point is estimated; final odds ROI failed forward-like June comparison. Never auto-bet."}
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prediction",type=Path,required=True)
    ap.add_argument("--ura",type=Path,default=Path("artifacts/ura_ai_v1.json"))
    ap.add_argument("--shape",type=Path,default=Path("artifacts/ura_shape_selector_v1.json"))
    ap.add_argument("--out",type=Path,default=Path("ura_two_stage_output.json"))
    args=ap.parse_args()
    result=choose(json.loads(args.prediction.read_text()),load_fixed(args.ura),
                  json.loads(args.shape.read_text()))
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"race_id":result["race_id"],"type":result["selected_type"]["type_id"] if result["selected_type"] else None,
                      "research_only":True,"out":str(args.out)},ensure_ascii=False))
if __name__=="__main__":main()
