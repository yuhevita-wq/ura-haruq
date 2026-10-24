"""Independent marginal formation learner, research only (7-9 riders).

Preserves original frozen HaruQ, frozen URA AI, and previous type selector.
All 1..6 v1 formations are disjoint ONLY at field sizes 7, 8, and 9.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np
from ura.formation_selector import choose, load_fixed, FIXED_HARUQ_SHA256

FROZEN_URA_BLOB="5a353643cfe8196ab695521d479237294c3f9b66"
MODEL_NAME="URA_MUTUALLY_EXCLUSIVE_FORMATION_CLASSIFIER_V1"

def predict_marginals(source, ura, shape, model):
    if source.get("model_sha256")!=FIXED_HARUQ_SHA256:
        raise ValueError("Original fixed model mismatch")
    if model.get("name")!=MODEL_NAME or model.get("frozen_ura_git_blob")!=FROZEN_URA_BLOB:
        raise ValueError("Marginal model fingerprint mismatch")
    first=choose(source,ura,shape)
    if first.get("selected_type") is None:
        return {"decision":"NOT_ELIGIBLE","research_only":True,"purchase_approved":False}
    shapes={int(z["type_id"]):z for z in [first["selected_type"],*first["other_types"]]}
    if sorted(shapes)!=list(range(1,7)):
        raise ValueError("Six formation types required")
    n=len(source["ranked"])
    if n not in (7,8,9):
        return {"decision":"UNSUPPORTED_5_6_RIDER_OVERLAP","research_only":True,"purchase_approved":False}
    gap=float(first["gap"])
    p=[float(shapes[i]["unadjusted_first_stage_probability"]) for i in range(1,7)]
    if any(z<=0 or z>=1 for z in p) or sum(p)>=1:
        raise ValueError("Invalid rank-formation probabilities")
    x=np.array([math.log(z/(1-z)) for z in p]+[gap,(n-7)/2],float)
    mean=np.array(model["scaler_mean"],float)
    scale=np.array(model["scaler_scale"],float)
    coef=np.array(model["coefficients"],float)
    intercept=np.array(model["intercepts"],float)
    if x.shape!=(8,) or mean.shape!=(8,) or scale.shape!=(8,) or coef.shape!=(7,8) or intercept.shape!=(7,):
        raise ValueError("Unexpected shape of marginal model")
    logits=coef@((x-mean)/scale)+intercept
    logits-=max(logits)
    probabilities=np.exp(logits);probabilities/=sum(probabilities)
    return {"decision":"RESEARCH","race_id":source.get("race_id"),"gap":gap,
            "n":n,"shapes":shapes,
            "probabilities_by_outcome":{str(i):float(probabilities[i]) for i in range(7)},
            "research_only":True,"purchase_approved":False}

def marginal_choices(prediction,already_selected=()):
    if prediction["decision"]!="RESEARCH":
        return {**prediction,"additional_options":[]}
    selected=tuple(already_selected)
    if len(set(selected))!=len(selected) or any(type(t)!=int or t not in range(1,7) for t in selected):
        raise ValueError("Selected types must be distinct integers 1..6")
    forms=prediction["shapes"];q=prediction["probabilities_by_outcome"]
    occupied=set().union(*(set(forms[i]["tickets"]) for i in selected)) if selected else set()
    options=[]
    for tid in range(1,7):
        if tid in selected:continue
        ticket_set=set(forms[tid]["tickets"])
        added=ticket_set-occupied
        if not added:continue
        if len(added)!=len(ticket_set):
            raise RuntimeError("Overlapping shapes need a different model; this v1 is disjoint")
        gain=float(q[str(tid)])
        options.append({"type_id":tid,"name":forms[tid]["name"],
                        "added_points":len(added),
                        "incremental_capture_probability":gain,
                        "capture_probability_per_new_point":gain/len(added),
                        "new_tickets":sorted(added)})
    options.sort(key=lambda z:(-z["capture_probability_per_new_point"],z["type_id"]))
    return {"decision":"RESEARCH","race_id":prediction["race_id"],
            "already_selected":list(selected),"existing_points":len(occupied),
            "already_covered_probability":sum(float(q[str(t)]) for t in selected),
            "none_of_six_probability":float(q["0"]),
            "additional_options":options,"research_only":True,"purchase_approved":False}

def frontier(prediction,max_types=6,required_types=()):
    if prediction["decision"]!="RESEARCH":return []
    if type(max_types)!=int or not 0<=max_types<=6:raise ValueError("max_types out of bounds")
    q=prediction["probabilities_by_outcome"];forms=prediction["shapes"]
    plans=[]
    for n_types in range(max_types+1):
        for combo in itertools.combinations(range(1,7),n_types):
            if not all(t in combo for t in required_types):continue
            points=sum(int(forms[i]["points"]) for i in combo)
            probability=sum(float(q[str(i)]) for i in combo)
            plans.append({"types":list(combo),"points":points,"expected_capture_probability":probability})
    plans.sort(key=lambda a:(a["points"],-a["expected_capture_probability"],a["types"]))
    efficient=[];best=-1.
    for p in plans:
        if p["expected_capture_probability"]>best+1e-12:
            efficient.append(p);best=p["expected_capture_probability"]
    return efficient

def analyze(source,ura,shape,model,already_selected=(),max_types=6,max_points=None):
    prediction=predict_marginals(source,ura,shape,model)
    if prediction["decision"]!="RESEARCH":return prediction
    if max_points is not None and (type(max_points)!=int or max_points<0):
        raise ValueError("max_points must be nonnegative integer")
    selections=marginal_choices(prediction,already_selected)
    efficient=frontier(prediction,max_types,already_selected)
    feasible=[z for z in efficient if (max_points is None or z["points"]<=max_points)
              and all(t in z["types"] for t in already_selected)]
    best=max(feasible,key=lambda z:(z["expected_capture_probability"],-z["points"])) if feasible else None
    return {**selections,
        "outcome_probabilities":prediction["probabilities_by_outcome"],
        "pareto_frontier":efficient,
        "max_points_scenario":max_points,
        "best_within_scenario":best,
        "no_outcomes_or_payouts_are_model_inputs":True,
        "research_only":True,"purchase_approved":False,
        "warning":"June patterns explored previously. 7-class June later-period logloss was worse than frozen URA baseline; final odds not pre-start verified."}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prediction",required=True,type=Path)
    p.add_argument("--ura",type=Path,default=Path("artifacts/ura_ai_v1.json"))
    p.add_argument("--shape",type=Path,default=Path("artifacts/ura_shape_selector_v1.json"))
    p.add_argument("--marginal",type=Path,default=Path("artifacts/ura_marginal_v1.json"))
    p.add_argument("--already",default="")
    p.add_argument("--max-types",type=int,default=6)
    p.add_argument("--max-points",type=int)
    p.add_argument("--out",type=Path,default=Path("ura_marginal_research.json"))
    args=p.parse_args()
    selected=tuple(int(z) for z in args.already.split(",") if z.strip())
    source=json.loads(args.prediction.read_text())
    result=analyze(source,load_fixed(args.ura),
                   json.loads(args.shape.read_text()),json.loads(args.marginal.read_text()),
                   selected,args.max_types,args.max_points)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps({"race_id":result.get("race_id"),"decision":result["decision"],
                      "research_only":True,"out":str(args.out)},ensure_ascii=False))
if __name__=="__main__":
    main()
