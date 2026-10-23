#!/usr/bin/env python3
"""Learn URA-HARUQ from real historical results and the unchanged frozen HaruQ AI.

The original 44-feature ranker is used read-only. This code trains ONLY the
URA probability calibration layer. No original model, feature or policy edits.
"""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from config import CLASS_MAP, DB_PATH, DEFAULT_GEAR_RATIO
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker
from ura.core import DEFAULT_COEFFICIENTS, FEATURE_NAMES, prepare

FROZEN_MODEL_SHA256="e70690f4538b77e927a042e7ae2db312f8a0a4f16bf72def37e7ccad3568c193"
DATE_START="2025-07-01"
TRAIN_END="2026-04-01"
VALID_END="2026-06-01"
TEST_END="2026-10-01"
MIN_COUNTS={"training":500,"validation":100,"holdout":100}

def historical_frame(model_path):
    if hashlib.sha256(Path(model_path).read_bytes()).hexdigest()!=FROZEN_MODEL_SHA256:
        raise RuntimeError("Frozen source model SHA256 mismatch")
    model=LGBMRanker()
    model.load(str(model_path))
    with sqlite3.connect(str(DB_PATH)) as conn:
        eligible=pd.read_sql_query("""
        SELECT r.race_id, r.date, r.grade, r.race_name
        FROM races r JOIN race_results rr ON rr.race_id=r.race_id
        WHERE r.rider_count BETWEEN 5 AND 9
        GROUP BY r.race_id
        HAVING COUNT(*)=r.rider_count
          AND COUNT(DISTINCT rr.bike_number)=r.rider_count
          AND SUM(CASE WHEN rr.finish_position IS NULL THEN 1 ELSE 0 END)=0
          AND COUNT(DISTINCT rr.finish_position)=r.rider_count
          AND SUM(CASE WHEN rr.finish_position=1 THEN 1 ELSE 0 END)=1
          AND SUM(CASE WHEN rr.finish_position=2 THEN 1 ELSE 0 END)=1
          AND SUM(CASE WHEN rr.finish_position=3 THEN 1 ELSE 0 END)=1
        """,conn)
        gear=pd.read_sql_query("SELECT race_id,rider_id,gear_ratio AS source_gear FROM entries",conn)
    if len(eligible)<10000:raise RuntimeError("Not enough complete historical races")
    builder=FeatureBuilder()
    features=list(builder.feature_names)
    if len(features)!=44 or features!=list(model.feature_names):
        raise RuntimeError("Frozen HaruQ 44-feature contract mismatch")
    df=builder.build_dataset(2023,2026)
    if df.empty:raise RuntimeError("No dataset from existing history")
    df=df.merge(eligible[["race_id","grade","race_name"]],on="race_id",
                how="inner",validate="many_to_one")
    # Original faithful 2026-03-21 replay corrections, without re-training model.
    df["rider_class_num"]=df["class"].map(CLASS_MAP).fillna(6).astype(float)
    df["class_rank_in_race"]=df.groupby("race_id")["rider_class_num"].rank(method="min")
    df["rider_class_x_jiku"]=df["rider_class_num"]*df["line_is_jiku"]
    df=df.merge(gear,on=["race_id","rider_id"],how="left",validate="many_to_one")
    df["entry_gear_ratio"]=pd.to_numeric(df["source_gear"],errors="coerce").fillna(DEFAULT_GEAR_RATIO)
    df.drop(columns=["source_gear"],inplace=True)
    df=df[(df["race_date"]>=DATE_START)&(df["race_date"]<TEST_END)].copy()
    # HaruQ is men's race only. Exclude women's classes and explicitly labeled girls races.
    is_girls=df["class"].fillna("").astype(str).str.upper().str.startswith("L")
    is_girls|=df["race_name"].fillna("").astype(str).str.contains("ガールズ",regex=False)
    skip=set(df.loc[is_girls,"race_id"])
    df=df.loc[~df["race_id"].isin(skip)].copy()
    if not len(df):raise RuntimeError("No eligible historical rows after exclusions")
    frame=df[features].apply(pd.to_numeric,errors="coerce").fillna(0.0)
    df["score"]=model.predict(frame).values
    return df,{"source_complete_races":len(eligible),
               "after_date_and_gender_filter":int(df["race_id"].nunique()),
               "excluded_girls_races":len(skip)}

def collect(df):
    rows=[]
    n_ineligible=0
    for race_id,g in df.groupby("race_id",sort=True):
        try:
            ordered,features,gap,low_flags=prepare(
                [{"bike_number":int(x.bike_number),"score":float(x.score)}
                 for x in g.itertuples(index=False)])
            actual={}
            for x in g.itertuples(index=False):
                actual[int(x.finish_position)]=int(x.bike_number)
            y=[next(i for i,t in enumerate(ordered)
                    if int(t["bike_number"])==actual[pos]) for pos in (1,2,3)]
            if len(set(y))!=3:raise ValueError("Nonunique 1st-3rd")
            rows.append({"race_id":str(race_id),"date":str(g.iloc[0]["race_date"])[:10],
                         "features":features,"actual":y,"gap":gap,
                         "low_flags":low_flags,"n":len(ordered)})
        except (KeyError,StopIteration,TypeError,ValueError):
            n_ineligible+=1
    if not rows:raise RuntimeError("No valid race features/outcomes")
    return rows,n_ineligible

def tensorize(rows):
    x=np.zeros((len(rows),9,len(FEATURE_NAMES)),float)
    mask=np.zeros((len(rows),9),bool)
    yy=np.zeros((len(rows),3),int)
    for i,r in enumerate(rows):
        n=r["n"]
        x[i,:n,:]=r["features"]
        mask[i,:n]=True
        yy[i,:]=r["actual"]
    return x,mask,yy

def loss_gradient(theta,tensors,penalty=0.):
    x,mask,finish=tensors
    logits=np.einsum("rnf,f->rn",x,theta)
    available=mask.copy()
    idx=np.arange(len(finish))
    loss=0.
    gradient=np.zeros(len(theta),float)
    for pos in range(3):
        logits_valid=np.where(available,logits,-np.inf)
        norm=logsumexp(logits_valid,axis=1)
        p=np.exp(logits_valid-norm[:,None])
        winner=finish[:,pos]
        loss+=float(np.mean(norm-logits[idx,winner]))
        gradient+=np.mean(np.einsum("rn,rnf->rf",p,x)-x[idx,winner],axis=0)
        available[idx,winner]=False
    delta=theta-DEFAULT_COEFFICIENTS
    return loss+penalty*float(delta@delta),gradient+2*penalty*delta

def subset(tensor,choice):
    return tuple(x[choice] for x in tensor)

def summarize(tensor,choice,theta):
    n=int(choice.sum())
    if not n:return {"races":0,"nll":None,"untrained_reference_nll":None}
    sub=subset(tensor,choice)
    return {"races":n,
            "nll":round(float(loss_gradient(theta,sub)[0]),7),
            "untrained_reference_nll":round(float(loss_gradient(DEFAULT_COEFFICIENTS,sub)[0]),7)}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",required=True,type=Path)
    parser.add_argument("--out",default="ura_outputs",type=Path)
    args=parser.parse_args()
    df,source_audit=historical_frame(args.model)
    rows,rejected=collect(df)
    low_rows=[r for r in rows if r["gap"]<1.0]
    d=np.asarray([r["date"] for r in low_rows])
    train=d<TRAIN_END
    val=(d>=TRAIN_END)&(d<VALID_END)
    test=(d>=VALID_END)&(d<TEST_END)
    counts={"training":int(train.sum()),"validation":int(val.sum()),
            "holdout":int(test.sum())}
    if any(counts[k]<MIN_COUNTS[k] for k in MIN_COUNTS):
        raise RuntimeError(f"Insufficient genuine LOW races: {counts}; minimum {MIN_COUNTS}")
    tensor=tensorize(low_rows)
    fitted=minimize(loss_gradient,DEFAULT_COEFFICIENTS.copy(),
                    args=(subset(tensor,train),0.05),
                    jac=True,method="L-BFGS-B",options={"maxiter":300})
    if not fitted.success:raise RuntimeError("URA model fit failed: "+str(fitted.message))
    y=tensor[2]
    lower=np.array([[bool(r["low_flags"][i]) for i in actual]
                    for r,actual in zip(low_rows,y)])
    holdout_tail_hits=int((lower.any(axis=1)&test).sum())
    output={
       "system":"URA_HARUQ_INDEPENDENT_AI_V1",
       "training_completed":True,
       "trained_on_real_results":True,
       "source_frozen_model_sha256":FROZEN_MODEL_SHA256,
       "original_model_modified":False,
       "source_code_modified":False,
       "policy":"LOW confidence gap < 1.0 AND at least one lower-rank rider in the ticket",
       "low_rank_definition":"Ranks 5-7 for 7 riders, ranks 6-9 for 9 riders",
       "date_ranges":{"start":DATE_START,"train_end_exclusive":TRAIN_END,
                      "validation_end_exclusive":VALID_END,"holdout_end_exclusive":TEST_END},
       "features":list(FEATURE_NAMES),
       "parameters":list(map(float,fitted.x)),
       "sample_counts":counts,
       "source_data_audit":source_audit,
       "all_qualified_races":len(rows),
       "low_confidence_races":len(low_rows),
       "rejected_races":rejected,
       "train":summarize(tensor,train,fitted.x),
       "validation":summarize(tensor,val,fitted.x),
       "holdout":summarize(tensor,test,fitted.x),
       "holdout_actual_lower_rider_in_top3":holdout_tail_hits,
       "holdout_lower_rider_top3_share":round(holdout_tail_hits/counts["holdout"],6),
       "warning":"No pre-race full trifecta odds in this pipeline. Profitability, expected value and live feature parity are NOT verified. Historical builder uses race_results odds/popularity; verify pre-start availability before claiming deployable predictive performance.",
       "purchase_approved":False
    }
    args.out.mkdir(parents=True,exist_ok=True)
    target=args.out/"ura_ai_v1.json"
    target.write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("URA_AI_LEARNED",json.dumps({"counts":counts,
          "holdout":output["holdout"],
          "lower_rider_hit_share":output["holdout_lower_rider_top3_share"],
          "artifact":str(target)},ensure_ascii=False),flush=True)

if __name__=="__main__":main()
