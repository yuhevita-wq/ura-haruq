"""Experimental LOW-conditioned combinatorial portfolios from concrete line-role shapes.

Research-only, never edits frozen 2-2-2-1, URA V1, HaruQ, website, or purchase policy.

Unlike ranking top-four individual shapes, optimize *entire* subset of all eligible
real 3-2-2 |12|23|13 line geometry shapes, separately in each LOW quarter.

Optimization uses 2023/24 inputs exclusively, with one predeclared bound:
nonnegative recorded net profit in EACH 2023 and 2024 cohort.

Strategies
  profit:   maximize combined observed net profit (then minimize tickets);
  coverage: maximize combined distinct hits (then minimize tickets).
No fixed maximum number of shapes: MILP chooses cardinality up to full library.

2025, 2026Q1, 2026Q2 are only evaluated AFTER selections are frozen.
The 2023/24 optimism from 108-shape testing and historical model leakage
remain substantial! This is neither forward performance nor live EV.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from db.schema import get_connection
from research.line_patterns_20261025.geometry_322 import BANDS, get_rows, load_322, structural_key
from research.line_patterns_20261025.low_portfolio import load_2026, make_record, as_years, aggregate

PERMITTED=("12","23","13")
PERIODS=("2023","2024","2025","2026Q1","2026Q2")
TRAIN=("2023","2024")
VARIANTS=("profit","coverage")
TOL=1e-7


def get_inventory(rows23,rows24):
    # ONLY the discovery years may introduce eligible types.
    inventory=sorted({
        kind for r in itertools.chain(rows23,rows24)
        for kind,points in r["tickets"].items()
        if points>0 and kind.rsplit("|",1)[-1] in PERMITTED
    })
    # Require at least ONE historical actual hit in each train year,
    # to avoid treating a one-off payout as year-to-year support.
    types=[]
    for kind in inventory:
        yr23=aggregate(rows23,[kind])
        yr24=aggregate(rows24,[kind])
        if yr23["hits"]<1 or yr24["hits"]<1:
            continue
        if yr23["investment_yen"]<=0 or yr24["investment_yen"]<=0:
            continue
        types.append({
            "type":kind,"2023_hits":yr23["hits"],"2024_hits":yr24["hits"],
            "2023_points":yr23["points"],"2024_points":yr24["points"],
            "2023_net_yen":yr23["net_yen"],"2024_net_yen":yr24["net_yen"],
            "2023_roi_pct":yr23["roi_pct"],"2024_roi_pct":yr24["roi_pct"],
        })
    return inventory,types


def solve(types,mode):
    """Exact binary optimization with explicit yearly no-loss constraints.

    Stage A: maximize financial objective (yen) or distinct hits.
    Stage B: lock optimum to exact solver tolerance, then minimize total tickets.
    Both steps are 2023+24 ONLY; no 2025+ holdout can influence selection.
    """
    if mode not in VARIANTS:raise ValueError("invalid research objective")
    if not types:
        return {"selected":[],"objective_value":0,"solver":"empty_library","types_considered":0}
    n=len(types)
    net23=np.asarray([x["2023_net_yen"] for x in types],dtype=float)
    net24=np.asarray([x["2024_net_yen"] for x in types],dtype=float)
    npoints=np.asarray([x["2023_points"]+x["2024_points"] for x in types],float)
    nhits=np.asarray([x["2023_hits"]+x["2024_hits"] for x in types],float)
    profit=(net23+net24)  # integer yen objective, no artificial rounding
    scores=profit if mode=="profit" else nhits
    rows=np.vstack([net23,net24])
    yearly_constraints=LinearConstraint(rows,np.zeros(2),np.full(2,np.inf))
    args=dict(integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),options={"time_limit":90,"mip_rel_gap":0.0})
    solution=milp(c=-scores,constraints=yearly_constraints,**args)
    if solution.status!=0 or solution.x is None:
        raise RuntimeError(f"Solver failure {mode}: {solution.message}")
    best=np.rint(solution.x).astype(int)
    opt=float(scores@best)
    if abs(round(opt)-opt)>TOL:raise RuntimeError("Unexpected non-integer objective")
    # Within the EXACT stage A maximum, choose cheaper betting coverage.
    # Rows & score are integer-valued, so bound optimal score to opt-0.01.
    tightened=LinearConstraint(np.vstack([rows,scores]),
                               np.array([0,0,opt-.01]),
                               np.array([np.inf,np.inf,np.inf]))
    sparse=milp(c=npoints,constraints=tightened,**args)
    if sparse.status!=0 or sparse.x is None:
        raise RuntimeError(f"Tie-break failure {mode}: {sparse.message}")
    chosen=np.rint(sparse.x).astype(int)
    if abs(scores@chosen-opt)>TOL:
        raise RuntimeError("Solver sacrificed primary objective")
    if (rows@chosen < -TOL).any():
        raise RuntimeError("Historic nonnegative constraint broken")
    result=[x["type"] for i,x in enumerate(types) if chosen[i]]
    return {"selected":result,"objective_value":int(round(opt)),
            "solver":"scipy.milp_two_stage_exact","types_considered":n,
            "selected_count":len(result),
            "train_profit_2023_yen":int(net23@chosen),
            "train_profit_2024_yen":int(net24@chosen),
            "train_hits":int(nhits@chosen),
            "train_points":int(npoints@chosen)}


def format_summary(cohort,selected):
    acc=aggregate(cohort,selected)
    acc.pop("days")
    return acc


def candidate_explanations(types,selected):
    selected=set(selected)
    outputs=[]
    for row in types:
        y23,y24=row["2023_net_yen"],row["2024_net_yen"]
        outputs.append({**row,"selected":int(row["type"] in selected),
                        "combined_net_yen":y23+y24,
                        "train_worst_net_yen":min(y23,y24)})
    return sorted(outputs,key=lambda x:(-x["selected"],-x["combined_net_yen"],x["type"]))


def evaluate(label_dir,year26,out):
    past=get_rows(label_dir)
    future=load_2026(year26)
    audit=Counter()
    records=[]
    with get_connection() as conn:
        for r in itertools.chain(past,future):
            if int(r["n"])!=7:continue
            if not 0<=float(r["gap"])<1:
                raise ValueError("Discovered non-LOW input row")
            ranked=tuple(int(v) for v in r["ranked_bikes"].split("-"))
            lookup,status=load_322(conn,r["race_id"],ranked)
            audit[status]+=1
            if lookup is None:continue
            records.append(make_record(r,lookup))
    by=as_years(records)
    # The ONLY read of training rows in optimizers; all 2025+ cohorts are
    # untouched until the selected lists are finalized.
    policies={};catalog=[];training=[]
    for interval in BANDS:
        r23=by[("2023",interval)]
        r24=by[("2024",interval)]
        inventory,eligible=get_inventory(r23,r24)
        policies[interval]={
            "discovered_total_geometry_types":len(inventory),
            "eligible_after_1hit_per_year":len(eligible),
            "strategies":{mode:solve(eligible,mode) for mode in VARIANTS}
        }
        selected_by_mode={m:set(policies[interval]["strategies"][m]["selected"])
                          for m in VARIANTS}
        for rec in eligible:
            catalog.append({**{"LOW_band":interval},**rec,
                            **{f"selected_{m}":int(rec["type"] in selected_by_mode[m]) for m in VARIANTS}})
        for mode in VARIANTS:
            selected=policies[interval]["strategies"][mode]["selected"]
            for kind in selected:
                base=next(z for z in eligible if z["type"]==kind)
                training.append({"LOW_band":interval,"strategy":mode,**base})
    # Benchmark = prior 4-type low_portfolio chooser; always research-only
    from research.line_patterns_20261025.low_portfolio import select_band
    baseline={i:select_band(by[("2023",i)],by[("2024",i)])["selected_types"] for i in BANDS}
    portfolio_by_band=[]
    overall=[]
    per_race=[]
    for mode in VARIANTS+("prior_4type",):
        choices={i:policies[i]["strategies"][mode]["selected"] if mode in VARIANTS else baseline[i] for i in BANDS}
        for year in PERIODS:
            year_records=[]
            for interval in BANDS:
                cohort=by[(year,interval)]
                result=format_summary(cohort,choices[interval])
                portfolio_by_band.append({"year":year,"LOW_band":interval,"variant":mode,
                                          "selected_count":len(choices[interval]),
                                          "selected_shapes":";".join(choices[interval]),**result})
                year_records.extend(cohort)
                for race in cohort:
                    included=set(choices[interval])
                    pts=sum(race["tickets"][k] for k in included)
                    hit=int(race["winner_type"] in included)
                    per_race.append({"race_id":race["race_id"],"date":race["date"],
                                     "year":year,"LOW_band":interval,"strategy":mode,
                                     "selected_types":len(choices[interval]),
                                     "winning_structure":race["observed_structure"],
                                     "points":pts,"hit":hit,"investment_yen":100*pts,
                                     "paid_yen":hit*race["payout"],
                                     "net_yen":hit*race["payout"]-100*pts})
            # Aggregate per race with different LOW band type choices
            combined=[]
            for race in year_records:
                subset=choices[race["LOW_band"]]
                pts=sum(race["tickets"][k] for k in subset)
                hit=int(race["winner_type"] in subset)
                combined.append({"date":race["date"],"winner_type":"X" if hit else None,
                                 "payout":race["payout"],"tickets":Counter({"X":pts})})
            total=format_summary(combined,["X"])
            overall.append({"year":year,"variant":mode,**total})
    result={
        "experiment":"URA_322_LOW_COMBINATORIAL_MILP_EXHAUSTIVE_REAL_ROLE_TYPES",
        "changes_to_frozen_models":False,"frozen_2221_unchanged":True,
        "research_only":True,"purchase_approved":False,
        "type_roles":"3H 3B 3T, 2H 2B, and same-line relation 12/23/13",
        "all_candidate_relation_types":list(PERMITTED),
        "training":["2023","2024"],
        "heldout_examination":["2025","2026Q1","2026Q2"],
        "optimization_policies":{
            "profit":"MILP exact: max 2023+2024 net yen; both years net >=0; exact optimum then min number of purchased tickets. No 4-type cardinality cap.",
            "coverage":"MILP exact: max 2023+2024 distinct winning tickets; both years net >=0; exact optimum then min purchased tickets. Loss-making individual shapes can be subsidized by profitable ones. No 4-type cap."
        },
        "explicit_training_eligibility":"only known 2023+2024 geometries with >=1 observed win IN EACH year within the specified LOW band",
        "LOW_bands":list(BANDS),"cohort_counts":{y:sum(r["split"]==y for r in records) for y in PERIODS},
        "policies":policies,"previous_4type_baseline":baseline,
        "overall":overall,"audit":dict(audit),
        "limitations":[
            "Holdout periods were already studied in prior conversations; not a genuinely unseen prospective test",
            "Optimizing ~100 sparse types across only 2023/24 can overfit rare high-payout wins",
            "Selection uses realized 3連単 settlements; winning payout is unknowable before a real bet",
            "Frozen HaruQ historic features may include unavailable prestart fields or in-sample model training",
            "No quoted odds at 15-minutes-to-close are validated here",
            "This solves retrospective combinatorial portfolio goals, NOT causal probabilities or guaranteed forward edge",
            "No modification to existing predictions, web, 2-2-2-1, purchase criteria or bankroll"
        ]
    }
    root=Path(out);root.mkdir(exist_ok=True,parents=True)
    (root/"research_model.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    for filename,rows in [
        ("all_train_candidate_types.csv",catalog),("selected_types_explained.csv",training),
        ("portfolio_by_low_and_year.csv",portfolio_by_band),
        ("overall_by_year.csv",overall),("all_race_ledgers.csv",per_race)]:
        if not rows:continue
        with (root/filename).open("w",newline="",encoding="utf8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print("URA_COMBINATORIAL_322",json.dumps({
        "library":{b:{"all":p["discovered_total_geometry_types"],
                       "eligible":p["eligible_after_1hit_per_year"]}
                  for b,p in policies.items()},
        "selection":{b:{mode:policies[b]["strategies"][mode]
                        for mode in VARIANTS} for b in BANDS},
        "overall":overall
    },ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--year26",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    evaluate(a.input,a.year26,a.out)


if __name__=="__main__":
    main()
