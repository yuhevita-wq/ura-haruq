"""Research-only global LOW-portfolio learning: NO per-type filtering.

Rather than discarding a shape because its own median/ROI is poor, this
evaluates whole-portfolio hit frequency, ticket cost, payout distribution,
and annual profitability, by LOW band.

2023: reference payout medians; 2023+24: joint MILP portfolio fit.
2025, 2026Q1/Q2: strictly score the already selected configurations.
2025/26 were seen in earlier studies, so NOT virgin test sets.

Never modify fixed HaruQ or URA V1 or 2-2-2-1. No live predictions.
"""
from __future__ import annotations
import argparse,csv,itertools,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from db.schema import get_connection
from research.line_patterns_20261025.geometry_322 import BANDS,get_rows,load_322
from research.line_patterns_20261025.low_portfolio import load_2026,make_record,as_years
from research.line_patterns_20261025.high_payout_stability import payout_reference_2023,annual_profile
from research.line_patterns_20261025.optimal_combinations import PERMITTED,PERIODS

# Predeclared portfolio-level targets; no per-shape cutoff is imposed.
CONFIGS=((0.0,True),(0.1,True),(0.2,True),(0.0,False))
SEED=20261025


def inventory_from_training(train23,train24):
    return sorted({
      k for r in itertools.chain(train23,train24) for k,v in r["tickets"].items()
      if v>0 and k.rsplit("|",1)[-1] in PERMITTED
    })


def cohort_vectors(cohort,types,cut):
    lookup={t:i for i,t in enumerate(types)}
    stake=np.zeros(len(types),float)
    paid=np.zeros(len(types),float)
    hits=np.zeros(len(types),float)
    high=np.zeros(len(types),float)
    low=np.zeros(len(types),float)
    for r in cohort:
        for key,pts in r["tickets"].items():
            i=lookup.get(key)
            if i is not None:stake[i]+=100*pts
        i=lookup.get(r["winner_type"])
        if i is not None:
            hits[i]+=1
            paid[i]+=r["payout"]
            if r["payout"]>=cut:high[i]+=1
            else:low[i]+=1
    return dict(stake=stake,paid=paid,hits=hits,high=high,low=low)


def exact_portfolio(types,yrs,margin,gate_median):
    """Max whole-portfolio distinct wins; constrain each 2023,24 year ROI,
    optionally annual above/below-median hit mix, then min ticket stakes.
    ALL types are candidates. Single-type frequency/ROI/median never gates.
    """
    n=len(types)
    if n==0:
        return {"types":[],"type_count":0,"training_hits":0,
                "status":"empty","reason":"no candidate types"}
    constraints=[]
    for y in ("2023","2024"):
        v=yrs[y]
        constraints.append(v["paid"]-(1+margin)*v["stake"])
        if gate_median:constraints.append(v["high"]-v["low"])
    A=np.stack(constraints)
    score=yrs["2023"]["hits"]+yrs["2024"]["hits"]
    cost=yrs["2023"]["stake"]+yrs["2024"]["stake"]
    base=LinearConstraint(A,0,np.inf)
    args=dict(integrality=np.ones(n),bounds=Bounds(np.zeros(n),np.ones(n)),
              options={"time_limit":60,"mip_rel_gap":0})
    primary=milp(-score,constraints=base,**args)
    if primary.status!=0 or primary.x is None:
        raise RuntimeError("portfolio MILP stage1: "+primary.message)
    best=float(score@np.rint(primary.x))
    # The objective is integer hits; this exact lock forbids losing one hit
    # when tie-breaking by number of tickets.
    additional=LinearConstraint(np.vstack([A,score]),
               np.concatenate([np.zeros(len(A)),[best-.01]]),np.inf)
    secondary=milp(cost,constraints=additional,**args)
    if secondary.status!=0 or secondary.x is None:
        raise RuntimeError("portfolio MILP stage2: "+secondary.message)
    z=np.rint(secondary.x).astype(int)
    if abs(score@z-best)>1e-7 or np.min(A@z)<-1e-6:
        raise RuntimeError("mip solution violates maximum hits or constraint")
    chosen=[t for t,selected in zip(types,z) if selected]
    return {"types":chosen,"type_count":len(chosen),
            "training_hits":int(score@z),"train_investment_yen":int(cost@z),
            "constraint_margin_roi_pct":100*(1+margin),
            "requires_high_payout_wins_at_least_half":gate_median,
            "status":"optimal",
            "train_each_year":{
                y:{"wins":int(yrs[y]["hits"]@z),
                   "investment_yen":int(yrs[y]["stake"]@z),
                   "paid_yen":int(yrs[y]["paid"]@z),
                   "high_paid_wins":int(yrs[y]["high"]@z),
                   "cheap_paid_wins":int(yrs[y]["low"]@z)}
                for y in ("2023","2024")}}


def portfolio_score(rows,selection,cut):
    allow=set(selection)
    rcount=len(rows)
    purchased=[];awards=[];days=defaultdict(lambda:[0,0])
    wagers=[];hits=0
    for r in rows:
        pts=sum(r["tickets"][k] for k in allow)
        hit=r["winner_type"] in allow
        amount=r["payout"] if hit else 0
        wagers.append(pts)
        purchased.append(100*pts)
        if hit:awards.append(r["payout"]);hits+=1
        d=days[r["date"]];d[0]+=100*pts;d[1]+=amount
    stake=sum(purchased);paid=sum(awards)
    mean=float(np.mean(awards)) if awards else None
    med=float(np.median(awards)) if awards else None
    q25=float(np.quantile(awards,.25)) if awards else None
    return {"races":rcount,"type_count":len(allow),
            "races_with_any_ticket":sum(v>0 for v in wagers),
            "hits":hits,"hitrate_pct":round(100*hits/rcount,3) if rcount else None,
            "points":sum(wagers),
            "avg_points_per_race":round(float(np.mean(wagers)),3) if rcount else None,
            "median_points_per_race":float(np.median(wagers)) if rcount else None,
            "invested_yen":stake,"returned_yen":paid,"net_yen":paid-stake,
            "roi_pct":round(100*paid/stake,3) if stake else None,
            "payout_mean_yen":round(mean,2) if mean is not None else None,
            "payout_median_yen":round(med,2) if med is not None else None,
            "payout_q25_yen":round(q25,2) if q25 is not None else None,
            "needed_mean_payout_to_break_even_yen":round(stake/hits,2) if hits else None,
            "realized_high_payout_hits":sum(v>=cut for v in awards),
            "realized_below_reference_hits":sum(v<cut for v in awards),
            "top3_payout_share_pct":round(100*sum(sorted(awards,reverse=True)[:3])/paid,2)
                    if paid else None}


def diagnose_drops(train,types,chosen,ref,margin,gate_median):
    """An audit at joint-solution level: adding ANY excluded type must either
    contribute zero hits or violate annual *global* portfolio constraints.
    We do NOT say the excluded type by itself is bad.
    """
    decisions=[]
    core=set(chosen)
    base={yr:cohort_vectors(train[yr],list(core),ref) for yr in ("2023","2024")}
    # Calculations on groups via precomputed annual stats so no hidden heuristics.
    for kind in types:
        if kind in core:
            decisions.append({"type":kind,"selected":True,
                              "diagnostic":"chosen_as_part_of_joint_optimum"})
            continue
        alt=list(core|{kind})
        v={yr:cohort_vectors(train[yr],alt,ref) for yr in ("2023","2024")}
        fails=[]
        for yr in ("2023","2024"):
            if v[yr]["paid"].sum() + 1e-8 < (1+margin)*v[yr]["stake"].sum():
                fails.append(yr+"_portfolio_ROI_constraint")
            if gate_median and v[yr]["high"].sum()<v[yr]["low"].sum():
                fails.append(yr+"_portfolio_median_mix_constraint")
        found=sum(sum(1 for r in train[yr] if r["winner_type"]==kind)
                  for yr in ("2023","2024"))
        decisions.append({"type":kind,"selected":False,
                          "extra_train_hits_if_added":found,
                          "diagnostic":";".join(fails) if fails else
                                       "no_training_hit_gain_or_exact_joint_optimum"})
    return decisions


def main_run(labeldir,year26,outdir):
    source=get_rows(labeldir);future=load_2026(year26)
    audit=Counter();samples=[]
    with get_connection() as conn:
        for row in itertools.chain(source,future):
            if int(row["n"])!=7:continue
            order=tuple(int(v) for v in row["ranked_bikes"].split("-"))
            lookup,status=load_322(conn,row["race_id"],order)
            audit[status]+=1
            if lookup is None:continue
            samples.append(make_record(row,lookup))
    by=as_years(samples)
    manifest={"name":"URA_322_LOW_JOINT_HIT_PAYOUT_PORTFOLIO_FRONTIER",
              "frozen_2221_unchanged":True,"frozen_haruq_unchanged":True,
              "frozen_ura_unchanged":True,"purchase_approved":False,
              "research_only":True,
              "method":"all 66 roles, NO independent type gate. Whole portfolio ROI target each 2023/24 year, optional annual payout distribution target, maximize overall distinct hit count; min stakes after max hits",
              "training":["2023","2024"],"evaluation":["2025","2026Q1","2026Q2"],
              "payout_reference":"2023 LOW-band median of ALL realized |12,|23,|13 race results, regardless of underdog ticket eligibility",
              "scenarios":[{"margin":m,"median_constraint":g} for m,g in CONFIGS],
              "selection":{},"subject_to_historical_snapshot_caveat":True,
              "audit":dict(audit),
              "cohort_counts":{y:sum(r["split"]==y for r in samples) for y in PERIODS}}
    policy={};rows=[];decisions=[];baseline=[]
    for band in BANDS:
        train23=by[("2023",band)];train24=by[("2024",band)]
        ref=payout_reference_2023(train23)
        types=inventory_from_training(train23,train24)
        # There are no individual minimum-hit/ROI/median requirements.
        yy={y:cohort_vectors(by[(y,band)],types,ref["cutoff_yen"])
            for y in ("2023","2024")}
        manifest["selection"][band]={"reference":ref,"candidate_types":len(types),
                                      "policies":{}}
        for margin,require_median in CONFIGS:
            name=f"roi{int(100*(1+margin))}_{'median' if require_median else 'free'}"
            model=exact_portfolio(types,yy,margin,require_median)
            manifest["selection"][band]["policies"][name]=model
            policy[(band,name)]=model["types"]
            report=diagnose_drops({"2023":train23,"2024":train24},types,
                                  model["types"],ref["cutoff_yen"],margin,require_median)
            decisions.extend({"LOW_band":band,"scenario":name,**x} for x in report)
            for year in PERIODS:
                score=portfolio_score(by[(year,band)],model["types"],ref["cutoff_yen"])
                rows.append({"LOW_band":band,"scenario":name,"year":year,
                             "reference_payout_yen":ref["cutoff_yen"],**score})
            # 2023/24 training frontier: exact portfolio, not handpicked
            baseline.append({"LOW_band":band,"scenario":name,
                             "type_count":len(model["types"]),
                             "training_hits":model["training_hits"],
                             "training_stakes":model["train_investment_yen"]})
    totals=[]
    # An overall across-LOW summary requires the band-specific ticket sets.
    for margin,require_median in CONFIGS:
        name=f"roi{int(100*(1+margin))}_{'median' if require_median else 'free'}"
        for year in PERIODS:
            all_races=[r for r in samples if r["split"]==year]
            decorated=[]
            for r in all_races:
                types=set(policy[(r["LOW_band"],name)])
                n=sum(r["tickets"][k] for k in types)
                decorated.append({"date":r["date"],"LOW_band":r["LOW_band"],
                                  "tickets":Counter({"X":n}),
                                  "winner_type":"X" if r["winner_type"] in types else None,
                                  "payout":r["payout"]})
            # Need per-race distribution in original payout units for aggregated median.
            global_payouts=[r["payout"] for r in decorated if r["winner_type"]=="X"]
            s=sum(r["tickets"]["X"] for r in decorated)
            wins=len(global_payouts);paid=sum(global_payouts)
            totals.append({"scenario":name,"year":year,"races":len(decorated),
                           "hits":wins,"hitrate_pct":round(100*wins/len(decorated),3)
                                  if decorated else None,
                           "points":s,"avg_points_per_race":round(s/len(decorated),3)
                                  if decorated else None,
                           "invested_yen":100*s,"returned_yen":paid,
                           "net_yen":paid-100*s,
                           "roi_pct":round(paid/s,3) if s else None,
                           "payout_mean_yen":round(float(np.mean(global_payouts)),2)
                                  if global_payouts else None,
                           "payout_median_yen":round(float(np.median(global_payouts)),2)
                                  if global_payouts else None,
                           "needed_mean_payout_to_break_even_yen":round(100*s/wins,2)
                                  if wins else None,
                           "selected_band_type_counts":str([len(policy[(b,name)]) for b in BANDS])})
    manifest["overall"] = totals
    manifest["limitations"]=[
      "2025/2026 previously inspected elsewhere, not clean prospective holdouts",
      "Historical fixed AI model may have learned these years or relied on post-start data",
      "Only final settlement payouts; no cutoff-before-close market quotes",
      "Two-year in-sample combinatorial MILP can select rare large payout outliers",
      "Thresholds based on 2023 prices, not a certified predictive price floor",
      "Portfolio median constraint means >=half eligible actual wins exceed reference each training year, not all future payouts",
      "No operational prediction, purchase or fixed original HARUQ modifications"
    ]
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    (out/"portfolio_selection_and_audit.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    for filename,content in (("portfolio_frontier_by_low_year.csv",rows),
                             ("portfolio_overall_by_year.csv",totals),
                             ("all_shape_adding_decisions.csv",decisions),
                             ("portfolio_training_selection_frontier.csv",baseline)):
        with (out/filename).open("w",newline="",encoding="utf8") as f:
            if content:
                w=csv.DictWriter(f,fieldnames=list(content[0]));w.writeheader();w.writerows(content)
    print("LOW_GLOBAL_PORTFOLIO_RESULT",json.dumps({
      "cutoffs":{b:manifest["selection"][b]["reference"] for b in BANDS},
      "selected":{b:{k:{"count":v["type_count"],"types":v["types"]}
                     for k,v in manifest["selection"][b]["policies"].items()} for b in BANDS},
      "overall":totals,"band_metrics":rows,
    },ensure_ascii=False),flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--year26",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    main_run(a.input,a.year26,a.out)


if __name__=="__main__":
    main()
