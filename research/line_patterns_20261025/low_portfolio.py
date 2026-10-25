"""URA 3-2-2 LOW-conditioned portfolios: fit from 2023/24, audit 2025/26.

No base model change, no edit to fixed 2-2-2-1, no live bet approval.
A 'type' is a genuine 3-slot LINE ROLE geometry with line-equality label,
not an AI-rank position. Each type has nonoverlapping tickets by construction.
Selection reads 2023 discovery and 2024 screen ONLY.
2025/2026 labels are not opened until every LOW band's portfolio is frozen.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from db.schema import get_connection
from ura.core import low_rank_start
from research.line_patterns_20261025.geometry_322 import (
    BANDS, get_rows, load_322, structural_key, band,
)

YEAR_GROUPS=("2023","2024","2025","2026Q1","2026Q2")
SOURCE_GROUPS=("2023","2024")
SEED=20261025
MAX_LIBRARY=12
MAX_TYPES_PER_BAND=4
MIN_DISCOVERY_HITS=2
MIN_SCREEN_HITS=2


def load_2026(path):
    if not path:
        return []
    with Path(path).open(newline="",encoding="utf8") as f:
        rows=list(csv.DictReader(f))
    if not rows or len({r["race_id"] for r in rows})!=len(rows):
        raise RuntimeError("2026 labels missing or duplicated")
    for row in rows:
        if row["year"]!="2026" or int(row["n"])!=7:
            raise RuntimeError("Unexpected 2026 cohort")
    return rows


def group_for(row):
    year=int(row["year"])
    if year<2026:
        if year not in (2023,2024,2025):
            raise ValueError("unsupported year")
        return str(year)
    if year==2026 and "2026-01-01"<=row["date"]<"2026-07-01":
        return "2026Q1" if row["date"]<"2026-04-01" else "2026Q2"
    raise ValueError("2026 out of H1 scope")


def make_record(row,lookup):
    order=tuple(int(x) for x in row["ranked_bikes"].split("-"))
    result=tuple(int(x) for x in row["actual_bikes"].split("-"))
    if len(order)!=7 or len(set(order))!=7 or len(result)!=3 or len(set(result))!=3:
        raise ValueError("bad roster/actual")
    ranks={bike:i+1 for i,bike in enumerate(order)}
    if set(result)-set(order) or tuple(ranks[b] for b in result)!=tuple(int(x) for x in row["actual_ranks"].split("-")):
        raise ValueError("actual ranks do not match frozen rankings")
    tickets=Counter()
    for combo in itertools.permutations(order,3):
        if any(ranks[b]>=low_rank_start(7) for b in combo):
            tickets[structural_key(combo,lookup)]+=1
    if sum(tickets.values())!=186:
        raise ValueError("unexpected fixed URA eligible ticket set")
    actual_struct=structural_key(result,lookup)
    actual_eligible=any(ranks[b]>=5 for b in result)
    return {"race_id":row["race_id"],"date":row["date"],
            "split":group_for(row),"LOW_band":band(float(row["gap"])),
            "winner_type":actual_struct if actual_eligible else None,
            "observed_structure":actual_struct,
            "payout":int(row["payout_yen"]),"tickets":tickets}


def aggregate(records,types):
    types=set(types)
    count=0; stake=0;paid=0;hit=0;points=0;active=0
    days=defaultdict(lambda:[0,0])
    for r in records:
        count+=1
        n=sum(r["tickets"][k] for k in types)
        won=int(r["winner_type"] in types)
        if won and r["tickets"].get(r["winner_type"],0)<=0:
            raise ValueError("winning ticket absent from purchase set")
        inv=100*n
        returned=r["payout"]*won
        stake+=inv;paid+=returned;hit+=won;points+=n;active+=int(n>0)
        day=days[r["date"]];day[0]+=inv;day[1]+=returned
    return {"races":count,"active_races":active,"points":points,"hits":hit,
            "investment_yen":stake,"payout_yen":paid,"net_yen":paid-stake,
            "roi_pct":round(100*paid/stake,3) if stake else None,
            "hit_per_race_pct":round(100*hit/count,3) if count else None,
            "days":days}


def summary(rows,types,seed=SEED):
    data=aggregate(rows,types)
    days=np.asarray(list(data.pop("days").values()),float)
    ci=None
    if len(days)>1 and data["investment_yen"]>0:
        rng=np.random.default_rng(seed)
        results=[]
        for _ in range(1000):
            sample=days[rng.integers(0,len(days),len(days))].sum(axis=0)
            if sample[0]>0:results.append(100*sample[1]/sample[0])
        if results:ci=[round(float(q),2) for q in np.quantile(results,[.025,.975])]
    data["daily_cluster_roi95"]=ci
    return data


def as_years(rows):
    by=defaultdict(list)
    for r in rows:by[(r["split"],r["LOW_band"])].append(r)
    return by


def select_band(discovery,screen):
    """Predetermined two-stage discovery, no test-year labels.

    Stage 1: rank 2023 by single-type ROI given >=2 hits.
    Stage 2: among up to 12 discovered types rank by min(2023,2024 ROI),
             require 2024>=2 hits; keep up to four (zero is permitted).
    No demand that any individual type must exceed 100% ROI.
    """
    universe=sorted({k for r in discovery for k,n in r["tickets"].items()
                     if n>0 and k.rsplit("|",1)[-1] in ("12","23","13")})
    train=[]
    for shape in universe:
        m=aggregate(discovery,[shape])
        if m["hits"]>=MIN_DISCOVERY_HITS and m["investment_yen"]>0:
            train.append({"type":shape,"roi_2023":m["roi_pct"],"hits_2023":m["hits"],
                          "points_2023":m["points"]})
    train.sort(key=lambda z:(-z["roi_2023"],-z["hits_2023"],z["type"]))
    discovered=train[:MAX_LIBRARY]
    shortlisted=[]
    for row in discovered:
        s=aggregate(screen,[row["type"]])
        if s["hits"]<MIN_SCREEN_HITS or s["investment_yen"]<=0:continue
        both=min(row["roi_2023"],s["roi_pct"])
        shortlisted.append({**row,"roi_2024":s["roi_pct"],"hits_2024":s["hits"],
                            "points_2024":s["points"],"minimum_two_years_ROI":both})
    shortlisted.sort(key=lambda z:(-z["minimum_two_years_ROI"],
                                   -min(z["hits_2023"],z["hits_2024"]),z["type"]))
    return {"2023_discovered":discovered,
            "2024_screened":shortlisted,
            "selected_types":[x["type"] for x in shortlisted[:MAX_TYPES_PER_BAND]]}


def evaluate_all(label_dir,year26,out):
    past=get_rows(label_dir)
    new=load_2026(year26)
    audit=Counter()
    grouped=[]
    with get_connection() as conn:
        for row in itertools.chain(past,new):
            if int(row["n"])!=7:continue
            rid=row["race_id"]
            order=tuple(int(x) for x in row["ranked_bikes"].split("-"))
            lookup,reason=load_322(conn,rid,order)
            audit[f"{group_for(row)}_{reason}"]+=1
            if lookup is None:continue
            grouped.append(make_record(row,lookup))
    by=as_years(grouped)
    band_policies={}
    for interval in BANDS:
        # Explicitly ensure holdout years have not been loaded into policy fitting.
        trained=select_band(by[("2023",interval)],by[("2024",interval)])
        band_policies[interval]=trained
    chosen={interval:band_policies[interval]["selected_types"] for interval in BANDS}
    ledger=[];results=[];increments=[];individuals=[]
    for interval in BANDS:
        types=chosen[interval]
        for split in YEAR_GROUPS:
            cohort=by[(split,interval)]
            for k in range(1,MAX_TYPES_PER_BAND+1):
                subset=types[:k]
                result=summary(cohort,subset,seed=SEED+k)
                results.append({"year":split,"LOW_band":interval,
                                "strategy":f"first_{k}_unique_types",
                                "selected_types":";".join(subset),**result})
            for i,shape in enumerate(types):
                before=aggregate(cohort,types[:i])
                after=aggregate(cohort,types[:i+1])
                increments.append({"year":split,"LOW_band":interval,"add_index":i+1,
                                  "type":shape,"new_points":after["points"]-before["points"],
                                  "new_hits":after["hits"]-before["hits"],
                                  "new_investment":after["investment_yen"]-before["investment_yen"],
                                  "new_payout":after["payout_yen"]-before["payout_yen"],
                                  "new_net":after["net_yen"]-before["net_yen"]})
            for shape in types:
                individuals.append({"year":split,"LOW_band":interval,"type":shape,
                                    **summary(cohort,[shape])})
            for r in cohort:
                winners=set(types)
                ordered_count=[sum(r["tickets"][t] for t in types[:k])
                               for k in range(1,MAX_TYPES_PER_BAND+1)]
                ledger.append({"race_id":r["race_id"],"date":r["date"],"year":split,
                               "LOW_band":interval,"winning_geometry":r["observed_structure"],
                               "winner_underdog_eligible":int(r["winner_type"] is not None),
                               "payout_yen":r["payout"],
                               **{f"first_{k}_points":ordered_count[k-1] for k in range(1,MAX_TYPES_PER_BAND+1)},
                               **{f"first_{k}_hit":int(r["winner_type"] in types[:k])
                                  for k in range(1,MAX_TYPES_PER_BAND+1)}})
    totals=[]
    for split in YEAR_GROUPS:
        for k in range(1,MAX_TYPES_PER_BAND+1):
            cohort=[r for r in grouped if r["split"]==split]
            groups=[]
            for r in cohort:
                selection=set(chosen[r["LOW_band"]][:k])
                groups.append({"date":r["date"],"split":split,
                               "LOW_band":r["LOW_band"],"tickets":Counter({"X":sum(r["tickets"][x] for x in selection)}),
                               "winner_type":"X" if r["winner_type"] in selection else None,
                               "payout":r["payout"]})
            totals.append({"year":split,"strategy":f"first_{k}_unique_types",
                           **summary(groups,["X"],seed=SEED+k)})
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest={"experiment":"URA_322_LOW_BAND_MULTI_TYPE_PORTFOLIO",
              "data_years":{year:sum(r["split"]==year for r in grouped) for year in YEAR_GROUPS},
              "time_cohorts":["2023=discovery","2024=type screen","2025=not used for selection","2026Q1/Q2=evaluation only"],
              "frozen_original_haruq_unchanged":True,"frozen_original_ura_unchanged":True,
              "frozen_2221_unchanged":True,"research_only":True,"purchase_approved":False,
              "no_ai_rank_head_fixing":True,
              "LOW_intervals":list(BANDS),
              "allowed_decision_relations":["12","23","13"],
              "library_pool_per_band":MAX_LIBRARY,
              "max_portfolio_types_per_band":MAX_TYPES_PER_BAND,
              "minimum_hits_for_research":MIN_DISCOVERY_HITS,
              "selection_policy":"2023 ROI top12 among >=2 hits; 2024 min(two-year ROIs) among >=2 screen hits; take first 4, no requirement of individual profitability",
              "band_policies":band_policies,
              "audit":dict(audit),
              "summary_all_bands":totals,
              "warnings":[
                  "All types are disjoint outcome cells; adding individually losing types cannot by itself raise ROI above 100 in the same cohort",
                  "No actual pre-close 3連単 odds, payout is archived final settlement",
                  "Frozen base model historic features have not passed prestart as-of audit",
                  "Original fixed model may have learned parts of backtest history",
                  "2025 and 2026 have been inspected in related prior studies, so not a virgin forward validation",
                  "Small number of wins and dozens of screened shapes create multiple comparison risk",
                  "2026Q2 breakdown is reported unchanged; no hindsight filters permitted",
                  "This experiment generates candidate libraries only, not purchase or Web rules"
              ]}
    (out/"model_research_only.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    for path,rows in [("portfolio_by_band_year.csv",results),("portfolio_all_bands.csv",totals),
                      ("incremental_contribution.csv",increments),("selected_types_individual.csv",individuals),
                      ("per_race_ledger.csv",ledger)]:
        if not rows:continue
        with (out/path).open("w",newline="",encoding="utf8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]))
            w.writeheader();w.writerows(rows)
    print("LOW_MULTI_PORTFOLIO_RESULT",json.dumps({
        "selected":chosen,
        "annual_totals":[{"year":x["year"],"strategy":x["strategy"],
                          "races":x["races"],"hits":x["hits"],"points":x["points"],
                          "roi_pct":x["roi_pct"],"net_yen":x["net_yen"]}
                         for x in totals],
        "audit":dict(audit)
    },ensure_ascii=False),flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--year26",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    evaluate_all(a.input,a.year26,a.out)


if __name__=="__main__":
    main()
