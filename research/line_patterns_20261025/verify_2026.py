"""Frozen 2026H1 evaluation of THREE predeclared same-line-pair families.

No fitting, no optimization or adaptive 2026 threshold. Family comprises all
structurally conforming underdog-qualified 3連単 tickets. Exact role probes are
predeclared from 2023/24 patterns; 2026 only evaluates outcomes.
"""
from __future__ import annotations
import argparse, csv, itertools, json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from db.schema import get_connection
from ura.core import low_rank_start
from research.line_patterns_20261025.geometry_322 import load_322, structural_key

FAMILIES=("12","23","13")
PROBES={
    "12":"2H-2B-3B|12",
    "23":"2H-3H-3B|23",
    "13":"3B-2H-3T|13",
}
SEED=20261025


def summarize(rows,seed=SEED):
    n=len(rows)
    tickets=sum(x["points"] for x in rows)
    hits=sum(x["hit"] for x in rows)
    paid=sum(x["payout"]*x["hit"] for x in rows)
    stake=100*tickets
    days=defaultdict(lambda:[0,0])
    for x in rows:
        d=days[x["date"]];d[0]+=100*x["points"];d[1]+=x["payout"]*x["hit"]
    ci=None
    if stake and len(days)>=2:
        a=np.asarray(list(days.values()),float)
        rng=np.random.default_rng(seed)
        values=[]
        for _ in range(2000):
            s=a[rng.integers(len(a),size=len(a))].sum(axis=0)
            if s[0]:values.append(100*s[1]/s[0])
        if values:ci=[round(float(v),2) for v in np.quantile(values,[.025,.975])]
    return {"races":n,"tickets":tickets,"hits":hits,"investment":stake,
            "payout":paid,"net":paid-stake,
            "roi_pct":round(100*paid/stake,3) if stake else None,
            "hit_per_race_pct":round(100*hits/n,3) if n else None,
            "day_cluster_bootstrap_roi95":ci,
            "number_of_active_races":sum(x["points"]>0 for x in rows)}


def read_csv(path):
    with Path(path).open(newline="",encoding="utf8") as f:rows=list(csv.DictReader(f))
    if not rows or len({x["race_id"] for x in rows})!=len(rows):
        raise RuntimeError("Missing or duplicated 2026H1 input labels")
    if any(x["year"]!="2026" or int(x["n"])!=7 or not "2026-01-01"<=x["date"]<"2026-07-01" for x in rows):
        raise RuntimeError("Wrong year/period/field size")
    return rows


def evaluate(input_csv,outdir):
    source=read_csv(input_csv)
    counters=Counter()
    study=defaultdict(list)
    descript=defaultdict(list)
    day_log=[]
    with get_connection() as conn:
        for x in source:
            rid=x["race_id"]
            order=tuple(int(q) for q in x["ranked_bikes"].split("-"))
            if len(order)!=7 or len(set(order))!=7:
                raise ValueError("Bad frozen model source ranks")
            lookup,status=load_322(conn,rid,order)
            counters[status]+=1
            if lookup is None:continue
            rankings={bike:i+1 for i,bike in enumerate(order)}
            actual=tuple(int(q) for q in x["actual_bikes"].split("-"))
            if len(actual)!=3 or len(set(actual))!=3 or tuple(rankings[b] for b in actual)!=tuple(map(int,x["actual_ranks"].split("-"))):
                raise ValueError("Actual differs from frozen ranking")
            winner=structural_key(actual,lookup)
            relation=winner.split("|")[-1]
            if relation in FAMILIES:
                descript[relation].append(int(x["payout_yen"]))
            tickets=defaultdict(int)
            for p in itertools.permutations(order,3):
                if any(rankings[b]>=low_rank_start(7) for b in p):
                    kind=structural_key(p,lookup)
                    tickets[kind]+=1
            if sum(tickets.values())!=186:raise AssertionError("Eligible ticket universe changed")
            for family in FAMILIES:
                suffix="|"+family
                points=sum(n for k,n in tickets.items() if k.endswith(suffix))
                if points<0:raise AssertionError("Negative points")
                hit=int(relation==family and any(rankings[b]>=5 for b in actual))
                study["family_"+family].append({
                    "date":x["date"],"race_id":rid,"points":points,"hit":hit,
                    "payout":int(x["payout_yen"])})
            for family,probe in PROBES.items():
                hit=int(winner==probe and any(rankings[b]>=5 for b in actual))
                study["probe_"+family].append({
                    "date":x["date"],"race_id":rid,"points":tickets.get(probe,0),
                    "hit":hit,"payout":int(x["payout_yen"])})
            day_log.append({"race_id":rid,"date":x["date"],"geometry":winner,
                            "relation":relation,
                            "actual_yen":int(x["payout_yen"]),
                            "contains_underdog":int(any(rankings[b]>=5 for b in actual)),
                            "LOW_gap":float(x["gap"])})
    if counters["ok"] == 0:raise RuntimeError("No complete 2026H1 3-2-2")
    stats={k:summarize(v) for k,v in sorted(study.items())}
    for family in FAMILIES:
        observed=descript[family]
        stats["family_"+family]["observed_winning_races_all"]=len(observed)
        stats["family_"+family]["observed_median_payout_yen"]=int(np.median(observed)) if observed else None
    output={"system":"URA_322_FROZEN_FAMILY_VALIDATION_2026H1",
            "scope":"2026-01-01 to 2026-06-30, 3-2-2, 7-bike, original fixed HaruQ LOW<1",
            "selections_locked_before_2026_label_scan":True,
            "study_was_previously_explored_with_2023_to_2025":True,
            "not_deployable_without_prestart_feature_and_odds_audit":True,
            "frozen_2_2_2_1_untouched":True,
            "research_only":True,"purchase_approved":False,
            "source_2026_LOW_seven_car":len(source),
            "lineup_audit":dict(counters),
            "selected_probes":PROBES,
            "family_stats":stats,
            "caveats":[
                "Historical frozen model could have trained on 2026 results, not necessarily genuinely out of sample",
                "Source date for features from historical entry/result table not audited as-of prestart",
                "Final settlement payout is known only after race; no preclose odds used",
                "2026 used for confirmation only; it was not used to choose any geometry rule",
                "All three geometry families buy many tickets; overall ROI includes large transaction/coverage cost",
                "Exact role probes chosen from earlier research, not guaranteed independent discovery",
                "Bootstrap is uncertainty for observed period, not future positive EV confirmation"
            ]}
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    (out/"summary.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    with (out/"realized_2026_geometry.csv").open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(day_log[0]));w.writeheader();w.writerows(day_log)
    print("URA_2026H1_FROZEN_322",json.dumps(output,ensure_ascii=False),flush=True)
    return output


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    evaluate(a.input,a.out)
