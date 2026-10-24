"""Empirical 3-2-2 geometry research. Outcome-first, not ranked-first.

Different two-member lines are exchangeable: 3H = head of 3-line,
3B = second, 3T = third, 2H/2B = head/second of either 2-line;
explicit line-equality flags distinguish identical role-code sequences.
Original URA LOW and at least one lower-ranked rider remain fixed.
NO models or operational policy are modified.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np

from db.schema import get_connection
from ura.core import low_rank_start

YEARS=(2023,2024,2025)
BANDS=("0.00-0.25","0.25-0.50","0.50-0.75","0.75-1.00")
SEED=20261025


def role_code(value):
    x=str(value or "")
    if "3番手" in x or "三番手" in x: return "T"
    if "番手" in x: return "B"
    if "自力" in x or "逃" in x or "先頭" in x: return "H"
    return None


def load_322(conn, rid, expected):
    rows=conn.execute(
        "SELECT bike_number,line_group,line_role FROM entries WHERE race_id=?",
        (rid,)).fetchall()
    if len(rows)!=7:return None,"no_exact_7_entries"
    numbers=[r["bike_number"] for r in rows]
    if any(x is None for x in numbers) or set(map(int,numbers))!=set(expected) or len(set(numbers))!=7:
        return None,"roster_disagrees"
    if any(r["line_group"] is None or not str(r["line_group"]).strip() for r in rows):
        return None,"missing_line"
    by_line=defaultdict(list)
    for r in rows:by_line[str(r["line_group"]).strip()].append(r)
    if sorted(map(len,by_line.values()),reverse=True)!=[3,2,2]:
        return None,"not_322"
    lookup={}
    for group,items in by_line.items():
        size=len(items)
        actual_roles=[role_code(v["line_role"]) for v in items]
        required={"H","B","T"} if size==3 else {"H","B"}
        if len(set(actual_roles))!=size or set(actual_roles)!=required:
            return None,"role_not_unique"
        for item,position in zip(items,actual_roles):
            lookup[int(item["bike_number"])]=(group, str(size)+position)
    return lookup,"ok"


def same_line_mark(groups):
    a,b,c=groups
    if a==b==c:return "123"
    if a==b:return "12"
    if b==c:return "23"
    if a==c:return "13"
    return "X"


def structural_key(outcome,lookup):
    members=[lookup[int(b)] for b in outcome]
    codes=[r[1] for r in members]
    return "-".join(codes)+"|"+same_line_mark([x[0] for x in members])


def band(gap):
    assert 0<=gap<1.0
    return BANDS[min(3,int(gap*4))]


def get_rows(path):
    out=[]
    for yr in YEARS:
        with (Path(path)/f"year_{yr}.csv").open(newline="",encoding="utf8") as f:
            group=list(csv.DictReader(f))
        if len(group)!={2023:6565,2024:6468,2025:6507}[yr]:
            raise ValueError(f"Frozen yearly sample changed {yr}: {len(group)}")
        for row in group:
            if int(row["n"])!=7: continue
            row["year"]=yr
            out.append(row)
    return out


def roundpct(a,b):
    return round(100*a/b,3) if b else None


def aggregate(stats):
    grouped=defaultdict(lambda:{"races":0,"tickets":0,"hits":0,"investment_yen":0,
                                "paid_yen":0,"payouts":[],"days":defaultdict(lambda:[0,0])})
    for record in stats:
        key=(record["year"],record["type"])
        s=grouped[key]
        s["races"]+=1
        s["tickets"]+=record["points"]
        s["hits"]+=record["hit"]
        s["investment_yen"]+=record["stake"]
        s["paid_yen"]+=record["return"]
        if record["hit"]:s["payouts"].append(record["return"])
        d=s["days"][record["date"]]
        d[0]+=record["stake"];d[1]+=record["return"]
    output=[]
    for (yr,kind),s in sorted(grouped.items()):
        values=np.asarray(list(s["days"].values()),float)
        ci=None
        if s["investment_yen"] and len(values)>1:
            rng=np.random.default_rng(SEED+yr)
            ratios=[]
            for _ in range(750):
                p=values[rng.integers(len(values),size=len(values))].sum(axis=0)
                if p[0]:ratios.append(100*p[1]/p[0])
            if ratios:
                ci=[round(float(q),2) for q in np.quantile(ratios,[.025,.975])]
        output.append({"year":yr,"type":kind,"races":s["races"],
                       "tickets":s["tickets"],"hits":s["hits"],
                       "investment_yen":s["investment_yen"],
                       "paid_yen":s["paid_yen"],
                       "net_yen":s["paid_yen"]-s["investment_yen"],
                       "roi_pct":roundpct(s["paid_yen"],s["investment_yen"]),
                       "median_hit_payout":round(float(np.median(s["payouts"])),2) if s["payouts"] else None,
                       "ci95_days":ci})
    return output


def write_csv(path, rows):
    if not rows:return
    with path.open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def analyze(input_dir,out_dir):
    rows=get_rows(input_dir)
    audit=Counter()
    studied=[]
    ledger=[]
    low_roles=Counter()
    with get_connection() as conn:
        for row in rows:
            rid=row["race_id"]
            ranked=[int(v) for v in row["ranked_bikes"].split("-")]
            if len(set(ranked))!=7:raise ValueError(f"Bad rank roster {rid}")
            lookup,reason=load_322(conn,rid,ranked)
            audit[reason]+=1
            if lookup is None:continue
            real=tuple(int(v) for v in row["actual_bikes"].split("-"))
            ranks={bike:i+1 for i,bike in enumerate(ranked)}
            if len(real)!=3 or len(set(real))!=3 or any(v not in lookup for v in real):
                raise ValueError(f"Bad finish {rid}")
            if tuple(ranks[v] for v in real)!=tuple(map(int,row["actual_ranks"].split("-"))):
                raise ValueError(f"Winner vs ranked inconsistency {rid}")
            outcome=structural_key(real,lookup)
            cut=low_rank_start(7)
            low_fin=[i+1 for i,b in enumerate(real) if ranks[b]>=cut]
            points=Counter()
            for ticket in itertools.permutations(ranked,3):
                if any(ranks[b]>=cut for b in ticket):
                    points[structural_key(ticket,lookup)]+=1
            assert sum(points.values())==186
            eligible=bool(low_fin)
            if eligible and outcome not in points:
                raise ValueError("Actual eligible geometry lost")
            day=row["date"];year=int(row["year"])
            payout=int(row["payout_yen"])
            low_roles[tuple(low_fin)]+=1
            studied.append({
                "race_id":rid,"date":day,"year":year,"LOW_band":band(float(row["gap"])),
                "geometry":outcome,"payout_yen":payout,
                "low_finish_positions":",".join(map(str,low_fin)),
                "lower_in_top3":int(eligible),"winning_ranks":row["actual_ranks"],
                "ticket_eligibility_count":sum(points.values())
            })
            for kind,npts in points.items():
                won=int(eligible and outcome==kind)
                ledger.append({"race_id":rid,"date":day,"year":year,"type":kind,
                               "points":npts,"hit":won,"stake":100*npts,
                               "return":payout*won})
    summary=aggregate(ledger)
    possible=sorted(set(r["type"] for r in summary))
    yearmap={(z["year"],z["type"]):z for z in summary}
    # The following is merely a stable-pattern screening audit: NO tickets
    # selected for purchase, and 2025 is never fed into eligibility checks.
    researched=[]
    for kind in possible:
        a=yearmap.get((2023,kind))
        b=yearmap.get((2024,kind))
        c=yearmap.get((2025,kind))
        screened=(a is not None and b is not None and
                  a["hits"]>=5 and b["hits"]>=5 and
                  a["roi_pct"] is not None and a["roi_pct"]>=100 and
                  b["roi_pct"] is not None and b["roi_pct"]>=100)
        researched.append({
            "type":kind,"pre2025_stable_signal_research_only":bool(screened),
            "year_2023":a,"year_2024":b,"year_2025":c
        })
    # Purely descriptive 2025 observed winning geometry, irrespective of
    # strategy or odds. Captures outcomes with NO lower-ranked riders too.
    raw_geometry=defaultdict(list)
    for r in studied:raw_geometry[(r["year"],r["geometry"])].append(r["payout_yen"])
    occurrence=[{"year":yr,"geometry":g,"results":len(v),
                 "median_3rentan_yen":int(np.median(v)),
                 "mean_3rentan_yen":round(float(np.mean(v)),2)}
                for (yr,g),v in sorted(raw_geometry.items())]
    decorated=sorted(studied,key=lambda x:(x["year"],x["race_id"]))
    study={"name":"URA_322_LINE_POSITION_OUTCOME_FIRST",
           "original_frozen_2221_modified":False,
           "rank_based_heads_in_geometry":False,
           "training_scope":"2023 descriptive discovery, 2024 independent descriptive confirmation; 2025 further independent observation",
           "original_ticket_eligibility":"LOW<1; each trifecta has at least one 7-car lower-URA-rank 5-7 rider",
           "candidate_count":len(possible),
           "processed_by_year":{str(yr):sum(r["year"]==yr for r in studied) for yr in YEARS},
           "data_rejection":dict(audit),
           "observed_geometry":len(occurrence),
           "signals_exploratory":sum(r["pre2025_stable_signal_research_only"] for r in researched),
           "top_2025_by_actual_frequency":sorted(
                (r for r in occurrence if r["year"]==2025),
                key=lambda r:(-r["results"],r["geometry"]))[:30],
           "research_only":True,"purchase_approved":False,
           "limitations":["Full historical results, but original HaruQ model may have learned 2023-25 outcomes",
                          "Original entries are archived and not timestamped as a true pre-start snapshot",
                          "Final odds are used only for realized winning payout, not tradable pre-close odds",
                          "Pre-2025 search selects among many geometries and is exploratory, not significance-controlled",
                          "Role descriptions come from archived line_role; invalid role sets excluded, never guessed",
                          "Both 2-rider lines are symmetric: code 2H/2B contains no AI strength rank, and line equality is explicit",
                          "This has not discovered a deployable profitable betting strategy"]}
    root=Path(out_dir);root.mkdir(parents=True,exist_ok=True)
    (root/"summary.json").write_text(json.dumps(study,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    write_csv(root/"observed_winning_geometry.csv",occurrence)
    write_csv(root/"actual_322_races.csv",decorated)
    write_csv(root/"complete_structural_roi_by_year.csv",summary)
    write_csv(root/"structural_2023_2024_screen_2025_evaluate.csv",[
        {"type":v["type"],"screened_2023_2024":int(v["pre2025_stable_signal_research_only"]),
         **{f"{yr}_{k}":(v[f"year_{yr}"] or {}).get(k)
            for yr in YEARS for k in ("hits","tickets","paid_yen","roi_pct","median_hit_payout")}}
        for v in researched])
    print("URA_322_OUTCOME_GEOMETRY",json.dumps({
        "audited":study["processed_by_year"],
        "refused":study["data_rejection"],
        "types":study["candidate_count"],
        "research_signals":study["signals_exploratory"],
        "top_2025":study["top_2025_by_actual_frequency"][:14],
        "signals_2025":[{"type":v["type"],
                         "hits_2025":(v["year_2025"] or {}).get("hits"),
                         "roi_2025":(v["year_2025"] or {}).get("roi_pct")}
                        for v in researched if v["pre2025_stable_signal_research_only"]]
    },ensure_ascii=False),flush=True)
    return study


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    x=p.parse_args()
    analyze(x.input,x.out)
