"""Research-only 2025 year-exclusive LOW underdog rider admission leaderboard.

Population: archived immutable 2025 V3 CSV, 7-car LOW (gap<1),
every rank-5/6/7 entry in each matched race. All 7-car versus 3-2-2
reported separately, with LOW segment splits. No 2023, 2024, 2026
outcome enters numerator/denominator or threshold/sorting.

Numbers measure ACTUAL 2025 appearances and placements conditional on
original historical frozen model ranking (not prospective predictions).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from db.schema import get_connection
from research.line_patterns_20261025.geometry_322 import BANDS, band, load_322

DATA_YEAR = "2025"
EXPECTED_SOURCE_COUNT = 6507
MIN_RECORDS_FOR_PRIMARY = 10
MIN_RECORDS_FOR_SECONDARY = 5


def wilson_lower(k,n,z=1.959963984540054):
    if n == 0 or k == 0: return 0.0
    p=k/n
    den=1+z*z/n
    return max(0.0,((p+z*z/(2*n))-z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den)


def load_source(path):
    raw=Path(path).read_bytes()
    sha=hashlib.sha256(raw).hexdigest()
    with Path(path).open(newline="",encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=EXPECTED_SOURCE_COUNT:
        raise RuntimeError(f"Immutable 2025 sample changed: {len(rows)}")
    if len({r["race_id"] for r in rows})!=len(rows):
        raise RuntimeError("Duplicate race id")
    for r in rows:
        if not r["date"].startswith(DATA_YEAR+"-"):
            raise RuntimeError("Found other year in immutable 2025 sample")
    return rows,sha


def read_one_2025_race(conn,row,audit):
    audit["2025_all_source_labels"] +=1
    n=int(row["n"])
    if n!=7:
        audit["2025_not_7car"]+=1
        return None
    audit["2025_7car_source_labels"]+=1
    gap=float(row["gap"])
    if not math.isfinite(gap) or gap<0 or gap>=1:
        audit["2025_not_low"]+=1
        return None
    ranked=[int(v) for v in row["ranked_bikes"].split("-")]
    finished=[int(v) for v in row["actual_bikes"].split("-")]
    actual_ranks=[int(v) for v in row["actual_ranks"].split("-")]
    if (len(ranked)!=7 or len(set(ranked))!=7 or
        len(finished)!=3 or len(set(finished))!=3 or
        any(v not in ranked for v in finished) or
        actual_ranks!=[ranked.index(v)+1 for v in finished]):
        raise ValueError("Frozen AI rank / real outcome differs for "+row["race_id"])
    all7=conn.execute(
        "SELECT e.rider_id,e.rider_name,e.bike_number,e.class,e.line_group,"
        "e.line_role,COALESCE(r.name,e.rider_name) AS official_name,"
        "r.prefecture "
        "FROM entries e LEFT JOIN riders r ON r.rider_id=e.rider_id "
        "WHERE e.race_id=? ORDER BY e.bike_number",(row["race_id"],)
    ).fetchall()
    if len(all7)!=7 or len({x["rider_id"] for x in all7})!=7 or (
       {int(x["bike_number"]) for x in all7}!=set(ranked)):
        audit["2025_missing_or_disagreeing_roster"]+=1
        return None
    lookup,reason=load_322(conn,row["race_id"],ranked)
    audit["2025_line_"+reason]+=1
    out={"race_id":row["race_id"],"date":row["date"],"gap":gap,
         "LOW_band":band(gap),
         "is_322":lookup is not None,
         "ranks":ranked,"finish":finished,"entries":all7,"lookup":lookup}
    audit["2025_7car_low_matched"]+=1
    if lookup is not None:audit["2025_322_low_matched"]+=1
    return out


def select_name(entry_name, master_name):
    missing={"", "unknown", "none", "null", "不明", "-", "nan"}
    a=str(entry_name or "").strip()
    b=str(master_name or "").strip()
    if a.lower() not in missing:return a
    if b.lower() not in missing:return b
    return a or b


def make_exposures(races):
    exposures=[]
    for r in races:
        ranked=r["ranks"]
        podium={bike:i for i,bike in enumerate(r["finish"],start=1)}
        for e in r["entries"]:
            bike=int(e["bike_number"])
            rank=ranked.index(bike)+1
            # Restrict primary exposure to actually labelled 5th, 6th, 7th
            # according to IMMUTABLE original rank, not performance/points.
            if rank<5:continue
            pos=podium.get(bike)
            if r["lookup"] is not None:
                position_in_line=r["lookup"][bike][1]
            else:
                position_in_line=str(e["line_role"] or "")
            exposures.append({
                "race_id":r["race_id"],"date":r["date"],"half":"H1" if r["date"]<"2025-07-01" else "H2",
                "LOW_band":r["LOW_band"],"is_322":int(r["is_322"]),
                "rider_id":str(e["rider_id"]),"name":select_name(e["rider_name"],e["official_name"]),
                "prefecture":str(e["prefecture"] or ""),
                "bike_number":bike,"ura_rank":rank,"line_role":position_in_line,
                "finish_position":pos or "",
                "top3":int(pos is not None),
                "first":int(pos==1),"second":int(pos==2),"third":int(pos==3),
            })
    if len(exposures)!=3*len(races):
        raise RuntimeError("Every 7car LOW race must generate three low-rank exposures")
    return exposures


def stats(rs):
    n=len(rs);hit=sum(r["top3"] for r in rs)
    first=sum(r["first"] for r in rs)
    second=sum(r["second"] for r in rs)
    third=sum(r["third"] for r in rs)
    if n==0:raise ValueError("Empty exposures")
    if first+second+third!=hit:raise RuntimeError("Finish count mismatch")
    return {"low_rank_starts":n,"top3_count":hit,
            "top3_rate_pct":round(100*hit/n,3),
            "top3_wilson95_lower_pct":round(100*wilson_lower(hit,n),3),
            "first_count":first,"second_count":second,"third_count":third,
            "first_pct":round(100*first/n,3),
            "second_pct":round(100*second/n,3),
            "third_pct":round(100*third/n,3),
            "rank5_starts":sum(r["ura_rank"]==5 for r in rs),
            "rank6_starts":sum(r["ura_rank"]==6 for r in rs),
            "rank7_starts":sum(r["ura_rank"]==7 for r in rs),
            "rank5_top3":sum(r["ura_rank"]==5 and r["top3"] for r in rs),
            "rank6_top3":sum(r["ura_rank"]==6 and r["top3"] for r in rs),
            "rank7_top3":sum(r["ura_rank"]==7 and r["top3"] for r in rs),
            "2025_H1_starts":sum(r["half"]=="H1" for r in rs),
            "2025_H1_top3":sum(r["half"]=="H1" and r["top3"] for r in rs),
            "2025_H2_starts":sum(r["half"]=="H2" for r in rs),
            "2025_H2_top3":sum(r["half"]=="H2" and r["top3"] for r in rs),
            "unique_months":len(set(r["date"][:7] for r in rs)),
            "low_bands_seen":",".join(sorted(set(r["LOW_band"] for r in rs))),
            "line_roles_seen":",".join(sorted(set(r["line_role"] for r in rs)))}


def per_rider(exposures,mode):
    grouped=defaultdict(list)
    for e in exposures:grouped[e["rider_id"]].append(e)
    out=[]
    for rid,rs in grouped.items():
        out.append({"rider_id":rid,"rider":rs[0]["name"],
                    "prefecture":rs[0]["prefecture"],"sample_scope":mode,
                    **stats(rs)})
    out.sort(key=lambda x:(
        -x["top3_wilson95_lower_pct"],
        -x["low_rank_starts"],
        -x["top3_count"],x["rider_id"]))
    for i,x in enumerate(out,1):x["all_rider_rank"]=i
    qualified=[r for r in out if r["low_rank_starts"]>=MIN_RECORDS_FOR_PRIMARY]
    qualified.sort(key=lambda x:(-x["top3_wilson95_lower_pct"],
                                 -x["top3_rate_pct"],-x["low_rank_starts"],x["rider_id"]))
    for i,x in enumerate(qualified,1):x["10plus_rank"]=i
    return out


def split_by_band(exposures,only_322=False):
    out=[]
    for scope,filtered in (("all_7car_LOW",exposures),("322_7car_LOW",[x for x in exposures if x["is_322"]])):
        if only_322 and scope!="322_7car_LOW":continue
        for b in BANDS:
            rs=[r for r in filtered if r["LOW_band"]==b]
            for row in per_rider(rs,scope):
                out.append({"LOW_band":b,**row})
    return out


def write_csv(path,records):
    if not records:raise RuntimeError("No records for "+str(path))
    columns=list(dict.fromkeys(k for rec in records for k in rec))
    with path.open("w",encoding="utf8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def report(input_file,out):
    rows,sha=load_source(input_file)
    audit=Counter();races=[]
    with get_connection() as conn:
        for row in rows:
            r=read_one_2025_race(conn,row,audit)
            if r is not None:races.append(r)
    expos=make_exposures(races)
    all_people=per_rider(expos,"all_7car_LOW")
    only322=per_rider([x for x in expos if x["is_322"]],"322_7car_LOW")
    leaderboard_all=[r for r in all_people if r["low_rank_starts"]>=MIN_RECORDS_FOR_PRIMARY]
    leaderboard_322=[r for r in only322 if r["low_rank_starts"]>=MIN_RECORDS_FOR_SECONDARY]
    summary={}
    for name,ex in [("all_7car_LOW",expos),
                    ("322_7car_LOW",[x for x in expos if x["is_322"]])]:
        n=len(ex)
        summary[name]={
            "races":n//3,"rider_low_rank_starts":n,
            "low_rank_top3":sum(x["top3"] for x in ex),
            "low_rank_top3_rate_pct":round(100*sum(x["top3"] for x in ex)/n,3) if n else None,
            "finish_1st":sum(x["first"] for x in ex),
            "finish_2nd":sum(x["second"] for x in ex),
            "finish_3rd":sum(x["third"] for x in ex),
            "unique_low_rank_riders":len(set(x["rider_id"] for x in ex)),
            "riders_5plus":sum(r["low_rank_starts"]>=5 for r in (all_people if name=="all_7car_LOW" else only322)),
            "riders_10plus":sum(r["low_rank_starts"]>=10 for r in (all_people if name=="all_7car_LOW" else only322)),
            "with_any_eligible_podium_races":len(set(x["race_id"] for x in ex if x["top3"]))
        }
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    write_csv(out/"all_2025_7car_LOW_players.csv",all_people)
    write_csv(out/"2025_7car_LOW_10plus_ranked_leaderboard.csv",leaderboard_all)
    write_csv(out/"2025_322_LOW_all_players.csv",only322)
    write_csv(out/"2025_322_LOW_5plus_ranked_leaderboard.csv",leaderboard_322)
    write_csv(out/"2025_player_low_rank_by_band.csv",split_by_band(expos))
    write_csv(out/"all_2025_low_rank_race_exposures.csv",expos)
    result={
        "experiment":"2025_ONLY_LOW_RANK_RIDER_5_6_7_EXPOSURE_AND_PODIUM",
        "research_only":True,"purchase_approved":False,
        "frozen_original_models_modified":False,"fixed_2221_modified":False,
        "previous_4_20_portfolios_modified":False,"source_year_only":2025,
        "source_file_sha256":sha,"2025_source_labels":len(rows),
        "2025_cohort_audit":dict(audit),"cohort_summary":summary,
        "ranking":"Wilson 95% lower bound of observed top3 share, NOT raw 1/1 ratio; show exposure counts",
        "primary_threshold_all_7car":MIN_RECORDS_FOR_PRIMARY,
        "primary_threshold_322":MIN_RECORDS_FOR_SECONDARY,
        "top_20_all_races":leaderboard_all[:20],
        "top_20_322":leaderboard_322[:20],
        "limitations":[
           "2025 outcomes are used to DESCRIBE 2025 only, no 2023-24 or 2026 result mixing",
           "Underlying fixed model's training includes some 2025 race results; historical scores are not certified out-of-fold",
           "Association with original rank 5-7 is not a future-calibrated probability",
           "2025 race coverage is limited to archived complete V3 7car LOW subset, not all 2025 keirin starts",
           "3-2-2 classification relies on published archived entries; missing or changed line groups excluded",
           "All riders listed unfiltered; minimum 10/5 starts is a DISPLAY confidence threshold not a trading signal",
           "Subgroup search and player names selected after results; cannot certify expected profit or betting advantage"
        ]
    }
    (out/"2025_only_rider_rank_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("2025_RIDER_LOW_RESULT",json.dumps({
        "audit":result["2025_cohort_audit"],
        "summary":summary,
        "top_20_all_10plus":leaderboard_all[:20],
        "top_20_322_5plus":leaderboard_322[:20]
    },ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    x=p.parse_args()
    report(x.input,x.out)


if __name__=="__main__":
    main()
