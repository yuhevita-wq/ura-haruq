"""Attach original race-entry score and age, never outcomes, to frozen 2025 study exposures.

Inputs: exact 2025 underdog 7car LOW research cohort from GitHub run 40178812827;
original signed March/Oct historical archive, no new race filter or ranking.
Class / rider score / age auditing. Do not substitute result-table data.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from collections import Counter
from pathlib import Path
from db.schema import get_connection

EXPECTED_ROWS=40851
EXPECTED_RACES=13617

def create(source,out):
    source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data=source.read_bytes()
    with source.open(encoding="utf-8-sig",newline="") as f:
        exposures=list(csv.DictReader(f))
    if len(exposures)!=EXPECTED_ROWS or len(set(e["race_id"] for e in exposures))!=EXPECTED_RACES:
        raise RuntimeError("Original unaltered 2025 lower-rank cohort required")
    needed={(e["race_id"],e["rider_id"]) for e in exposures}
    if len(needed)!=len(exposures):raise RuntimeError("Duplicated rider/race in frozen exposure cohort")
    rows={};audit=Counter();raw_scores={}
    with get_connection() as conn:
        # Use all score records to measure within-race relative score, not only
        # previously chosen LOWER-rank riders.
        rs=conn.execute("""
          SELECT e.race_id,e.rider_id,e.avg_competition_score,e.class,
                 e.bike_number,r.birth_year,r.gender
          FROM entries e LEFT JOIN riders r ON r.rider_id=e.rider_id
          JOIN races t ON t.race_id=e.race_id
          WHERE t.date>='2025-01-01' AND t.date<'2026-01-01'
        """)
        for x in rs:
            rid=str(x["race_id"]);pid=str(x["rider_id"])
            try:
                score=float(x["avg_competition_score"]) if x["avg_competition_score"] is not None else None
            except (TypeError,ValueError):
                score=None
            if score is not None and not (20<=score<=160):score=None
            if score is not None:raw_scores.setdefault(rid,[]).append(score)
            if (rid,pid) in needed:
                if (rid,pid) in rows:raise RuntimeError("Duplicate join target")
                rows[(rid,pid)]={"prestart_competition_score":score,
                    "birth_year_from_master":x["birth_year"],
                    "entry_class":x["class"],"master_gender":x["gender"]}
        audit["cohort_rows"]=len(exposures)
        audit["found_matching_original_entries"]=len(rows)
        audit["matched_distinct_races"]=len(set(x[0] for x in rows))
    outrows=[]
    for e in exposures:
        key=(e["race_id"],e["rider_id"])
        x=rows.get(key,{})
        score=x.get("prestart_competition_score")
        age=None
        try:
            yr=int(x.get("birth_year_from_master"))
            age=2025-yr
            if age<16 or age>90:age=None
        except (ValueError,TypeError):
            pass
        if age is not None:audit["birth_year_valid"]+=1
        else:audit["birth_year_missing"]+=1
        if score is not None:audit["competition_score_valid"]+=1
        else:audit["competition_score_missing"]+=1
        peers=raw_scores.get(e["race_id"],[])
        mean=sum(peers)/len(peers) if peers else None
        delta=score-mean if score is not None and mean is not None else None
        outrows.append({**e,"age_calendar_year":age,
                        "age_source":"master_birth_year_calendar_only" if age is not None else "missing",
                        "prestart_competition_score":score,
                        "score_minus_race_mean":round(delta,6) if delta is not None else None,
                        "field_riders_with_entry_score":len(peers),
                        "entry_class_check":x.get("entry_class",""),
                        "master_gender_check":x.get("master_gender","")})
    audit["source_csv_sha256"]=hashlib.sha256(data).hexdigest()
    audit["research_only"]=True
    audit["model_or_formations_changed"]=False
    audit["source_birth_year_approximate_age"]=True
    audit["score_is_original_entry"]=True
    audit["observed_source_2025_only"]=True
    with (out/"2025_underdog_WITH_AGE_SCORE.csv").open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(outrows[0]));w.writeheader();w.writerows(outrows)
    (out/"age_score_join_audit.json").write_text(json.dumps(dict(audit),ensure_ascii=False,indent=2)+"\n")
    print("AGE_SCORE_JOIN_AUDIT",json.dumps(dict(audit),ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True);p.add_argument("--out",required=True)
    a=p.parse_args();create(a.source,a.out)
