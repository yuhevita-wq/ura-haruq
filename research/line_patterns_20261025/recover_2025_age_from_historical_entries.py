"""Supplement missing rider age from separate year-2025 KDreams archived entries.

Join to EXACT prior frozen 40,851 row cohort on YYYYMMDD+venue+race and
bike number, plus normalized full racer name. NEVER infer from another
rider or another race. Age must be plausible, source year=2025.
Keep all rows, audit every null/collision, no outcome or bet filtering.
"""
from __future__ import annotations
import argparse,csv,json,hashlib,unicodedata
from collections import Counter
from pathlib import Path
from ura.roi_holdout import odds_race_id

EXPECTED=40851
EXPECTED_RACES=13617

def norm(x):
    return "".join(c for c in unicodedata.normalize("NFKC",str(x or "")) if not c.isspace())

def extract(paths):
    source_by_race_bike={}
    conflicts=set()
    stats=Counter()
    for path in sorted(paths):
        h=hashlib.sha256()
        with open(path,encoding="utf-8-sig",newline="") as f:
            for row in csv.DictReader(f):
                stats["source_rows"]+=1
                if not row.get("race_date","").startswith("2025-"):continue
                stats["source_rows_2025"]+=1
                try:
                    day=odds_race_id(row["race_id"])
                    bike=int(row["car_no"])
                    name=norm(row["player_name"])
                    age=int(row["age"])
                    if not (16<=age<=85 and 1<=bike<=9 and len(name)>1):
                        stats["invalid_age_or_name"]+=1;continue
                except (ValueError,KeyError,TypeError):
                    stats["unparseable_archive_entry"]+=1;continue
                key=(day,bike)
                record=(name,age,row["race_date"],row.get("class",""),str(path))
                if key in conflicts:continue
                if key in source_by_race_bike:
                    prior=source_by_race_bike[key]
                    if record[:4]!=prior[:4]:
                        conflicts.add(key);source_by_race_bike.pop(key,None)
                        stats["conflicting_same_race_bike_sources"]+=1
                    else:stats["identical_source_duplicates"]+=1
                else:source_by_race_bike[key]=record
        stats["source_files"]+=1
    stats["unique_age_source_race_bikes"]=len(source_by_race_bike)
    stats["source_conflicts"]=len(conflicts)
    return source_by_race_bike,stats

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--ages",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    files=list(Path(a.ages).glob("*.csv"))
    if len(files)!=16:raise RuntimeError(f"Expected original 16 S/A12/A3 2025 source files, found {len(files)}")
    index,stats=extract(files)
    entries=[];seen=set();audit=Counter()
    with open(a.source,encoding="utf-8-sig",newline="") as f:
        for row in csv.DictReader(f):
            audit["frozen_cohort_rows"]+=1
            target=(row["race_id"],int(row["ura_rank"]),row["rider_id"])
            if target in seen:raise RuntimeError("Duplicate frozen row")
            seen.add(target)
            src=index.get((row["race_id"],int(row["bike_number"]))) if "bike_number" in row else None
            # Previous 40,851 rows did not store bike_number, so original
            # full-2025 rider exposure CSV is also required to join by RIDER ID.
            entries.append(row)
    # Alternate join file includes race_id,bike,original rider_id.
    # This file must be strictly derived from signed FULL-source 2025 run
    # 40177278424, not from any 2026 source or backfilled current entrants.
    full=Path(a.source).parent/"2025_all_source_scored_rider_exposure.csv"
    if not full.exists():raise RuntimeError("The separate original rider_id->bike file is required")
    per={}
    with full.open(encoding="utf-8",newline="") as f:
        for r in csv.DictReader(f):
            if r["date"].startswith("2025-"):
                k=(r["race_id"],r["rider_id"])
                if k in per:raise RuntimeError("Duplicate ID original race row")
                per[k]=(r["bike"],norm(r["rider_name"]))
    if len(entries)!=EXPECTED or len({r["race_id"] for r in entries})!=EXPECTED_RACES:
        raise RuntimeError("Not exact fixed cohort")
    fout=[]
    for row in entries:
        key=(row["race_id"],row["rider_id"])
        b=per.get(key)
        if b is None:
            audit["missing_original_bike"]+=1
            name=norm(row["rider_name"]);age=None;match="missing_original_bike"
        else:
            bike=int(b[0]);name=norm(row["rider_name"]) or b[1]
            candidate=index.get((row["race_id"],bike))
            if candidate is None:
                age=None;match="no_alt_source_age";audit[match]+=1
            elif candidate[0]!=name and candidate[0]!=b[1]:
                age=None;match="name_mismatch";audit[match]+=1
            else:
                age=candidate[1];match="matched_2025_race_bike_name";audit[match]+=1
        fout.append({**row,"race_age":age,"age_match_status":match,
                     "age_source":"2025_KDreams_race_specific_entry_archive" if age is not None else ""})
    target=out/"2025_underdog_WITH_AGE_SCORE_AND_EXTERNAL_AGE.csv"
    with target.open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(fout[0]));w.writeheader();w.writerows(fout)
    manifest={"original_frozen_exposures":len(fout),
              "original_frozen_races":len({r["race_id"] for r in fout}),
              "age_join_audit":dict(audit),"source_audit":dict(stats),
              "source_files":sorted(x.name for x in files),
              "age_not_backfilled_from_future":True,
              "raw_pages_collected_after_event_caution":True,
              "research_only":True,"models_modified":False}
    (out/"2025_race_age_join_audit.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n")
    print("2025_AGE_RECOVERY",json.dumps(manifest,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
