"""Aggregate all 12 GitHub Actions monthly 2025 history shards without selective LOW/career filter."""
import argparse,csv,json,math
from collections import Counter,defaultdict
from pathlib import Path

from research.line_patterns_20261025.rider_low_rank_2025_only import wilson_lower
from research.line_patterns_20261025.full_2025_historical_player_universe import OUTPUT_COLUMNS

def readcsv(path):
    with Path(path).open(newline="",encoding="utf8") as f:return list(csv.DictReader(f))

def writecsv(path,rows,fields):
    with Path(path).open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader();w.writerows(rows)

def aggregate(input_dir,out_dir):
    input_dir=Path(input_dir);out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    audits=[];race_census=[];all_exposures=[]
    for month in range(1,13):
        fname=input_dir/f"audit_{month:02d}.json"
        if not fname.exists():raise RuntimeError("Missing month: "+str(month))
        a=json.loads(fname.read_text(encoding="utf8"))
        if a["month"]!=month:raise RuntimeError("Unexpected shard")
        audits.append(a)
        x=readcsv(input_dir/f"race_census_{month:02d}.csv")
        y=readcsv(input_dir/f"rider_exposures_{month:02d}.csv")
        if len(x)!=a["audit"]["original_archive_races_in_month"]:
            raise RuntimeError("Race census missing records month "+str(month))
        if len(y)!=a["audit"]["saved_rider_exposures"]:
            raise RuntimeError("Original rider exposures differ month "+str(month))
        race_census.extend(x);all_exposures.extend(y)
    keys=[x["race_id"] for x in race_census]
    if len(keys)!=len(set(keys)):raise RuntimeError("Duplicate race across 2025")
    observed=Counter(x["status"] for x in race_census)
    cat_all=Counter(x["category"] for x in race_census)
    if not all(x["date"].startswith("2025-") for x in race_census):
        raise RuntimeError("Wrong-year input")
    stage_audit=Counter()
    for a in audits:
        for k,n in a["audit"].items():
            if isinstance(n,int):stage_audit[k]+=n
    if len(race_census)!=stage_audit["original_archive_races_in_month"]:
        raise RuntimeError("All archived 2025 races were not represented")
    if len(all_exposures)!=stage_audit["saved_rider_exposures"]:
        raise RuntimeError("Aggregate individual exposures inconsistent")

    # Check immutable model's original lower-rank cut for each race:
    # n7=5..7, n8=5..8, n9=6..9 (see ura.core.low_rank_start).
    groups=defaultdict(list)
    admissible=Counter()
    totals=defaultdict(lambda:Counter())
    cat_race=defaultdict(set)
    unique_ids=set()
    for r in all_exposures:
        n=int(r["rider_count"])
        ur=int(r["ura_rank"]);low=int(r["is_low"])
        if n not in (5,6,7,8,9) or ur<1 or ur>n:
            raise RuntimeError("Malformed archived rank field")
        admissible[(r["race_id"],ur)]+=1
        if ur<int(r["underdog_threshold"]):continue
        cat=r["category"]
        fin=int(r["top3"])
        rid=r["rider_id"]
        # Full universe + 7car slice; LOW and HIGH are both retained.
        for sz in ("ALL_FIELDS",f"{n}CAR"):
            for confidence in ("ALL_CONFIDENCE","LOW" if low else "HIGH"):
                group=(cat,sz,confidence,rid)
                groups[group].append(r)
                c=totals[(cat,sz,confidence)]
                c["lower_rank_appearances"]+=1
                c["lower_rank_podium"]+=fin
                c["first"]+=int(r["first"])
                c["second"]+=int(r["second"])
                c["third"]+=int(r["third"])
                cat_race[(cat,sz,confidence)].add(r["race_id"])
        unique_ids.add(rid)
    for k,v in admissible.items():
        if v!=1:raise RuntimeError("Nonunique race rank "+str(k))

    leaders=[]
    for (cat,sz,confidence,rid),rs in groups.items():
        n=len(rs)
        podium=sum(int(r["top3"]) for r in rs)
        first=sum(int(r["first"]) for r in rs)
        second=sum(int(r["second"]) for r in rs)
        third=sum(int(r["third"]) for r in rs)
        if (first+second+third)!=podium:raise RuntimeError("Placements not disjoint")
        name=next((str(x["rider_name"]) for x in rs if x["rider_name"] and x["rider_name"].lower() not in ("unknown","none")),rs[0]["rider_name"])
        leaders.append({"category":cat,"field_scope":sz,"confidence_scope":confidence,
                        "rider_id":rid,"rider_name":name,
                        "2025_lower_rank_appearances":n,"2025_top3_count":podium,
                        "2025_top3_rate_pct":round(100*podium/n,4),
                        "wilson95_lower_pct":round(100*wilson_lower(podium,n),4),
                        "first":first,"second":second,"third":third,
                        "months_seen":len(set(x["date"][:7] for x in rs)),
                        "rank5_appearances":sum(int(x["ura_rank"])==5 for x in rs),
                        "rank6_appearances":sum(int(x["ura_rank"])==6 for x in rs),
                        "rank7_appearances":sum(int(x["ura_rank"])==7 for x in rs),
                        "rank8_appearances":sum(int(x["ura_rank"])==8 for x in rs),
                        "rank9_appearances":sum(int(x["ura_rank"])==9 for x in rs)})
    leaders.sort(key=lambda r:(
        r["category"],r["field_scope"],r["confidence_scope"],
        -r["2025_top3_count"],-r["2025_lower_rank_appearances"],r["rider_id"]))
    cols=list(leaders[0]) if leaders else []
    if cols:writecsv(out/"2025_all_source_players_by_actual_class_field_and_low.csv",leaders,cols)
    writecsv(out/"2025_source_all_original_races_and_status.csv",race_census,
             list(dict.fromkeys(k for r in race_census for k in r)))
    writecsv(out/"2025_all_source_scored_rider_exposure.csv",all_exposures,OUTPUT_COLUMNS)

    main=[r for r in leaders if r["field_scope"]=="7CAR" and
          r["confidence_scope"]=="LOW"]
    main.sort(key=lambda x:(x["category"],-x["2025_top3_count"],-x["2025_lower_rank_appearances"],x["rider_id"]))
    if main:writecsv(out/"2025_7car_LOW_all_classes_NO_MINIMUM.csv",main,cols)

    yearly_groups=[]
    for (cat,sz,conf),counts in sorted(totals.items()):
        x=counts["lower_rank_appearances"]
        yearly_groups.append({"category":cat,"fields":sz,"confidence":conf,
          "race_count":len(cat_race[(cat,sz,conf)]),
          "lower_rank_appearances":x,"lower_rank_top3_count":counts["lower_rank_podium"],
          "lower_rank_top3_rate_pct":round(100*counts["lower_rank_podium"]/x,4) if x else None,
          "first":counts["first"],"second":counts["second"],"third":counts["third"],
          "riders":len({rid for (c,n,cf,rid) in groups if (c,n,cf)==(cat,sz,conf)})})
    writecsv(out/"2025_year_class_field_LOW_HIGH_coverage.csv",yearly_groups,
             list(yearly_groups[0]) if yearly_groups else [])

    top_by_class={}
    for cat in ("S級","A級1・2班","チャレンジA3","ガールズ","級班不明"):
        sub=[r for r in main if r["category"]==cat]
        top_by_class[cat]=sub[:10]
    main_audit={
      "status":"COMPLETED_IF_MONTHS_ALL_PRESENT","year":2025,
      "source":"original 2025 history database, NOT v3-year CSV, NO ODDS",
      "months_received":len(audits),"all_original_2025_race_rows":len(race_census),
      "original_stage_status":dict(observed),"original_category":dict(cat_all),
      "all_scored_rider_rows":len(all_exposures),
      "unique_scored_riders_of_all_categories":len(unique_ids),
      "scored_low_high_by_actual_category_field":yearly_groups,
      "original_source_audit":dict(stage_audit),
      "top10_2025_7car_LOW_by_class":top_by_class,
      "models_modified":False,"all_conditions_disclosed":True,
      "limitations":["Original historical DB may itself miss nationwide races, and its newest revision may contain later corrections.",
        "Scoring missing entry/outcome/supported rider count is not replaced by invented observations.",
        "Original HaruQ model may have fitted on 2025; 2025 is descriptive, not unseen evaluation.",
        "LOW and HIGH both scored; previous V3 final odds completeness not required.",
        "All source race IDs and all class categories included in census, every exclusion status retained.",
        "No purchase allocation, ticket prices, betting decisions, or fixed models changed."]
    }
    (out/"2025_FULL_UNIVERSE_AUDIT.json").write_text(
         json.dumps(main_audit,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print("2025_FULL_MERGE_RESULT",json.dumps({
        "all_original_2025_races":len(race_census),
        "source_categories":dict(cat_all),
        "audit":{"evaluated":stage_audit["evaluated_all_low_high_races"],
                 "LOW":stage_audit["evaluated_LOW"],
                 "HIGH":stage_audit["evaluated_HIGH"],
                 "invalid_entries":stage_audit["invalid_entries"],
                 "results_roster_incomplete":stage_audit["results_roster_incomplete"],
                 "model_failures":stage_audit["frozen_model_feature_failure"]},
        "2025_7car_LOW_by_class":[x for x in yearly_groups if x["fields"]=="7CAR" and x["confidence"]=="LOW"],
        "top10_7car_LOW":top_by_class,
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args();aggregate(a.input,a.out)
