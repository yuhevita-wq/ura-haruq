"""2025 WHOLE ARCHIVE rider outcome census and frozen-original-URA ranking.

No odds archive, no preselected V3 subset, no LOW-only/7car-only gate.
Each calendar month independent. All original archive race rows counted,
and exclusions/invalidity reasons recorded for every race. Girls reported
as separate category rather than silently thrown away. Ranking limited
to original immutable model-supported 5..9 field, not a source filter.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from collections import Counter,defaultdict
from datetime import date
from pathlib import Path

import numpy as np
from db.schema import get_connection
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker
from ura.core import prepare,low_rank_start
from ura.roi_holdout import feature_snapshot,FROZEN_HARUQ_MODEL_SHA256,frozen_artifact
from ura.formation_selector import load_fixed

YEAR=2025
UNIVERSE="ALL_RACES_IN_ORIGINAL_2025_HISTORY_DB"
OUTPUT_COLUMNS=[
 "race_id","date","source_grade","category","rider_count","rider_id","rider_name",
 "prefecture","entry_class","bike","ura_rank","underdog_threshold","is_low","score_gap",
 "line_group","line_role","finish_position","top3","first","second","third"
]

def category(entries,race_name):
    if "ガール" in str(race_name or "") or any(
         str(x["gender"] or "").upper() in ("F","W","女") for x in entries):
        return "ガールズ"
    classes={str(x["entry_class"] or "").strip().upper() for x in entries}
    groups=set()
    for s in classes:
        if s in {"S1","S2","SS"}:groups.add("S級")
        elif s in {"A1","A2"}:groups.add("A級1・2班")
        elif s=="A3":groups.add("チャレンジA3")
        else:groups.add("級班不明")
    return next(iter(groups)) if len(groups)==1 else "級班混在:"+",".join(sorted(groups))

def load_entries(conn,race_id):
    return conn.execute(
      """SELECT e.rider_id,e.rider_name,e.bike_number,e.class AS entry_class,
      e.prefecture,e.line_group,e.line_role,r.gender
      FROM entries e LEFT JOIN riders r ON r.rider_id=e.rider_id
      WHERE e.race_id=? ORDER BY e.bike_number""",(race_id,)).fetchall()

def real_podium(conn,rid,entries):
    rows=conn.execute(
      "SELECT bike_number,finish_position,rider_id FROM race_results WHERE race_id=?",
      (rid,)).fetchall()
    bikes={int(e["bike_number"]) for e in entries}
    if len(rows)!=len(entries) or any(x["bike_number"] is None for x in rows):
        return None,"results_roster_incomplete"
    rb=[int(x["bike_number"]) for x in rows]
    if set(rb)!=bikes or len(rb)!=len(set(rb)):
        return None,"result_bikes_disagree"
    good=[]
    for r in rows:
        p=r["finish_position"]
        if p is None:continue
        try:
            p=int(p)
        except Exception:
            return None,"noninteger_finish"
        if p>0:good.append((p,int(r["bike_number"])))
    if len({p for p,b in good})!=len(good):
        return None,"ties_or_duplicated_finish"
    podium=dict(good)
    if not all(p in podium for p in (1,2,3)):
        return None,"top3_missing"
    return {b:p for p,b in good},"ok"

def month_bounds(month):
    if not 1<=month<=12:raise ValueError("month outside 2025")
    start=f"2025-{month:02d}-01"
    end=f"2025-{month+1:02d}-01" if month<12 else "2026-01-01"
    return start,end

def write_csv(out,rows,fields):
    with out.open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def audit_month(month,out,model_path,as_census_only=False):
    start,end=month_bounds(month)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    frozen_artifact("artifacts/ura_ai_v1.json")
    if not as_census_only:
        if hashlib.sha256(Path(model_path).read_bytes()).hexdigest()!=FROZEN_HARUQ_MODEL_SHA256:
            raise RuntimeError("Reference ranker modified!")
        m=LGBMRanker();m.load(str(model_path))
        coef=np.asarray(load_fixed("artifacts/ura_ai_v1.json")["parameters"],float)
        builder=FeatureBuilder()
    cnt=Counter();all_race_status=[];exposures=[]
    with get_connection() as conn:
        races=conn.execute(
         """SELECT race_id,date,grade,race_name,rider_count FROM races
            WHERE date>=? AND date<? ORDER BY date,race_id""",(start,end)).fetchall()
        cnt["original_archive_races_in_month"]=len(races)
        for race in races:
            rid=str(race["race_id"]);day=str(race["date"])[:10]
            nominal=int(race["rider_count"] or 0)
            rec={"race_id":rid,"date":day,"grade":str(race["grade"] or ""),
                 "field_declared":nominal,"category":"","status":"","stage":""}
            cnt["original_grade:"+rec["grade"]]+=1
            ents=load_entries(conn,rid)
            rec["field_entries"]=len(ents)
            if ents:
                cat=category(ents,race["race_name"])
            else:cat="出走表なし"
            rec["category"]=cat
            cnt["original_category:"+cat]+=1
            cnt["original_declared_field:"+str(nominal)]+=1
            cnt["original_category_field:"+cat+"|"+str(nominal)]+=1
            if nominal not in range(5,10):
                rec["status"]="model_unsupported_field_size";cnt["unsupported_field_size"]+=1
                all_race_status.append(rec);continue
            if not ents or len(ents)!=nominal or len({str(e["rider_id"]) for e in ents})!=nominal or any(e["bike_number"] is None for e in ents) or len({int(e["bike_number"]) for e in ents})!=nominal:
                rec["status"]="invalid_or_missing_2025_entries";cnt["invalid_entries"]+=1
                all_race_status.append(rec);continue
            cnt["race_with_complete_entry_roster"]+=1
            podium,reason=real_podium(conn,rid,ents)
            if podium is None:
                rec["status"]=reason;cnt[reason]+=1
                all_race_status.append(rec);continue
            cnt["race_with_entry_and_top3_result"]+=1
            if as_census_only:
                rec["status"]="census_pass_only";all_race_status.append(rec);continue
            try:
                snap=feature_snapshot(builder,m,conn,rid,day)
                if {int(p["bike_number"]) for p in snap}!={int(e["bike_number"]) for e in ents}:
                    raise ValueError("predicted roster mismatch")
                ranked,feat,gap,_=prepare(snap)
                if not math.isfinite(gap):raise ValueError("nonfinite gap")
                util=feat@coef
                order=sorted(range(len(ranked)),key=lambda i:(-float(util[i]),int(ranked[i]["bike_number"])))
                bike_rank={int(ranked[i]["bike_number"]):j+1 for j,i in enumerate(order)}
            except Exception as e:
                rec["status"]="frozen_model_feature_failure";cnt["frozen_model_feature_failure"]+=1
                if cnt["frozen_model_feature_failure"]<=5:
                    print("FEATURE_SKIP",rid,type(e).__name__,str(e)[:170],flush=True)
                all_race_status.append(rec);continue
            low=int(gap<1)
            cnt["evaluated_all_low_high_races"]+=1
            cnt["evaluated_LOW" if low else "evaluated_HIGH"]+=1
            cnt["evaluated_category:"+cat]+=1
            cnt["evaluated_category_field:"+cat+"|"+str(nominal)]+=1
            cnt["evaluated_category_confidence:"+cat+"|"+("LOW" if low else "HIGH")]+=1
            threshold=low_rank_start(nominal)
            rec["status"]="evaluated"
            rec["is_low"]=low
            rec["gap"]=round(gap,8)
            all_race_status.append(rec)
            for e in ents:
                bike=int(e["bike_number"]);pos=podium.get(bike)
                exposures.append({
                    "race_id":rid,"date":day,"source_grade":rec["grade"],
                    "category":cat,"rider_count":nominal,
                    "rider_id":str(e["rider_id"]),
                    "rider_name":str(e["rider_name"] or ""),"prefecture":str(e["prefecture"] or ""),
                    "entry_class":str(e["entry_class"] or ""),"bike":bike,
                    "ura_rank":bike_rank[bike],
                    "underdog_threshold":threshold,
                    "is_low":low,"score_gap":round(gap,8),
                    "line_group":str(e["line_group"] or ""),"line_role":str(e["line_role"] or ""),
                    "finish_position":pos or "",
                    "top3":int(pos is not None and pos<=3),
                    "first":int(pos==1),"second":int(pos==2),"third":int(pos==3)})
        cnt["saved_race_status_rows"]=len(all_race_status)
        cnt["saved_rider_exposures"]=len(exposures)
    assert cnt["saved_race_status_rows"]==cnt["original_archive_races_in_month"]
    assert cnt["race_with_entry_and_top3_result"]>=cnt["evaluated_all_low_high_races"]
    write_csv(out/f"race_census_{month:02d}.csv",all_race_status,
              ["race_id","date","grade","field_declared","field_entries","category","status","is_low","gap"])
    write_csv(out/f"rider_exposures_{month:02d}.csv",exposures,OUTPUT_COLUMNS)
    summary={"year":YEAR,"month":month,"date_start":start,"end_exclusive":end,
        "no_final_odds_requirement":True,"no_low_filter":True,
        "no_7car_filter":True,"all_class_census_including_girls":True,
        "race_universe":"entire original March2026 history database, as archived for HaruQ",
        "source_model_sha256":FROZEN_HARUQ_MODEL_SHA256,
        "model_changed":False,"purchase_rules_changed":False,"production_unmodified":True,
        "warnings":["Race information may be absent from source archive; this is not official nationwide 2025 total.",
        "Model history may have trained on 2025 results; retrospective description only.",
        "Full as-of pre-race feature availability has not been proved.",
        "All 2025 archived race rows included in census; scoring errors explicitly excluded, no fallback ranking."],
        "audit":dict(cnt)}
    (out/f"audit_{month:02d}.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("FULL_2025_MONTH_CENSUS",json.dumps({"month":month,"audited":dict(cnt)},ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--month",type=int,required=True)
    p.add_argument("--model",type=Path)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--census-only",action="store_true")
    x=p.parse_args()
    if not x.census_only and not x.model:raise RuntimeError("Frozen original model required")
    audit_month(x.month,x.out,x.model,as_census_only=x.census_only)
if __name__=="__main__":main()
