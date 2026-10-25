"""Audit the 2025 V3 rider leaderboard source by ACTUAL original entry class.

Do not infer S/A class from F1/F2, and do not change the original data or
rank models. Every class and racer count is auditable. All outputs label
coverage and selection; no purchase decisions.
"""
import argparse,csv,json
from collections import Counter,defaultdict
from pathlib import Path
from research.line_patterns_20261025.rider_low_rank_2025_only import load_source,read_one_2025_race,make_exposures,per_rider,stats,write_csv
from db.schema import get_connection

def group_one(c):
    s=str(c or "").upper().strip().replace(" ","")
    if s in ("S1","S2","SS"):return "S級"
    if s in ("A1","A2"):return "A級1・2班"
    if s in ("A3",):return "チャレンジ(A3)"
    return "不明・分類不能"

def one_race_tier(entries):
    groups={group_one(r["class"]) for r in entries}
    if len(groups)==1:return next(iter(groups))
    return "混在("+",".join(sorted(groups))+")"

def audit(input_csv,out):
    rows,sha=load_source(input_csv)
    source_grade=Counter(r.get("grade","") for r in rows)
    all7_grade=Counter(r.get("grade","") for r in rows if r.get("n")=="7")
    all7=0;bad=Counter();races=[]
    t_by_grade=Counter();t_by_low=Counter()
    with get_connection() as conn:
        for row in rows:
            if row.get("n")!="7":continue
            all7+=1
            r=read_one_2025_race(conn,row,bad)
            if r is None:continue
            tier=one_race_tier(r["entries"])
            r["tier"]=tier
            r["grade"]=row.get("grade","")
            t_by_grade[(tier,r["grade"])]+=1
            t_by_low[(tier,r["LOW_band"])]+=1
            races.append(r)
    exposures=make_exposures(races)
    matched_counts=Counter(r["tier"] for r in races)
    if len(races)!=4866 or len(exposures)!=14598:
        raise RuntimeError("Frozen 2025 study population was silently changed")
    by_tier=[]
    player_rows=[]
    rows_322=[]
    for tier in sorted(matched_counts):
        selected=[e for e in exposures if next(x for x in races if x["race_id"]==e["race_id"])["tier"]==tier]
        sub_322=[e for e in selected if e["is_322"]]
        p=per_rider(selected,"2025_7car_LOW_"+tier)
        p322=per_rider(sub_322,"2025_322_LOW_"+tier) if sub_322 else []
        for entry in p:
            player_rows.append({"tier":tier,**entry})
        for entry in p322:
            rows_322.append({"tier":tier,**entry})
        by_tier.append({"tier":tier,"races":len(selected)//3,
          "races_322":len(sub_322)//3,"low_rank_appearances":len(selected),
          "low_rank_top3":sum(x["top3"] for x in selected),
          "top3_rate_pct":round(100*sum(x["top3"] for x in selected)/len(selected),4),
          "unique_riders":len(p),
          "riders_low10plus":sum(x["low_rank_starts"]>=10 for x in p),
          "riders_low5plus_322":sum(x["low_rank_starts"]>=5 for x in p322)})
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if player_rows:write_csv(out/"2025_7car_LOW_players_SPLIT_BY_CLASS.csv",player_rows)
    if rows_322:write_csv(out/"2025_322_LOW_players_SPLIT_BY_CLASS.csv",rows_322)
    grade_rows=[{"class_from_entries":t,"grade_on_source":g,"races":n}
                for (t,g),n in sorted(t_by_grade.items())]
    write_csv(out/"2025_7car_LOW_CLASS_VS_GRADE.csv",grade_rows)
    write_csv(out/"2025_7car_LOW_class_summary.csv",by_tier)
    main={
       "2025_v3_input_count":len(rows),
       "source_grade_ALL_7to9":dict(source_grade),
       "source_grade_7car":dict(all7_grade),
       "source_only_LOW_filter":True,
       "source_requires_complete_final_trifecta_odds":True,
       "races_7car_LOW":len(races),
       "class_race_totals":dict(matched_counts),
       "class_versus_grade":grade_rows,
       "by_tier":by_tier,
       "study_input_sha256":sha,
       "original_haruq_ura_unmodified":True,
       "research_only":True,
       "limitations":["This is an archived selected dataset, not the all-keirin race universe",
         "The 2025 labeled original source was selected by LOW and full final 3-ren-tan odds before this class audit",
         "Race tiers are derived only from the original race-specific entry class, never from meeting grade",
         "2025 past results may overlap frozen HaruQ training.",
         "No marketing/production workflow or purchase rule is affected."]
    }
    (out/"2025_CLASS_SCOPE_CORRECTION_AUDIT.json").write_text(
       json.dumps(main,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("2025_CLASS_AUDIT",json.dumps(main,ensure_ascii=False),flush=True)
    return main

def main():
    p=argparse.ArgumentParser();p.add_argument("--input",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args();audit(a.input,a.out)
if __name__=="__main__":main()
