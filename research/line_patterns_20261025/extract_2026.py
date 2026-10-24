"""Research-only 2026H1 snapshot extractor. Repeats immutable V3 recipe.

No original feature/model retraining. Audits output as a historical reconstruction,
not a genuine live prediction. 2026 records never enter the 2023/24 discovery.
"""
from __future__ import annotations
import argparse, csv, hashlib, itertools, json
from collections import Counter
from pathlib import Path

import numpy as np
from db.schema import get_connection
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker
from ura.core import prepare
from ura.roi_holdout import actual_and_payout,feature_snapshot,FROZEN_HARUQ_MODEL_SHA256,frozen_artifact
from ura.formation_selector import load_fixed
from ura.formation_external_audit import archive_odds

START="2026-01-01"
END="2026-07-01"
URA_SHA="5a353643cfe8196ab695521d479237294c3f9b66"


def extract(model_path, odds_path, out_dir):
    model_path=Path(model_path)
    if hashlib.sha256(model_path.read_bytes()).hexdigest()!=FROZEN_HARUQ_MODEL_SHA256:
        raise RuntimeError("Frozen HaruQ SHA256 mismatch")
    frozen_artifact("artifacts/ura_ai_v1.json")
    base=load_fixed("artifacts/ura_ai_v1.json")
    ranker=LGBMRanker();ranker.load(str(model_path))
    builder=FeatureBuilder()
    books,archived=archive_odds(odds_path, START, END)
    counts=Counter()
    saved=[]
    with get_connection() as conn:
        for rid,book in sorted(books.items()):
            counts["archived"]+=1
            race=conn.execute("SELECT date,rider_count,race_name,grade FROM races WHERE race_id=?",(rid,)).fetchone()
            if race is None:counts["race_missing"]+=1;continue
            day=str(race["date"])[:10]
            if not START<=day<END or day.replace("-","")!=rid[:8]:
                counts["date_mismatch"]+=1;continue
            if "ガール" in str(race["race_name"] or ""):
                counts["girls"]+=1;continue
            settled=actual_and_payout(conn,rid)
            if settled is None:counts["no_complete_result"]+=1;continue
            combo,payout,entrants=settled
            n=len(entrants)
            if n!=7 or int(race["rider_count"])!=7:
                counts["not_7_car"]+=1;continue
            if payout is None:
                counts["missing_payout"]+=1;continue
            prices=book["odds"]
            if len(prices)!=210 or set(prices)!={"-".join(map(str,p)) for p in itertools.permutations(sorted(entrants),3)}:
                counts["incomplete_final_trifecta"]+=1;continue
            if combo not in prices or abs(100*float(prices[combo])-payout)>1.5:
                counts["price_settlement_disagrees"]+=1;continue
            try:
                raw=feature_snapshot(builder,ranker,conn,rid,day)
                if {int(x["bike_number"]) for x in raw}!=entrants:
                    counts["feature_roster_mismatch"]+=1;continue
                riders,feat,gap,_=prepare(raw)
                if gap>=1.0:
                    counts["not_low"]+=1;continue
                utilities=feat@np.asarray(base["parameters"],dtype=float)
                ordering=sorted(range(7), key=lambda i:(-utilities[i],int(riders[i]["bike_number"])))
                ranked=[int(riders[i]["bike_number"]) for i in ordering]
                index={v:i+1 for i,v in enumerate(ranked)}
                actual=tuple(index[int(v)] for v in combo.split("-"))
                if len(set(actual))!=3:raise ValueError("Repeated actual")
            except Exception as exc:
                counts["feature_failed"]+=1
                if counts["feature_failed"]<=4:
                    print("RESEARCH_SKIP",rid,type(exc).__name__,str(exc)[:130],flush=True)
                continue
            saved.append({
                "race_id":rid,"date":day,"year":2026,"n":7,
                "grade":str(race["grade"] or ""),"gap":round(gap,8),
                "ranked_bikes":"-".join(map(str,ranked)),
                "actual_bikes":combo,"actual_ranks":"-".join(map(str,actual)),
                "payout_yen":int(payout),"final_winning_odds":float(prices[combo])
            })
    root=Path(out_dir);root.mkdir(parents=True,exist_ok=True)
    with (root/"year_2026H1.csv").open("w",encoding="utf8",newline="") as fp:
        if not saved:raise RuntimeError("No 2026 records")
        writer=csv.DictWriter(fp,fieldnames=list(saved[0]))
        writer.writeheader();writer.writerows(saved)
    report={"period":[START,END],"matched_low_7_car":len(saved),
            "rejection":dict(counts),"archive":archived,
            "frozen_model_sha256":FROZEN_HARUQ_MODEL_SHA256,
            "frozen_ura_git_sha":URA_SHA,
            "research_only":True,"purchase_approved":False,
            "source_labels_selected_families":False,
            "limit":"Historical features/final odds do not prove pre-race execution"}
    (root/"extract_2026_audit.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    print("EXTRACT_2026_H1",json.dumps(report,ensure_ascii=False),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--odds",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    extract(a.model,a.odds,a.out)
