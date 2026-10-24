"""Reproduceable cash-accounting audit for frozen URA coefficients.

Not live-bet evidence: historical odds archive holds final, not timestamped T-15, odds.
No test labels inform selections. One fixed rule, 100 JPY per positive-EV ticket.
"""
import csv
import gzip
import hashlib
import itertools
import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from config import DB_PATH
from db.schema import get_connection
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker
from ura.core import select_candidates

FROZEN_URA_BLOB_SHA="5a353643cfe8196ab695521d479237294c3f9b66"
FROZEN_HARUQ_MODEL_SHA256="e70690f4538b77e927a042e7ae2db312f8a0a4f16bf72def37e7ccad3568c193"
ODDS_PHASE="final - not documented as pre-deadline snapshots"
PER_TICKET_YEN=100
DATE_PREFIX="202606"
EV_CUTOFF=1.0

def odds_race_id(rid):
    """KD-style ID NN+YYYYMMDD+venue(2)+race(4) to YYYYMMDD+venue(2)+race(2)."""
    rid=str(rid)
    if not re.fullmatch(r"\d{16}",rid):
        raise ValueError("Unsupported archive ID format")
    num=int(rid[-4:])
    if not 1<=num<=12:
        raise ValueError("Archive race number out of bounds")
    return rid[2:12]+f"{num:02d}"

def frozen_artifact(path):
    p=Path(path)
    data=p.read_bytes()
    blob=hashlib.sha1(f"blob {len(data)}\0".encode()+data).hexdigest()
    if blob!=FROZEN_URA_BLOB_SHA:
        raise RuntimeError("Saved URA coefficient file has changed; audit aborted")
    obj=json.loads(data)
    if not (obj["training_completed"] and obj["trained_on_real_results"]):
        raise RuntimeError("Not an actual previously trained frozen model")
    if obj["date_ranges"]["holdout_end_exclusive"]!="2026-10-01":
        raise RuntimeError("Unexpected frozen model provenance")
    return obj

def load_quoted_odds(path):
    found={}
    quarantined=set()
    counters=Counter()
    with gzip.open(path,"rt",encoding="utf-8") as fp:
        for line in fp:
            obj=json.loads(line)
            try:
                rid=odds_race_id(obj["race_id"])
            except (KeyError,ValueError):
                counters["invalid_id"]+=1
                continue
            if not rid.startswith(DATE_PREFIX):
                continue
            market=obj.get("odds",{}).get("3連単",{})
            if not isinstance(market,dict) or not market:
                counters["no_trifecta"]+=1
                continue
            if rid in quarantined:
                counters["already_quarantined_duplicate"]+=1
                continue
            prices={}
            for ticket,price in market.items():
                if (not isinstance(ticket,str)
                        or not re.fullmatch(r"[1-9]-[1-9]-[1-9]",ticket)
                        or len(set(ticket.split("-")))!=3
                        or not isinstance(price,(int,float))
                        or not math.isfinite(price) or price<=1):
                    continue
                prices[ticket]=float(price)
            if rid in found:
                if found[rid]["prices"]==prices:
                    counters["identical_duplicate_quotes"]+=1
                else:
                    # Ambiguous same-race quotes are excluded entirely, never resolved
                    # by which market price later won or by choosing a better payout.
                    del found[rid]
                    quarantined.add(rid)
                    counters["conflicting_duplicate_races"]+=1
                continue
            found[rid]={"prices":prices,"archive_id":obj["race_id"],
                        "race_type":obj.get("race_type")}
            counters["archived_june_races"]+=1
    counters["quarantined_races"]=len(quarantined)
    counters["unique_unambiguous_june_races"]=len(found)
    return found,dict(counters)

def actual_and_payout(conn,rid):
    rows=conn.execute("""
        SELECT bike_number,finish_position FROM race_results
        WHERE race_id=? ORDER BY finish_position""",(rid,)).fetchall()
    if len(rows)<5:return None
    riders=[int(r["bike_number"]) for r in rows]
    pos=[r["finish_position"] for r in rows]
    if len(riders)!=len(set(riders)) or not all(x is not None for x in pos) or len(pos)!=len(set(pos)):
        return None
    finished={int(r["finish_position"]):int(r["bike_number"]) for r in rows}
    if not all(k in finished for k in (1,2,3)):return None
    outcome="-".join(str(finished[i]) for i in (1,2,3))
    pay=conn.execute("""
        SELECT sanrentan_combo,sanrentan_payout
        FROM race_payouts WHERE race_id=?""",(rid,)).fetchone()
    settlement=None
    if pay and pay["sanrentan_combo"] and pay["sanrentan_payout"]:
        combo="-".join(re.findall(r"[1-9]",str(pay["sanrentan_combo"])))
        if combo==outcome and float(pay["sanrentan_payout"])>0:
            settlement=int(pay["sanrentan_payout"])
    return outcome,settlement,set(riders)

def feature_snapshot(builder,ranker,conn,rid,day):
    # NEVER allow a silent same-race outcome fallback from unavailable entry rows.
    entries=conn.execute("SELECT rider_id,bike_number FROM entries WHERE race_id=?",(rid,)).fetchall()
    if not entries or len(entries)!=len({r["rider_id"] for r in entries}):
        raise ValueError("No complete uniquely identified pre-race entries")
    feat=builder.build_race_features(rid,day,conn)
    if len(feat)!=len(entries):
        raise ValueError("Rider feature rows do not match entries")
    bike={str(r["rider_id"]):int(r["bike_number"]) for r in entries}
    if any(str(rid2) not in bike for rid2 in feat["rider_id"]):
        raise ValueError("Feature rider IDs not in entries")
    cols=list(ranker.feature_names)
    if len(cols)!=44 or not all(c in feat.columns for c in cols):
        raise ValueError("Fixed 44 features missing")
    score=ranker.predict(feat[cols].apply(pd.to_numeric,errors="coerce").fillna(0.0))
    return [{"bike_number":bike[str(rid2)],"score":float(v)}
            for rid2,v in zip(feat["rider_id"],score)]

def bootstrap(races,rep=1000):
    """Cluster bootstrap by race: stake and payout are paired per race."""
    if not races:return {"ci95_roi_percent":None}
    totals=np.array([[z["investment"],z["actual_return"]] for z in races],float)
    rng=np.random.default_rng(20261024)
    returns=[]
    n=len(races)
    for _ in range(rep):
        ix=rng.integers(0,n,n)
        stake,paid=totals[ix].sum(axis=0)
        if stake>0:returns.append(100*paid/stake)
    return {"ci95_roi_percent":[round(float(v),2) for v in np.quantile(returns,[.025,.975])]}

def backtest(model_path,artifact_path,odds_path,output_dir):
    model_data=frozen_artifact(artifact_path)
    if hashlib.sha256(Path(model_path).read_bytes()).hexdigest()!=FROZEN_HARUQ_MODEL_SHA256:
        raise RuntimeError("Source HaruQ base model changed")
    coefficients=model_data["parameters"]
    odds,archive_counts=load_quoted_odds(odds_path)
    model=LGBMRanker()
    model.load(str(model_path))
    builder=FeatureBuilder()
    audit=Counter()
    race_rows=[]
    ticket_rows=[]
    with get_connection() as conn:
        for rid,book in sorted(odds.items()):
            audit["odds_races"]+=1
            details=conn.execute("SELECT date,rider_count,race_name FROM races WHERE race_id=?",(rid,)).fetchone()
            if details is None:
                audit["not_in_history_db"]+=1;continue
            if not str(details["date"]).startswith("2026-06"):
                audit["date_disagreement"]+=1;continue
            if "ガール" in str(details["race_name"] or ""):
                audit["girls_excluded"]+=1;continue
            result=actual_and_payout(conn,rid)
            if result is None:
                audit["invalid_or_unfinished_result"]+=1;continue
            actual,payout,entrants=result
            n=len(entrants)
            if n<5 or n>9 or int(details["rider_count"])!=n:
                audit["roster_count_mismatch"]+=1;continue
            # Full-book validation; never treat a missing final price as a loss.
            expected=n*(n-1)*(n-2)
            if len(book["prices"])!=expected:
                audit["incomplete_trifecta_book"]+=1;continue
            if set(book["prices"])!={"-".join(map(str,t)) for t in itertools.permutations(sorted(entrants),3)}:
                audit["price_roster_mismatch"]+=1;continue
            try:
                ranked=feature_snapshot(builder,model,conn,rid,str(details["date"]))
                if {int(x["bike_number"]) for x in ranked}!=entrants:
                    audit["feature_roster_mismatch"]+=1;continue
                candidate_data=select_candidates(ranked,coefficients)
            except Exception as e:
                audit["feature_or_score_failure"]+=1
                if len(audit)<35:print("SKIPPED",rid,type(e).__name__,str(e)[:180],flush=True)
                continue
            audit["scored_complete_races"]+=1
            if not candidate_data["low_confidence"]:
                audit["high_confidence_excluded"]+=1;continue
            audit["eligible_low_races"]+=1
            positive=[]
            for c in candidate_data["candidates"]:
                price=book["prices"][c["ticket"]]
                ratio=float(c["probability"])*price
                if ratio>EV_CUTOFF:
                    positive.append((c["ticket"],float(c["probability"]),price,ratio))
            if not positive:
                audit["no_positive_ev_candidates"]+=1
                continue
            audit["races_with_positive_ev"]+=1
            hit=[t for t in positive if t[0]==actual]
            # When official payouts are missing, do NOT manufacture actual return;
            # retain separately for coverage, but omit from cash ROI denominator.
            if payout is None:
                audit["missing_official_settlement"]+=1
                continue
            invested=PER_TICKET_YEN*len(positive)
            returned=payout if hit else 0
            predicted_return=PER_TICKET_YEN*sum(z[3] for z in positive)
            race_rows.append({
               "race_id":rid,"date":str(details["date"]),"low_gap":candidate_data["confidence_gap"],
               "all_research_candidates":candidate_data["candidate_count"],
               "tickets_bought":len(positive),
               "expected_return_yen":round(predicted_return,4),
               "investment":invested,"actual_return":returned,
               "profit":returned-invested,"hit":int(bool(hit)),
               "actual_combo":actual,"official_payout_per_100_yen":payout})
            for t,p,price,ratio in positive:
                ticket_rows.append({
                    "race_id":rid,"ticket":t,"estimated_probability":round(p,10),
                    "archived_final_odds":price,"estimated_ev_per_yen":round(ratio-1,7),
                    "stake_yen":PER_TICKET_YEN,
                    "actual_payout_yen":payout if t==actual else 0,
                    "hit":int(t==actual)})
    invested=sum(z["investment"] for z in race_rows)
    returned=sum(z["actual_return"] for z in race_rows)
    total_expected=sum(z["expected_return_yen"] for z in race_rows)
    summary={
      "system":"URA_HARUQ_FROZEN_ODDS_ROI_AUDIT_V1",
      "result_status":"completed" if invested else "insufficient matched audited data",
      "frozen_ura_git_blob_sha":FROZEN_URA_BLOB_SHA,
      "frozen_source_model_sha256":FROZEN_HARUQ_MODEL_SHA256,
      "coefficient_reestimated":False,
      "date_range":"2026-06-01..2026-06-30",
      "not_used_to_fit_ura_coefficients":True,
      "previously_used_for_logloss_holdout":True,
      "odds_phase":ODDS_PHASE,
      "ticket_type":"3連単",
      "rule":"LOW gap<1.0; any lower-ranked rider in finishing positions 1-3; buy every candidate with p*final_odds>1.0; 100 yen each; no post-outcome selection",
      "archive_coverage":archive_counts,"join_audit":dict(audit),
      "evaluated_races_with_official_settlement":len(race_rows),
      "tickets":len(ticket_rows),
      "hits":sum(z["hit"] for z in race_rows),
      "hit_rate_per_ticket":round(sum(z["hit"] for z in race_rows)/len(ticket_rows),7) if ticket_rows else None,
      "investment_yen":invested,"actual_return_yen":returned,
      "actual_profit_yen":returned-invested,
      "actual_roi_percent":round(100*returned/invested,4) if invested else None,
      "theoretical_expected_roi_percent":round(100*total_expected/invested,4) if invested else None,
      "bootstrap_1000":bootstrap(race_rows),
      "limitations":[
        "Archived odds are FINAL prices, not timestamped pre-deadline tradable quotes.",
        "The held-out June results were included in the prior numerical holdout audit; ROI rule here is fixed before seeing these June settlements, but independence beyond all earlier examination is not established.",
        "Current historical entries, rider master class and weather may not reconstruct exact start-time features; no profitable live-trading claim permitted.",
        "No price was imputed, incomplete odds books or missing official settlements were excluded."
      ],
      "purchase_approved":False
    }
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    for name,rows in (("races.csv",race_rows),("tickets.csv",ticket_rows)):
        with (output_dir/name).open("w",newline="",encoding="utf-8") as f:
            if rows:
                w=csv.DictWriter(f,fieldnames=list(rows[0]))
                w.writeheader();w.writerows(rows)
    print("URA_ROI_AUDIT_RESULT",json.dumps(summary,ensure_ascii=False),flush=True)
    return summary

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",type=Path,required=True)
    ap.add_argument("--artifact",type=Path,default=Path("artifacts/ura_ai_v1.json"))
    ap.add_argument("--odds",type=Path,required=True)
    ap.add_argument("--out",type=Path,default=Path("ura_roi_outputs"))
    args=ap.parse_args()
    backtest(args.model,args.artifact,args.odds,args.out)

if __name__=="__main__":main()
