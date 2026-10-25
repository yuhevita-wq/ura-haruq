"""Separate-cohort test: pre-frozen LOW 4/20, July-Aug 2026 S F1 races.

Research-only. Scores with immutable HaruQ ranker and immutable URA weights.
The July-Aug event rows were captured after their respective race dates;
this is a historically reproduced test, not documented pre-start forecasting.
None of these outcomes may influence the portfolio or thresholds.
"""
from __future__ import annotations
import argparse,csv,hashlib,itertools,json,math,re,unicodedata
from collections import Counter,defaultdict
from pathlib import Path
from statistics import median,mean

import numpy as np
from db.schema import get_connection
from db.schema import init_db
from features.builder import FeatureBuilder
from models.lgbm_ranker import LGBMRanker
from ura.core import prepare,low_rank_start
from ura.roi_holdout import feature_snapshot,frozen_artifact,FROZEN_HARUQ_MODEL_SHA256,odds_race_id
from ura.formation_selector import load_fixed
from research.line_patterns_20261025.geometry_322 import load_322,structural_key,band

BEGIN="2026-07-01"
END="2026-08-31"
PERIODS=(("2026-07",BEGIN,"2026-08-01"),("2026-08","2026-08-01",END))
REQUIRED={"0.00-0.25":0,"0.25-0.50":4,"0.50-0.75":0,"0.75-1.00":20}
SOURCE_PATH="data/2026_future_block1/s_class_f1_20260701_20260830"
ARCHIVE_SOURCE="yuhevita-wq/keirin-market-analysis-"
BASELINE_SHA="e70690f4538b77e927a042e7ae2db312f8a0a4f16bf72def37e7ccad3568c193"
URA_BLOB="5a353643cfe8196ab695521d479237294c3f9b66"


def clean_name(n):
    return "".join(ch for ch in unicodedata.normalize("NFKC",str(n or ""))
                   if not ch.isspace() and ch not in {"　","·"})


def all_csv(directory,name):
    with (Path(directory)/(name+".csv")).open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def num(s):
    if s is None or s=="":return None
    return float(str(s).replace(",","").replace("%",""))


def f_int(s):
    v=num(s)
    return int(v) if v is not None else None


def canonical_id(kd_race_id):
    return odds_race_id(kd_race_id)


def historical_rider_index(conn):
    rows=conn.execute("SELECT rider_id,name,prefecture FROM riders").fetchall()
    names=defaultdict(set);pref=defaultdict(set)
    for r in rows:
        name=clean_name(r["name"])
        if not name:continue
        names[name].add(str(r["rider_id"]))
        pf=clean_name(r["prefecture"])
        if pf:pref[(name,pf)].add(str(r["rider_id"]))
    return names,pref


def determine_rider_id(entry,names,pref):
    name=clean_name(entry["player_name"])
    pf=clean_name(entry["prefecture"])
    group=pref.get((name,pf)) or names.get(name,set())
    if len(group)>1:
        raise ValueError("Ambiguous original rider ID: "+name)
    if len(group)==1:return next(iter(group)),True
    if not name or not entry["term"] or not pf:
        raise ValueError("Missing rider identity: "+name)
    return "h2test:"+name+":"+pf+":"+str(entry["term"]),False


def check_source(races,entries,results,payouts):
    if len(races)!=901 or len(entries)!=6225 or len(results)!=6225:
        raise RuntimeError("Source cohort unexpectedly changed")
    if len({x["race_id"] for x in races})!=901:
        raise ValueError("Duplicate race IDs")
    rids=set(x["race_id"] for x in races)
    if any(x["race_id"] not in rids for x in itertools.chain(entries,results,payouts)):
        raise ValueError("Orphan source record")
    if any(not BEGIN<=x["race_date"]<END for x in races):
        raise ValueError("Future-period source changed")
    if any(x["meeting_grade"]!="F1" or not x["race_type"].startswith("Ｓ級") for x in races):
        raise ValueError("Non-F1/non-S class source")
    return rids


def payout_index(payouts):
    idx=defaultdict(list)
    for r in payouts:
        if r["ticket_type"]=="3連単" and r["status"]=="paid":
            combo=r["combination"].replace("→","-").replace("=","-")
            digits=re.findall(r"[1-9]",combo)
            if len(digits)!=3 or len(set(digits))!=3:
                raise ValueError("Malformed winning combo: "+combo)
            val=f_int(r["payout_yen"])
            if val is None or val<=0:raise ValueError("Invalid payout")
            idx[r["race_id"]].append(("-".join(digits),val))
    return idx


def observed_result(results,n):
    if len(results)!=n:return None
    pairs=[(f_int(x["finish_position"]),f_int(x["car_no"])) for x in results]
    if any(a is None or b is None for a,b in pairs):return None
    if sorted(x[0] for x in pairs)!=list(range(1,n+1)):return None
    if len(set(x[1] for x in pairs))!=n:return None
    by=dict(pairs)
    return "-".join(str(by[k]) for k in (1,2,3))


def import_source(conn,source):
    races=all_csv(source,"races")
    entries=all_csv(source,"entries")
    results=all_csv(source,"results")
    payouts=all_csv(source,"payouts")
    check_source(races,entries,results,payouts)
    ent=defaultdict(list);res=defaultdict(list);pays=payout_index(payouts)
    for x in entries:ent[x["race_id"]].append(x)
    for x in results:res[x["race_id"]].append(x)
    names,pref=historical_rider_index(conn)
    audit=Counter()
    audit["snapshot_total_preexisting_results"]=conn.execute(
         "SELECT count(*) FROM race_results WHERE finish_position IS NOT NULL").fetchone()[0]
    r=conn.execute("SELECT MIN(date),MAX(date) FROM races").fetchone()
    audit["history_first_race_date"]=r[0];audit["history_last_race_date"]=r[1]
    audit["history_results_2026_AprJun"]=conn.execute(
        "SELECT count(*) FROM race_results rr JOIN races r ON rr.race_id=r.race_id "
        "WHERE r.date >= '2026-04-01' AND r.date < '2026-07-01' AND rr.finish_position IS NOT NULL"
    ).fetchone()[0]
    # Knowable before the first July race; never learn the test results to
    # estimate odds, thresholds, rider categories, or shape selection.
    original_win=conn.execute(
       "SELECT AVG(win_rate),MAX(win_rate) FROM entries WHERE win_rate IS NOT NULL").fetchone()
    original_win_avg=float(original_win[0]) if original_win[0] is not None else None
    original_win_max=float(original_win[1]) if original_win[1] is not None else None
    if original_win_avg is None:raise RuntimeError("No historical win-rate scale to compare")
    source_vals=[num(x["win_rate"]) for x in entries if x["win_rate"]]
    source_avg=float(np.mean(source_vals))
    # The original netkeirin scraper uses percent points (not decimals).
    # Require same scale instead of silently multiplying.
    if (original_win_avg>1)!=(source_avg>1):
        raise RuntimeError("Historical/source win-rate scale mismatch: "+str((original_win_avg,source_avg)))
    audit["source_win_rate_mean"]=round(source_avg,4)
    audit["history_win_rate_mean"]=round(original_win_avg,4)
    audit["history_win_rate_max"]=original_win_max
    eligible=[]
    source_rids=set()
    for race in sorted(races,key=lambda x:(x["race_date"],x["race_id"])):
        source_id=race["race_id"];rid=canonical_id(source_id)
        n=int(race["entry_count"])
        audit["source_total_races"]+=1
        audit["source_7car" if n==7 else "source_other_car"]+=1
        # Existing archived races are expected: this database goes through
        # 2026-10-22. Use genuine archived pre-race entries and results only
        # after independently confirming KDreams result and settlement.
        existing=conn.execute(
            "SELECT date,rider_count,velodrome FROM races WHERE race_id=?",(rid,)).fetchone()
        if existing:audit["history_race_id_collision"]+=1
        rs=res[source_id];es=ent[source_id]
        winners=observed_result(rs,n)
        if winners is None or len(es)!=n or len({int(e["car_no"]) for e in es})!=n:
            audit["incomplete_outcome_or_entry"]+=1;continue
        win_pays=pays.get(source_id,[])
        match=[v for combo,v in win_pays if combo==winners]
        if len(win_pays)!=1 or len(match)!=1:
            audit["payout_not_unique_or_disagree"]+=1;continue
        payout=match[0]
        if existing:
            # NO replacement of original archived entries, lines or results.
            if str(existing["date"])[:10]!=race["race_date"] or int(existing["rider_count"])!=n:
                audit["existing_date_or_n_disagreement"]+=1;continue
            prior=conn.execute(
               "SELECT bike_number,rider_name,line_group,line_role FROM entries WHERE race_id=?",
                (rid,)).fetchall()
            if len(prior)!=n or {int(x["bike_number"]) for x in prior} != {int(x["car_no"]) for x in es}:
                audit["existing_entry_roster_disagreement"]+=1;continue
            expected_name={int(x["car_no"]):clean_name(x["player_name"]) for x in es}
            if any(clean_name(x["rider_name"])!=expected_name.get(int(x["bike_number"]))
                   for x in prior):
                audit["existing_entry_name_disagreement"]+=1;continue
            actual_data=conn.execute(
               "SELECT bike_number,finish_position FROM race_results WHERE race_id=?",
                (rid,)).fetchall()
            historic=observed_result([{"car_no":str(x["bike_number"]),
                                     "finish_position":str(x["finish_position"])}
                                     for x in actual_data if x["finish_position"] is not None],n)
            payout_hist=conn.execute(
                "SELECT sanrentan_combo,sanrentan_payout FROM race_payouts WHERE race_id=?",
                (rid,)).fetchone()
            if historic!=winners or not payout_hist or not payout_hist["sanrentan_payout"]:
                audit["existing_result_incomplete_or_disagree"]+=1;continue
            histcombo="-".join(re.findall("[1-9]",str(payout_hist["sanrentan_combo"] or "")))
            if histcombo!=winners or int(payout_hist["sanrentan_payout"])!=payout:
                audit["existing_payout_disagreement"]+=1;continue
            # Independently compare the two published line forecasts before
            # scoring a 3-2-2 race; grouping labels are arbitrary.
            def lineset(source_rows,bikekey,groupkey):
                groups=defaultdict(set)
                for x in source_rows:
                    lg=x[groupkey]
                    if lg is not None and str(lg).strip():
                        groups[str(lg)].add(int(x[bikekey]))
                return {frozenset(g) for g in groups.values()}
            src_groups=lineset(es,"car_no","line_id")
            his_groups=lineset(prior,"bike_number","line_group")
            if sorted(map(len,src_groups),reverse=True)==[3,2,2] and (
                 src_groups!=his_groups):
                audit["existing_322_line_mismatch"]+=1;continue
            audit["existing_independent_verified"]+=1
            if n==7:eligible.append((rid,source_id,race["race_date"],payout,winners))
            continue
        bike_to_id={}
        matched_count=0
        try:
            for e in es:
                bike=int(e["car_no"])
                rider_id,matched=determine_rider_id(e,names,pref)
                bike_to_id[bike]=rider_id
                matched_count+=int(matched)
            if len(set(bike_to_id.values()))!=n:
                raise ValueError("Different riders assigned identical ID")
            if {int(r["car_no"]) for r in rs}!=set(bike_to_id):
                raise ValueError("Entry/result roster discrepancy")
        except ValueError:
            audit["missing_or_ambiguous_rider_identity"]+=1;continue
        # Race outcome already exists in source, but predictions always
        # use strictly earlier race dates to avoid leaking current/future outcomes.
        conn.execute("INSERT INTO races "
           "(race_id,date,velodrome,race_number,race_name,grade,round,rider_count,start_time) "
           "VALUES (?,?,?,?,?,?,?,?,?)",
           (rid,race["race_date"],str(race["track"]).removesuffix("競輪"),
            int(race["race_no"]),race["race_type"],"F1",race["race_type"],
            n,race["start_time"]))
        for e in es:
            bike=int(e["car_no"]);rider_id=bike_to_id[bike]
            if rider_id.startswith("h2test:"):
                name=clean_name(e["player_name"])
                age=f_int(e["age"])
                conn.execute("INSERT OR IGNORE INTO riders "
                    "(rider_id,name,prefecture,class,period,birth_year) VALUES (?,?,?,?,?,?)",
                    (rider_id,name,clean_name(e["prefecture"]),e["class"],
                     f_int(e["term"]),int(race["race_date"][:4])-age if age else None))
            role_pos=f_int(e["line_position"])
            role_map={1:"自力",2:"番手",3:"3番手"}
            # In ungrouped solo lines this remains a valid code, but
            # 3-2-2 requirement will be verified separately.
            role=role_map.get(role_pos,"自力")
            group=e["line_id"]
            if not group:group="solo_"+str(bike)
            conn.execute("INSERT INTO entries "
                "(race_id,rider_id,frame_number,bike_number,rider_name,class,prefecture,"
                "gear_ratio,win_rate,place_rate,avg_competition_score,line_group,line_role) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rid,rider_id,bike if n==7 else bike,bike,e["player_name"],
                 e["class"],e["prefecture"],num(e["gear"]),
                 num(e["win_rate"]),num(e["top2_rate"]),num(e["score"]),group,role))
        for r in rs:
            bike=int(r["car_no"]);rider_id=bike_to_id[bike]
            line_ent=next(e for e in es if int(e["car_no"])==bike)
            pos=f_int(line_ent["line_position"])
            conn.execute("INSERT INTO race_results "
                "(race_id,rider_id,finish_position,frame_number,bike_number,rider_name,"
                "class,prefecture,gear_ratio,last_1lap,line_group,line_role)"
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (rid,rider_id,f_int(r["finish_position"]),bike,bike,
                 r["player_name"],line_ent["class"],line_ent["prefecture"],
                 num(line_ent["gear"]),num(r["last200"]),
                 line_ent["line_id"] or "solo_"+str(bike),
                 {1:"自力",2:"番手",3:"3番手"}.get(pos,"自力")))
        conn.execute(
          "INSERT INTO race_payouts(race_id,sanrentan_combo,sanrentan_payout) VALUES (?,?,?)",
          (rid,winners,payout))
        audit["inserted_races"]+=1
        audit["matched_rider_appearances"]+=matched_count
        audit["unmatched_rider_appearances"]+=n-matched_count
        source_rids.add(rid)
        if n==7:eligible.append((rid,source_id,race["race_date"],payout,winners))
    conn.commit()
    audit["inserted_7car_races"]=len(eligible)
    return eligible,dict(audit)


def bootstrap_by_days(ledger):
    per=defaultdict(lambda:[0,0])
    for r in ledger:
        key=r["date"]
        per[key][0]+=100*r["points"]
        per[key][1]+=r["returned_yen"]
    if len(per)<3 or sum(z[0] for z in per.values())==0:return None
    a=np.asarray(list(per.values()),dtype=float)
    rng=np.random.default_rng(20261025)
    vals=[]
    for i in range(2500):
        s=a[rng.integers(0,len(a),len(a))].sum(axis=0)
        if s[0]>0:vals.append(100*s[1]/s[0])
    return [round(float(x),2) for x in np.quantile(vals,[0.025,.975])]


def summarize(ledger):
    races=len(ledger)
    active=sum(x["points"]>0 for x in ledger)
    hits=sum(x["hit"] for x in ledger)
    points=sum(x["points"] for x in ledger)
    invested=100*points
    returns=sum(x["returned_yen"] for x in ledger)
    prizes=[x["returned_yen"] for x in ledger if x["hit"]]
    return {"races":races,"betting_races":active,"hits":hits,
            "hit_rate_pct":round(100*hits/races,3) if races else None,
            "avg_points_per_race":round(points/races,3) if races else None,
            "points":points,"invested_yen":invested,"returned_yen":returns,
            "net_yen":returns-invested,
            "roi_pct":round(100*returns/invested,3) if invested else None,
            "hit_payout_median_yen":float(median(prizes)) if prizes else None,
            "hit_payout_mean_yen":round(mean(prizes),2) if prizes else None,
            "break_even_avg_payout_per_hit_yen":round(invested/hits,2) if hits else None,
            "top1_payout_removed_roi_pct":round(100*(returns-max(prizes))/invested,3)
                 if invested and prizes else None,
            "ci95_day_bootstrap":bootstrap_by_days(ledger)}


def audit_policy(policy):
    if not policy["research_only"] or policy["selection_uses_period_for_future_test"]:
        raise ValueError("Frozen policy not research-only or future data contaminates selection")
    if policy["policy_id"]!="URA_322_HIGH_FIRST_4_20_FROZEN_20261025":
        raise ValueError("Unexpected policy identifier")
    arr=policy["LOW_types"]
    if {k:len(v) for k,v in arr.items()}!=REQUIRED:
        raise ValueError("Fixed 0/4/0/20 LOW scheme modified")
    for v in arr.values():
        if len(set(v))!=len(v):raise ValueError("Duplicate or changed shape")
    if policy["period_for_future_test"]!={"start":BEGIN,"end_exclusive":END}:
        raise ValueError("Wrong evaluation time range")
    return arr


def evaluate(model_file,policy_file,source_dir,out_dir):
    if hashlib.sha256(Path(model_file).read_bytes()).hexdigest()!=BASELINE_SHA:
        raise RuntimeError("Main HaruQ ranker not frozen")
    if BASELINE_SHA!=FROZEN_HARUQ_MODEL_SHA256:
        raise RuntimeError("Baseline different from internal frozen audit")
    frozen_artifact("artifacts/ura_ai_v1.json")
    base=load_fixed("artifacts/ura_ai_v1.json")
    policy=json.loads(Path(policy_file).read_text(encoding="utf8"))
    by= audit_policy(policy)
    ranker=LGBMRanker();ranker.load(str(model_file))
    feature_builder=FeatureBuilder()
    assessment=Counter()
    ledger=[]
    with get_connection() as conn:
        future,audit=import_source(conn,source_dir)
        # Earlier known test race results are valid for later July-Aug races,
        # but test-period results are never used to refit models or the policy.
        for rid,source_id,day,payout,winner in future:
            if "2026-07-01"<=day<"2026-08-31": pass
            else:raise ValueError("Outside frozen evaluation time period")
            assessment["source_7car_auditable"]+=1
            order=[r[0] for r in conn.execute(
                 "SELECT bike_number FROM entries WHERE race_id=? ORDER BY bike_number",(rid,))]
            lookup,reason=load_322(conn,rid,order)
            assessment["line_"+reason]+=1
            if lookup is None:continue
            assessment["eligible_322_before_low"]+=1
            try:
                raw=feature_snapshot(feature_builder,ranker,conn,rid,day)
                if {int(q["bike_number"]) for q in raw}!=set(order):
                    assessment["feature_roster_mismatch"]+=1;continue
                ranked,features,gap,_=prepare(raw)
                if not math.isfinite(gap) or gap<0:raise ValueError("bad LOW gap")
            except Exception as exc:
                assessment["model_feature_failure"]+=1
                if assessment["model_feature_failure"]<=4:
                    print("MODEL_FAILED",rid,type(exc).__name__,str(exc)[:150],flush=True)
                continue
            if gap>=1:
                assessment["not_low"]+=1;continue
            assessment["eligible_low_322"]+=1
            interval=band(gap)
            labels=by[interval]
            # The exact original policy applies frozen URA linear utility to
            # HaruQ-score-ranked features, then sorts URA descending.
            utility=features@np.asarray(base["parameters"],dtype=float)
            chosen_order=sorted(range(7),key=lambda i:(-utility[i],int(ranked[i]["bike_number"])))
            rank_map={int(ranked[i]["bike_number"]):j+1 for j,i in enumerate(chosen_order)}
            actual=tuple(map(int,winner.split("-")))
            winner_eligible=any(rank_map[b]>=low_rank_start(7) for b in actual)
            wanted=set(labels)
            ticket_count=0
            win=winner_eligible and structural_key(actual,lookup) in wanted
            for triplet in itertools.permutations(rank_map,3):
                if not any(rank_map[x]>=5 for x in triplet):continue
                if structural_key(triplet,lookup) in wanted:ticket_count+=1
            if interval in ("0.00-0.25","0.50-0.75") and (ticket_count or win):
                raise RuntimeError("Paused LOW was accidentally purchased")
            if win and ticket_count<=0:raise RuntimeError("Winning ticket not purchased")
            ledger.append({
                "race_id":rid,"archive_race_id":source_id,
                "date":day,"period":day[:7],"LOW_band":interval,
                "gap":round(gap,8),
                "selected_type_count":len(wanted),
                "points":ticket_count,"invested_yen":100*ticket_count,
                "hit":int(win),"returned_yen":payout if win else 0,
                "actual_combo":winner,"official_3rentan_payout_yen":payout,
                "actual_shape":structural_key(actual,lookup),
                "winner_rank_has_5to7":int(winner_eligible)
            })
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    with (out/"per_race_fixed_policy_ledger.csv").open("w",encoding="utf8",newline="") as f:
        if ledger:
            w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
    slices=[]
    for period in ("2026-07","2026-08","2026-07_to_08"):
        for interval in ("0.25-0.50","0.75-1.00","ALL"):
            l=[x for x in ledger if (x["period"]==period or period.endswith("to_08")) and
               (x["LOW_band"]==interval or interval=="ALL")]
            slices.append({"period":period,"LOW_band":interval,**summarize(l)})
    with (out/"score_by_period_and_low.csv").open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(slices[0]));w.writeheader();w.writerows(slices)
    flags=["UNKNOWN_PRESTART_FEATURE_SNAPSHOT","HISTORICAL_SOURCE_CAPTURED_AFTER_RACES",
           "TEST_IS_S_CLASS_F1_ONLY","MISSING_2026_PRIOR_RACES_POSSIBLE",
           "HISTORICAL_ARCHIVE_RIDER_ID_MATCH_AUDIT_REQUIRED"]
    manifest={
        "status":"scored" if ledger else "insufficient_exact_cohort_matches",
        "frozen_4_20_policy_sha256":hashlib.sha256(Path(policy_file).read_bytes()).hexdigest(),
        "frozen_haruq_sha256":BASELINE_SHA,"frozen_ura_blob_sha":URA_BLOB,
        "period":[BEGIN,END],"source":ARCHIVE_SOURCE+"/"+SOURCE_PATH,
        "selected_from":["2023","2024","retro_2025_2026H1"],
        "future_results_used_for_rule_selection":False,
        "fixed_policy":policy["LOW_types"],
        "research_only":True,"purchase_approved":False,
        "existing_web_or_ai_modified":False,
        "source_import_audit":audit,"prediction_audit":dict(assessment),
        "metrics":slices,"limitations":flags
    }
    (out/"future_cohort_audit.json").write_text(
       json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("FUTURE_4_20_SCORING",json.dumps({
      "source":audit,"prediction":dict(assessment),
      "scores":slices,"policy_sha":manifest["frozen_4_20_policy_sha256"],
    },ensure_ascii=False),flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True,type=Path)
    p.add_argument("--policy",required=True,type=Path)
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    evaluate(a.model,a.policy,a.source,a.out)


if __name__=="__main__":
    main()
