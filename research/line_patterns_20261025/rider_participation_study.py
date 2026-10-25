"""Research-only rider attribution on 2026 Jul-Aug fixed LOW 4/20 test.

Question: do the originally low-ranked riders (URA ranks 5..7) who finish
top-three show repeatability, starts, historical top-3 production, role?
NO ticket changes, NO selection based on contemporaneous outcomes.
Source: audited scored per-race ledger + original same-race entries in db;
prior statistics only from result dates STRICTLY before race date.
"""
from __future__ import annotations
import argparse,csv,json,itertools
from collections import Counter,defaultdict
from pathlib import Path
from statistics import mean,median

from db.schema import get_connection
from research.line_patterns_20261025.geometry_322 import load_322,structural_key

BANDS=("0.00-0.25","0.25-0.50","0.50-0.75","0.75-1.00")
ACTIVE=("0.25-0.50","0.75-1.00")


def read_rows(path):
    with Path(path).open(encoding="utf8",newline="") as f:
        return list(csv.DictReader(f))


def past_only_stats(conn,rider_id,day):
    q=conn.execute("""
        SELECT COUNT(*) starts,
               COALESCE(SUM(CASE WHEN rr.finish_position <=3 THEN 1 ELSE 0 END),0) top3,
               COALESCE(SUM(CASE WHEN rr.finish_position =1 THEN 1 ELSE 0 END),0) wins
        FROM race_results rr JOIN races r ON r.race_id=rr.race_id
        WHERE rr.rider_id=? AND r.date<? AND r.date >= date(?, '-180 day')
          AND rr.finish_position IS NOT NULL
        """,(rider_id,day,day)).fetchone()
    return int(q["starts"]),int(q["top3"]),int(q["wins"])


def past_starts_band(v):
    if v<5:return "00-04"
    if v<10:return "05-09"
    if v<20:return "10-19"
    return "20+"


def evidence_group(records,kind,reader):
    grouped=defaultdict(list)
    for r in records:grouped[reader(r)].append(r)
    output=[]
    for group,rs in sorted(grouped.items()):
        starts=len(rs)
        got=sum(x["in_top3"] for x in rs)
        lead=sum(x["in_1st"] for x in rs)
        second=sum(x["in_2nd"] for x in rs)
        third=sum(x["in_3rd"] for x in rs)
        output.append({"dimension":kind,"group":group,"starts":starts,
                       "top3_finishes":got,"top3_rate_pct":round(100*got/starts,3),
                       "firsts":lead,"seconds":second,"thirds":third,
                       "eligible_lower_rank_starts":sum(x["ura_rank"]>=5 for x in rs),
                       "avg_prior_180day_starts":round(mean(x["prior180_starts"] for x in rs),3),
                       "avg_prior_180day_top3_rate_pct":round(
                          mean(100*x["prior180_top3"]/x["prior180_starts"]
                               for x in rs if x["prior180_starts"]>0),3)
                               if any(x["prior180_starts"]>0 for x in rs) else None})
    return output


def rider_frequency(records):
    by_rider=defaultdict(list)
    for x in records:by_rider[x["rider_id"]].append(x)
    out=[]
    for rid,rs in sorted(by_rider.items()):
        starts=len(rs)
        got=sum(x["in_top3"] for x in rs)
        by_era=Counter(x["date"][:7] for x in rs)
        out.append({"rider_id":rid,"rider_name":rs[0]["rider_name"],
                    "prefecture":rs[0]["prefecture"],
                    "starts_in_exact_LOW_322_sample":starts,
                    "times_in_top3":got,
                    "top3_pct":round(100*got/starts,3),
                    "wins":sum(x["in_1st"] for x in rs),
                    "second":sum(x["in_2nd"] for x in rs),
                    "third":sum(x["in_3rd"] for x in rs),
                    "rank5to7_starts":sum(x["ura_rank"]>=5 for x in rs),
                    "rank5to7_top3":sum(x["ura_rank"]>=5 and x["in_top3"] for x in rs),
                    "rank5to7_top3_pct":round(100*sum(x["ura_rank"]>=5 and x["in_top3"] for x in rs)/
                                               sum(x["ura_rank"]>=5 for x in rs),3)
                        if any(x["ura_rank"]>=5 for x in rs) else None,
                    "mean_prior180_starts_at_entry":round(mean(x["prior180_starts"] for x in rs),3),
                    "mean_prior180_top3_at_entry":round(mean(x["prior180_top3"] for x in rs),3),
                    "july_starts":by_era["2026-07"],"august_starts":by_era["2026-08"]})
    return sorted(out,key=lambda x:(-x["rank5to7_top3"],-x["times_in_top3"],
                                     -x["starts_in_exact_LOW_322_sample"],x["rider_id"]))


def miss_reasons(races):
    counts=Counter()
    hits=[]
    for r in races:
        counts["audited_low_322_races"]+=1
        counts["purchased_races"]+=r["points"]>0
        counts["purchased_points"]+=r["points"]
        if r["LOW_band"] not in ACTIVE:
            counts["paused_low_band_races"]+=1
            continue
        counts["active_low_band_races"]+=1
        if r["hit"]:
            counts["matched_eligible_win"]+=1
            hits.append(r)
            continue
        if not r["winner_rank_has_5to7"]:
            counts["excluded_due_to_no_lower_rank_in_podium"]+=1
        else:
            counts["missing_geometry_but_eligible_lower_rank_podium"]+=1
    if counts["active_low_band_races"] != (
        counts["matched_eligible_win"]+
        counts["excluded_due_to_no_lower_rank_in_podium"]+
        counts["missing_geometry_but_eligible_lower_rank_podium"]):
        raise RuntimeError("Miss categories not exhaustive")
    return dict(counts),hits


def study(ledger_file,outdir):
    original=read_rows(ledger_file)
    if not original:raise RuntimeError("No audited ledger")
    if any("ura_ranked_bikes" not in r for r in original):
        raise RuntimeError("Missing frozen rank snapshots; cannot infer URA order from outcome")
    expected_unique=len({r["race_id"] for r in original})
    if expected_unique!=len(original):raise RuntimeError("Duplicate trial races")
    race_rows=[];rider_rows=[];audited=Counter()
    with get_connection() as conn:
        for r in original:
            day=r["date"];rid=r["race_id"]
            ranked=[int(n) for n in r["ura_ranked_bikes"].split("-")]
            if len(ranked)!=7 or len(set(ranked))!=7:raise ValueError("Bad frozen rank roster")
            rank={bike:i+1 for i,bike in enumerate(ranked)}
            lines,status=load_322(conn,rid,ranked)
            if status!="ok":raise RuntimeError("Line no longer matches policy ledger "+rid)
            entries=conn.execute(
                "SELECT rider_id,rider_name,bike_number,prefecture FROM entries WHERE race_id=?",
                (rid,)).fetchall()
            if len(entries)!=7:raise ValueError("Rider roster changed")
            actual=[int(n) for n in r["actual_combo"].split("-")]
            if len(actual)!=3 or len(set(actual))!=3 or any(x not in rank for x in actual):
                raise ValueError("Bad outcome")
            if structural_key(actual,lines)!=r["actual_shape"]:raise RuntimeError("Actual shape differs")
            podium={b:i for i,b in enumerate(actual,1)}
            active=r["LOW_band"] in ACTIVE
            purchased=int(r["points"])
            lower_any=any(rank[b]>=5 for b in actual)
            if bool(int(r["winner_rank_has_5to7"]))!=lower_any:raise RuntimeError("LOW label mismatch")
            if bool(int(r["hit"])) and (not lower_any or not active):
                raise RuntimeError("Winner violates fixed precommitted LOW policy")
            for e in entries:
                car=int(e["bike_number"]);pos=podium.get(car,0)
                prior,prior_top3,prior_wins=past_only_stats(conn,e["rider_id"],day)
                role=lines[car][1]
                rider_rows.append({
                    "date":day,"race_id":rid,"rider_id":str(e["rider_id"]),
                    "rider_name":e["rider_name"],"prefecture":e["prefecture"],
                    "bike_number":car,"LOW_band":r["LOW_band"],
                    "active_band":int(active),
                    "ura_rank":rank[car],
                    "line_role":role,
                    "finished_position":pos or None,
                    "in_top3":int(pos>0),
                    "in_1st":int(pos==1),"in_2nd":int(pos==2),"in_3rd":int(pos==3),
                    "prior180_starts":prior,"prior180_top3":prior_top3,
                    "prior180_wins":prior_wins,
                    "prior180_top3_pct":round(100*prior_top3/prior,3) if prior else None,
                    "prior180_starts_band":past_starts_band(prior),
                    "race_purchased_points":purchased,
                    "race_hit":int(r["hit"]),
                    "winning_race_payout_yen":int(r["official_3rentan_payout_yen"]) if pos else 0})
            race_rows.append({
                "race_id":rid,"date":day,"LOW_band":r["LOW_band"],
                "active":int(active),"points":purchased,"hit":int(r["hit"]),
                "invested":int(r["invested_yen"]),
                "paid":int(r["returned_yen"]),
                "lower_rank_podium":int(lower_any),
                "lower_rank_1st":int(rank[actual[0]]>=5),
                "lower_rank_2nd":int(rank[actual[1]]>=5),
                "lower_rank_3rd":int(rank[actual[2]]>=5),
                "lower_rank_podium_count":sum(rank[v]>=5 for v in actual),
                "winner_actual_geometry":r["actual_shape"],
                "rank_1st":rank[actual[0]],
                "rank_2nd":rank[actual[1]],
                "rank_3rd":rank[actual[2]],
                "payout":int(r["official_3rentan_payout_yen"])
            })
            audited["processed_races"]+=1
    if len(rider_rows)!=7*len(race_rows):raise RuntimeError("Seven-rider exposure mismatch")
    overview=[]
    for key,subset in (
        ("ALL",rider_rows),
        ("ACTIVE_LOW", [r for r in rider_rows if r["active_band"]]),
        ("PAUSED_LOW", [r for r in rider_rows if not r["active_band"]]),
        ("ACTIVE_LOW_RANK5TO7", [r for r in rider_rows if r["active_band"] and r["ura_rank"]>=5]),
        ("ACTIVE_RANK1TO4",[r for r in rider_rows if r["active_band"] and r["ura_rank"]<5]),
        ("PAUSED_LOW_RANK5TO7", [r for r in rider_rows if not r["active_band"] and r["ura_rank"]>=5]),
    ):
        if subset:
            overview.extend(evidence_group(subset,key+":URA_RANK",lambda x:str(x["ura_rank"])))
            overview.extend(evidence_group(subset,key+":LINE_ROLE",lambda x:x["line_role"]))
            overview.extend(evidence_group(subset,key+":PRIOR_180_STARTS",lambda x:x["prior180_starts_band"]))
            overview.extend(evidence_group(subset,key+":PRIOR_180_TOP3_BIN",
                lambda x:"none" if not x["prior180_starts"] else (
                    "0-19%" if x["prior180_top3"]/x["prior180_starts"]<.2 else
                    "20-39%" if x["prior180_top3"]/x["prior180_starts"]<.4 else "40%+")))
    reason,hits=miss_reasons(race_rows)
    rank_summary=[r for r in overview if r["dimension"]=="ACTIVE_LOW:URA_RANK"]
    reasons_by_band=[]
    for band in BANDS:
        groups=[r for r in race_rows if r["LOW_band"]==band]
        tmp,_=miss_reasons(groups)
        reasons_by_band.append({"LOW_band":band,**tmp})
    rider_profiles=rider_frequency(rider_rows)
    multi=[r for r in rider_profiles if r["starts_in_exact_LOW_322_sample"]>=3]
    low_repeats=[r for r in multi if r["rank5to7_top3"]>=2]
    out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
    for file,rows in (
        ("rider_race_exposure_and_prestart_form.csv",rider_rows),
        ("rider_repeated_top3_profiles.csv",rider_profiles),
        ("LOW_by_role_rank_prior180_exposure.csv",overview),
        ("miss_reason_by_low.csv",reasons_by_band),
        ("winning_races_role_and_rank.csv",[r for r in race_rows if r["hit"]]),
    ):
        if not rows:continue
        with (out/file).open("w",newline="",encoding="utf8") as f:
            fields=list(dict.fromkeys(k for row in rows for k in row))
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    manifest={
        "name":"URA_HARUQ_RIDER_EXPOSURE_AND_REPEATED_PODIUM_DIAGNOSTIC",
        "research_only":True,"purchase_approved":False,
        "fixed_models_and_4_20_policy_unchanged":True,
        "dates":["2026-07-01","2026-08-30"],
        "race_count":len(race_rows),"rider_exposure_count":len(rider_rows),
        "unique_riders":len(rider_profiles),
        "riders_with_3plus_sample_starts":len(multi),
        "riders_with_3plus_starts_2plus_low_rank_top3":len(low_repeats),
        "miss_reasons":reason,
        "by_LOW":reasons_by_band,
        "rank_metrics_active_low":rank_summary,
        "active_low_races":sum(r["active"] for r in race_rows),
        "paired_prestart_window":"[race_date -180 days, race_date), from historical results strictly before race date",
        "important_limits":[
           "Only 299 audited low 3-2-2 S-class F1 races, not all keirin",
           "Earlier rider histories may be incomplete or contain biased race coverage",
           "Raw entries were downloaded after race dates, so not a timestamp-certified pre-start snapshot",
           "This is descriptive rider attribution only; no selected riders or new policy",
           "Repeated top-three counts MUST be interpreted conditional on exposure, not raw frequency",
           "No player identities are used to reselect previously locked 4/20 and no use of payout in buying"
        ],
    }
    (out/"rider_diagnostic_audit.json").write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print("RIDER_LOW_ATTRIBUTION",json.dumps({
        "races":len(race_rows),"riders":len(rider_profiles),
        "reason":reason,"bands":reasons_by_band,
        "active_rank":rank_summary,"repeat_rider":low_repeats[:20],
    },ensure_ascii=False),flush=True)
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ledger",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    x=p.parse_args()
    study(x.ledger,x.out)


if __name__=="__main__":
    main()
