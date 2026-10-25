"""Audit 2026 out-of-time return for three 2025-locked rider formation strategies.

No re-ranking by achieved ROI; present originally precommitted winner first.
Jan-Sep 2026 whole months; Oct available days only and separately.
"""
import argparse, csv, json, collections, pathlib
from pathlib import Path

def lines(path):
    with open(path,encoding="utf8",newline="") as f:return list(csv.DictReader(f))

def summarize(rows):
    z=collections.Counter()
    for r in rows:
        z["races"]+=1
        z["tickets"]+=int(r["tickets_per_race"])
        z["hits"]+=int(r["hit"])
        z["stake_yen"]+=int(r["stake_yen"])
        z["return_yen"]+=int(r["return_yen"])
        z["return_capped_100000"]+=min(100000,int(r["return_yen"]))
    z["profit_yen"]=z["return_yen"]-z["stake_yen"]
    if z["stake_yen"]:
        z["roi_pct"]=round(100*z["return_yen"]/z["stake_yen"],4)
        z["capped_roi_pct"]=round(100*z["return_capped_100000"]/z["stake_yen"],4)
    return dict(z)

def main():
    p=argparse.ArgumentParser();p.add_argument("--input",required=True);p.add_argument("--out",required=True)
    a=p.parse_args()
    source=Path(a.input);files=sorted(source.glob("*/races_2026_THREE_2025_FROZEN_STRATEGIES.csv"))
    if len(files)!=10:raise RuntimeError(f"Require exactly 10 month artifacts, found {len(files)}")
    allrows=[];month_audit=[];allmonths=[]
    for f in files:
        sub=lines(f);months={r["date"][:7] for r in sub}
        if len(months)!=1:raise RuntimeError("Month shard inconsistent: "+str(months))
        audit=json.loads((f.parent/"2026_3WAY_MONTHLY_VALIDATION_AUDIT.json").read_text())
        if not audit["from_2025_validation_only"] or audit["2026_tuning"] or audit["production_modified"]:
            raise RuntimeError("Leak or production mutation found")
        if audit["2026_all_method_rows"]!=len(sub):
            raise RuntimeError("Accountant mismatch "+str(f))
        if audit["2026_races_scored"]*3!=len(sub):
            raise RuntimeError("Missing one comparator per race "+str(f))
        allmonths.extend(months);allrows+=sub;month_audit.append({"month":next(iter(months)),
            "reconstructed_original_races":audit["2026_source_races"],
            "offically_settled_evaluable":audit["2026_races_scored"],
            "excluded":audit["2026_census"]})
    if sorted(allmonths)!=[f"2026-{i:02d}" for i in range(1,11)]:
        raise RuntimeError("Missing 2026 month or duplicate source!")
    allIDs=collections.Counter((r["date"],r["race_id"],r["rider_weight"]) for r in allrows)
    if any(n!=1 for n in allIDs.values()):raise RuntimeError("Duplicate 2026 race + approach")
    scenarios=sorted(set(r["approach"] for r in allrows))
    if len(scenarios)!=3:raise RuntimeError("Missing pre-2026 controls")
    report=[]
    for scope,choose in (
        ("JAN_SEP_PRIMARY",lambda x:x["date"]<"2026-10-01"),
        ("OCT_PARTIAL_SEPARATE",lambda x:x["date"]>="2026-10-01"),
        ("JAN_OCT_ARCHIVE_TO_DATE",lambda x:True)):
        for label in scenarios:
            r=[x for x in allrows if x["approach"]==label and choose(x)]
            report.append({"period":scope,"approach":label,**summarize(r)})
            for klass in ("S級","A級1・2班","チャレンジA3"):
                report.append({"period":scope,"approach":label,"class":klass,**summarize(
                    [x for x in r if x["category"]==klass])})
    bymonth=[]
    for month in sorted(set(r["date"][:7] for r in allrows)):
        for label in scenarios:
            sub=[r for r in allrows if r["date"][:7]==month and r["approach"]==label]
            bymonth.append({"month":month,"approach":label,**summarize(sub)})
    # Compare same race of w=0 vs w=.7 and 1.5; those have
    # distinct 2025-selected K. Never compare hit counts without stake.
    selected={}
    for row in allrows:
        if row["date"]<"2026-10-01":
            selected.setdefault(row["race_id"],{})[row["rider_weight"]]=row
    delta={}
    for strength in ("0.7","1.5"):
        differences=[]
        for r,m in selected.items():
            if set(m)!={"0.0","0.7","1.5"}:raise RuntimeError("Missing comparator case")
            b=m["0.0"];p=m[strength]
            differences.append((int(p["return_yen"])-int(p["stake_yen"]))-
                               (int(b["return_yen"])-int(b["stake_yen"])))
        delta[strength]={"compared_same_races":len(differences),
                         "total_profit_diff_vs_no_player_yen":sum(differences),
                         "monthly_net_advantages_yen":{
                             m:sum((int(r["return_yen"])-int(r["stake_yen"]))-
                                   (int(selected[r["race_id"]]["0.0"]["return_yen"])-
                                    int(selected[r["race_id"]]["0.0"]["stake_yen"]))
                                   for r in allrows if r["rider_weight"]==strength
                                   and r["date"][:7]==m)
                             for m in sorted({x["date"][:7] for x in allrows if x["date"]<"2026-10-01"})}}
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for path,records in (
      ("2026_3strategy_monthly_roi.csv",bymonth),
      ("2026_3strategy_JANSEP_OCT_roi_by_class.csv",report),
      ("2026_3strategy_all_selected_2026_tickets.csv",allrows)):
        with (out/path).open("w",encoding="utf8",newline="") as f:
            keys=list(dict.fromkeys(k for row in records for k in row))
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
    capsule={"research_only":True,"production_untouched":True,
      "selection_horizon_2025":True,"2026_used_only_for_validation":True,
      "original_2026_month_coverage":month_audit,
      "primary_2026_full_months_Jan_Sep":[r for r in report if r["period"]=="JAN_SEP_PRIMARY" and "class" not in r],
      "oct_partial":[r for r in report if r["period"]=="OCT_PARTIAL_SEPARATE" and "class" not in r],
      "sensitivity_2026_monthly":bymonth,
      "same_race_net_profit_comparison":delta,
      "limitations":["2026 model originated as frozen March 2026 HaruQ model and URA calibration may have consulted 2026 holdout; not pristine model OOS.",
      "2025 winner selected after trying multiple combinations on 2025 Q4; winner's curse possible",
      "Archived payout 100JPY tickets assume official settlement; pre-deadline tradability not established",
      "No current market odds included; this is realized ROI backtest only",
      "Source DB 2026 only through archive collection date, October incomplete."]}
    (out/"2026_COMPARATIVE_ROI_AUDIT.json").write_text(json.dumps(capsule,indent=2,ensure_ascii=False)+"\n")
    print("2026_FULL_RIDER_FORMATION_FINAL",json.dumps({
        "primary":capsule["primary_2026_full_months_Jan_Sep"],
        "oct":capsule["oct_partial"],"monthly":bymonth,"same_race":delta,
        "archive_months":len(month_audit)},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
