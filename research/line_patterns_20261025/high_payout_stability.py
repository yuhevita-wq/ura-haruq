"""Auditable 3-2-2 LOW-band payout/frequency/point/stability gate + combo solver.

Pre-set methodology (before reading new evaluation):
* Training 2023 payout cutoffs by LOW are the 2023 MEDIAN of actual
  |12|23|13 3連単 official payouts (including races with no eligible
  underdog rider). No hardcoded yen cutoff, no access to evaluation years.
* An individual type passes a band if in BOTH 2023 and 2024 it had >=3
  eligible real hits; ALL real wins' median payout >= 2023 reference cutoff;
  and the 2024/2023 eligible hit-frequency ratio is in [0.5,2.0].
* Only after these gates do exact two-stage 0/1 MILP portfolio learners
  optimize total net or breadth of coverage, without max type count.
* 2025 + 2026 held out from all filtering, thresholds, and optimization.
* Cheap realized outcomes are NOT deleted after races: ALL actual
  winning payouts, including cheap outcomes in approved types, enter ROI.
* Existing fixed 2-2-2-1, main HaruQ, original URA V1 not modified.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from db.schema import get_connection
from research.line_patterns_20261025.geometry_322 import BANDS, get_rows, load_322
from research.line_patterns_20261025.low_portfolio import (
    load_2026, make_record, as_years, aggregate
)
from research.line_patterns_20261025.optimal_combinations import (
    solve, VARIANTS, PERIODS, PERMITTED
)

MIN_HITS_PER_TRAIN_YEAR=3
MIN_FREQ_RATIO=0.5
MAX_FREQ_RATIO=2.0
SEED=20261025


def types_of_pair_group(cohort):
    return sorted({name for r in cohort
                   for name,n in r["tickets"].items()
                   if n>0 and name.rsplit("|",1)[-1] in PERMITTED})


def annual_profile(cohort,shape):
    n=len(cohort)
    outcomes=[int(r["payout"]) for r in cohort
              if r["observed_structure"]==shape]
    eligible=[int(r["payout"]) for r in cohort
              if r["winner_type"]==shape]
    points=[int(r["tickets"][shape]) for r in cohort]
    investment=100*sum(points)
    paid=sum(eligible)
    def med(values):
        return round(float(np.median(values)),2) if values else None
    return {"races":n,
            "observed_wins_all":len(outcomes),
            "observed_frequency_pct":round(100*len(outcomes)/n,4) if n else None,
            "eligible_hits":len(eligible),
            "eligible_frequency_pct":round(100*len(eligible)/n,4) if n else None,
            "payout_median_all_yen":med(outcomes),
            "payout_q25_all_yen":round(float(np.quantile(outcomes,.25)),2) if outcomes else None,
            "payout_median_eligible_yen":med(eligible),
            "required_total_points":sum(points),
            "required_mean_points_per_race":round(float(np.mean(points)),4) if n else None,
            "required_median_points_per_race":med(points),
            "required_max_points_per_race":max(points,default=0),
            "investment_yen":investment,"paid_yen":paid,"net_yen":paid-investment,
            "roi_pct":round(100*paid/investment,3) if investment else None}


def payout_reference_2023(cohort):
    payouts=[int(r["payout"]) for r in cohort
             if r["observed_structure"].rsplit("|",1)[-1] in PERMITTED]
    if not payouts:raise RuntimeError("No real 2023 payouts in target family")
    return {"cutoff_yen":float(np.median(payouts)),
            "reference_payout_n":len(payouts),
            "reference_q25_yen":float(np.quantile(payouts,.25)),
            "reference_q75_yen":float(np.quantile(payouts,.75))}


def frequency_ratio(a,b):
    """Rates per race not raw hits: survives differing year sample sizes."""
    r1=a["eligible_hits"]/a["races"] if a["races"] else 0
    r2=b["eligible_hits"]/b["races"] if b["races"] else 0
    return round(r2/r1,4) if r1>0 else None


def profile_and_gate(cohort23,cohort24):
    baseline=payout_reference_2023(cohort23)
    cutoff=baseline["cutoff_yen"]
    types=sorted(set(types_of_pair_group(cohort23))|set(types_of_pair_group(cohort24)))
    catalog=[];survivors=[]
    for shape in types:
        yr23=annual_profile(cohort23,shape)
        yr24=annual_profile(cohort24,shape)
        ratio=frequency_ratio(yr23,yr24)
        reasons=[]
        if yr23["eligible_hits"]<MIN_HITS_PER_TRAIN_YEAR:
            reasons.append("2023_fewer_than_3_eligible_hits")
        if yr24["eligible_hits"]<MIN_HITS_PER_TRAIN_YEAR:
            reasons.append("2024_fewer_than_3_eligible_hits")
        # Actual full outcome medians estimate payout level without
        # conflating it with the underdog-eligibility screen.
        if yr23["payout_median_all_yen"] is None or yr23["payout_median_all_yen"]<cutoff:
            reasons.append("2023_low_actual_median_payout")
        if yr24["payout_median_all_yen"] is None or yr24["payout_median_all_yen"]<cutoff:
            reasons.append("2024_low_actual_median_payout")
        if ratio is None or not MIN_FREQ_RATIO<=ratio<=MAX_FREQ_RATIO:
            reasons.append("hit_frequency_unstable_2023_to_2024")
        accepted=not reasons
        combined={"type":shape,
                  "line_relation":shape.rsplit("|",1)[-1],
                  "cutoff_2023_yen":cutoff,
                  "eligible_2023":yr23["eligible_hits"],
                  "eligible_2024":yr24["eligible_hits"],
                  "ratio_eligible_frequency_2024_to_2023":ratio,
                  "payout_median_2023":yr23["payout_median_all_yen"],
                  "payout_median_2024":yr24["payout_median_all_yen"],
                  "points_per_race_2023":yr23["required_mean_points_per_race"],
                  "points_per_race_2024":yr24["required_mean_points_per_race"],
                  "accepted":accepted,"exclusion_reasons":";".join(reasons),
                  "2023_net_yen":yr23["net_yen"],"2024_net_yen":yr24["net_yen"],
                  "2023_hits":yr23["eligible_hits"],"2024_hits":yr24["eligible_hits"],
                  "2023_points":yr23["required_total_points"],
                  "2024_points":yr24["required_total_points"],
                  "2023_roi_pct":yr23["roi_pct"],"2024_roi_pct":yr24["roi_pct"]}
        catalog.append(combined)
        if accepted:survivors.append(combined)
    return baseline,catalog,survivors


def full_annual_catalog(by,types):
    """ALL six dozen types x 4 LOW bands x every available period."""
    rows=[]
    for band in BANDS:
        for shape in types:
            for year in PERIODS:
                profile=annual_profile(by[(year,band)],shape)
                rows.append({"LOW_band":band,"type":shape,
                             "relation":shape.rsplit("|",1)[-1],
                             "year":year,**profile})
    return rows


def portfolio_summary_by_band(by,choices):
    results=[];detail=[]
    for variant in (*VARIANTS,"unfiltered_profit","unfiltered_coverage"):
        for year in PERIODS:
            total_investment=0;total_paid=0;total_hits=0;total_points=0;total_races=0
            for interval in BANDS:
                chosen=choices[interval][variant]
                members=by[(year,interval)]
                out=aggregate(members,chosen)
                out.pop("days")
                results.append({"strategy":variant,"year":year,"LOW_band":interval,
                                "selected_type_count":len(chosen),
                                "selected_types":";".join(chosen),**out})
                total_investment+=out["investment_yen"]
                total_paid+=out["payout_yen"]
                total_hits+=out["hits"]
                total_points+=out["points"]
                total_races+=out["races"]
                for r in members:
                    purchased=sum(r["tickets"][x] for x in chosen)
                    hit=int(r["winner_type"] in chosen)
                    detail.append({"race_id":r["race_id"],"date":r["date"],
                                   "LOW_band":interval,"year":year,
                                   "strategy":variant,"winning_type":r["observed_structure"],
                                   "payout_yen":r["payout"],"tickets":purchased,"hit":hit,
                                   "stake_yen":100*purchased,
                                   "return_yen":hit*r["payout"]})
            results.append({"strategy":variant,"year":year,"LOW_band":"ALL",
                            "selected_type_count":sum(len(choices[b][variant]) for b in BANDS),
                            "selected_types":"PER-LOW",
                            "races":total_races,"active_races":None,"points":total_points,
                            "hits":total_hits,"investment_yen":total_investment,
                            "payout_yen":total_paid,"net_yen":total_paid-total_investment,
                            "roi_pct":round(100*total_paid/total_investment,3)
                                      if total_investment else None,
                            "hit_per_race_pct":round(100*total_hits/total_races,3)
                                                if total_races else None})
    return results,detail


def write_csv(dest,items):
    if not items:return
    with (dest).open("w",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(items[0]))
        w.writeheader();w.writerows(items)


def run(inputs,inputs26,out):
    past=get_rows(inputs)
    year26=load_2026(inputs26)
    audited=Counter();records=[]
    with get_connection() as conn:
        for row in itertools.chain(past,year26):
            if int(row["n"])!=7: continue
            bikes=[int(i) for i in row["ranked_bikes"].split("-")]
            lookup,status=load_322(conn,row["race_id"],bikes)
            audited[f"{row['year']}_{status}"]+=1
            if lookup is None:continue
            records.append(make_record(row,lookup))
    by=as_years(records)
    baselines={};catalog=[];policy={};choices={}
    for interval in BANDS:
        reference,inspected,filtered=profile_and_gate(
            by[("2023",interval)],by[("2024",interval)])
        baselines[interval]=reference
        catalog.extend([{"LOW_band":interval,**r} for r in inspected])
        # Optimize entire candidate subset, not individual ROI sorted ranking.
        policy[interval]={"reference":reference,
                          "discovered":len(inspected),
                          "survived":len(filtered),
                          "discarded_low_median":[r["type"] for r in inspected if
                                "low_actual_median_payout" in r["exclusion_reasons"]],
                          "filtered":{strategy:solve(filtered,strategy) for strategy in VARIANTS},
                          "unfiltered":{strategy:solve([
                              t for t in inspected if t["eligible_2023"]>=1 and
                              t["eligible_2024"]>=1],strategy)
                              for strategy in VARIANTS}}
        choices[interval]={
            **{strategy:policy[interval]["filtered"][strategy]["selected"] for strategy in VARIANTS},
            **{"unfiltered_"+strategy:policy[interval]["unfiltered"][strategy]["selected"]
               for strategy in VARIANTS}}
    results,detail=portfolio_summary_by_band(by,choices)
    all_types=sorted({r["type"] for r in catalog})
    # All 2025/2026 evaluation data only touched AFTER policy selection.
    full=full_annual_catalog(by,all_types)
    total={"name":"URA_322_LOW_PAYOUT_MEDIAN_AND_STABILITY_PORTFOLIO",
           "research_only":True,"purchase_approved":False,
           "frozen_haruq_untouched":True,"frozen_ura_untouched":True,
           "frozen_2221_untouched":True,
           "years_training":["2023","2024"],"years_evaluation":["2025","2026Q1","2026Q2"],
           "tested_structure_count":len(all_types),
           "LOW_2023_thresholds":baselines,
           "predeclared_filter":{"cutoff":"2023 median of ALL observed results in the three pair-line finish families, per LOW band",
                                 "min_eligible_hits_2023":MIN_HITS_PER_TRAIN_YEAR,
                                 "min_eligible_hits_2024":MIN_HITS_PER_TRAIN_YEAR,
                                 "median_each_training_year":"both 2023 and 2024 type payout medians must reach the same 2023 reference",
                                 "eligible_occurrence_frequency_ratio_2024_vs_2023":[MIN_FREQ_RATIO,MAX_FREQ_RATIO],
                                 "no_result_or_final_price_time_filter":True,
                                 "selection_uses_2025_or_2026":False},
           "optimized":{"profit":"exact 2023+2024 aggregate net maximum, each year profit>=0, tie -> fewer tickets",
                        "coverage":"exact 2023+2024 eligible hit count maximum, each year profit>=0, tie -> fewer tickets",
                        "max_types":None},
           "policies":policy,
           "data_audit":dict(audited),
           "cohort_races":{year:sum(r["split"]==year for r in records) for year in PERIODS},
           "comparison_ROI_all_low":[{"year":x["year"],"strategy":x["strategy"],
                                      "type_count":x["selected_type_count"],
                                      "hits":x["hits"],"points":x["points"],"investment_yen":x["investment_yen"],
                                      "payout_yen":x["payout_yen"],"net_yen":x["net_yen"],"roi_pct":x["roi_pct"]}
                                     for x in results if x["LOW_band"]=="ALL"],
           "research_caveats":[
               "2025 and 2026 already observed by earlier experiments; therefore NOT a clean prospective holdout",
               "Pattern median payout cutoff based on historical results; it never removes cheap realized payouts ex post",
               "Even filtering for stable two-year frequency does not prove future true frequency or payout",
               "All 2023/24 training results are re-used for exact combinatorial selection; multiple-testing overfit still possible",
               "Historical original HaruQ AI features and dated model may overlap in-sample results or contain poststart information",
               "No 15-minute-preclose odds, only exact official final settlement payout and archival entry line roles",
               "This is strictly a descriptive exploratory candidate pool, NOT a deployed positive-EV trading policy"
           ]}
    dest=Path(out);dest.mkdir(parents=True,exist_ok=True)
    (dest/"method_and_results.json").write_text(json.dumps(total,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    write_csv(dest/"all_66_pattern_annual_frequency_payout_points.csv",full)
    write_csv(dest/"all_band_training_gate_explanations.csv",catalog)
    write_csv(dest/"LOW_combination_results_all_periods.csv",results)
    write_csv(dest/"per_race_ledger.csv",detail)
    print("PAYOUT_STABILITY_FILTER_REPORT",json.dumps({
        "studied_pattern_count":len(all_types),
        "cohort_races":total["cohort_races"],
        "cutoffs":baselines,
        "per_band_candidates":{b:{"all":policy[b]["discovered"],
                                  "passed":policy[b]["survived"],
                                  "selected":{k:v["selected_count"] for k,v in policy[b]["filtered"].items()}}
                               for b in BANDS},
        "all_low_portfolios":total["comparison_ROI_all_low"],
    },ensure_ascii=False),flush=True)
    return total


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--year26",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.input,a.year26,a.out)


if __name__=="__main__":
    main()
