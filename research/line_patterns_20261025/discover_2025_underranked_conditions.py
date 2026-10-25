"""2025 real-history pre-start feature discovery; NO model or formation changes.

Original source: research GitHub Actions run 40177278424, all 26,004 original
2025 race IDs; 24,141 successfully scored original races (171,297 entries).
Every pre-start rider variable uses records dated strictly BEFORE each race day.
H1 conditions are selected without H2 outcomes, then frozen for H2 assessment.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import pandas as pd

CLASSES=("S級","A級1・2班","チャレンジA3")
COLS=("race_id","date","category","rider_id","rider_name","ura_rank",
      "score_gap","low_band","line_role","line_size","prior_starts",
      "prior_podium","prior_podium_rate","prev5_podium","prev10_podium",
      "top3","first","second","third","half")

def create(input_file,out_dir):
    source=Path(input_file)
    output=Path(out_dir);output.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(source,dtype={"race_id":str,"rider_id":str})
    if len(d)!=171297 or d.race_id.nunique()!=24141 or (
        not d.date.str.startswith("2025-").all()):
        raise RuntimeError("Original full-source 2025 ranked rider rows changed")
    d["date"]=pd.to_datetime(d["date"])
    h=d.groupby(["rider_id","date"],as_index=False,sort=True).agg(
        n=("top3","size"),p=("top3","sum"))
    h.sort_values(["rider_id","date"],inplace=True)
    h["prior_starts"]=h.groupby("rider_id").n.cumsum()-h.n
    h["prior_podium"]=h.groupby("rider_id").p.cumsum()-h.p
    h["prev5_podium"]=h.groupby("rider_id").p.transform(
        lambda s:s.shift(1).rolling(5,min_periods=1).sum()).fillna(0).astype(int)
    h["prev10_podium"]=h.groupby("rider_id").p.transform(
        lambda s:s.shift(1).rolling(10,min_periods=1).sum()).fillna(0).astype(int)
    h["prior_podium_rate"]=h.prior_podium/h.prior_starts.replace(0,np.nan)
    h["prev_date"]=h.groupby("rider_id").date.shift(1)
    h["days_since_last"]=(h.date-h.prev_date).dt.days
    d=d.merge(h.drop(columns=["n","p","prev_date"]),on=["rider_id","date"],
              how="left",validate="many_to_one")
    d["line_size"]=d.groupby(["race_id","line_group"],dropna=False).line_group.transform("size")
    t=d[(d.rider_count==7)&(d.is_low==1)&(d.ura_rank>=5)&
        (d.category.isin(CLASSES))].copy()
    if len(t)!=40851 or t.race_id.nunique()!=13617:
        raise RuntimeError("Seven-car LOW 2025 source denominator changed")
    t["half"]=np.where(t.date<"2025-07-01","H1","H2")
    t["low_band"]=pd.cut(t.score_gap,[-1e-9,.25,.5,.75,1],
            labels=("0-.25",".25-.5",".5-.75",".75-1"),right=False).astype(str)
    t["line_role"]=t.line_role.fillna("未記録")

    known5=t.prior_starts>=5
    known10=t.prior_starts>=10
    b=t.line_role.eq("番手");third=t.line_role.eq("3番手");front=t.line_role.eq("自力")
    conditions={
      "three_line_bante":b&t.line_size.eq(3),
      "three_line_third":third&t.line_size.eq(3),
      "three_line_front":front&t.line_size.eq(3),
      "bante":b,"third":third,"front":front,
      "prev5_podium_ge2":known5&t.prev5_podium.ge(2),
      "prev5_podium_ge3":known5&t.prev5_podium.ge(3),
      "prev5_podium_ge4":known5&t.prev5_podium.ge(4),
      "prev5_podium_eq0":known5&t.prev5_podium.eq(0),
      "prev10_podium_ge5":known10&t.prev10_podium.ge(5),
      "prev10_podium_ge6":known10&t.prev10_podium.ge(6),
      "prev10_podium_le2":known10&t.prev10_podium.le(2),
      "prior_2025_10plus_top3_ge40pct":known10&t.prior_podium_rate.ge(.4),
      "prior_2025_10plus_top3_ge50pct":known10&t.prior_podium_rate.ge(.5),
      "prior_2025_10plus_top3_lt20pct":known10&t.prior_podium_rate.lt(.2),
      "days_since_last_le7":t.days_since_last.le(7),
      "days_since_last_ge21":t.days_since_last.ge(21),
      "bante_prev5_podium_ge2":b&known5&t.prev5_podium.ge(2),
      "third_prev5_podium_ge2":third&known5&t.prev5_podium.ge(2),
      "front_prev5_podium_ge2":front&known5&t.prev5_podium.ge(2),
      "bante_prior_2025_10plus_ge40pct":b&known10&t.prior_podium_rate.ge(.4),
      "third_prior_2025_10plus_ge40pct":third&known10&t.prior_podium_rate.ge(.4),
    }

    def match_metrics(sub,selected,with_role=False):
        controls=sub.loc[~selected]
        cols=["category","ura_rank","low_band"]+(["line_role"] if with_role else [])
        a=controls.groupby(cols,observed=True,dropna=False).top3.agg(["mean","size"]).reset_index()
        a=a[a["size"]>=10]
        picked=sub.loc[selected]
        match=picked.merge(a,on=cols,how="inner")
        n=len(picked);y=int(picked.top3.sum())
        expected=float(match["mean"].sum())
        observed=float(match.top3.sum())
        return {"exposures":n,"podiums":y,"observed_top3_pct":round(100*y/n,3) if n else None,
                "matched_exposures":len(match),
                "matched_expected_podiums":round(expected,2),
                "matched_real_podiums":int(observed),
                "adjusted_top3_lift_pp":round(100*(observed-expected)/len(match),3)
                    if len(match) else None,
                "adjusted_observed_over_expected":round(observed/expected,4)
                    if expected else None},match

    outputs=[]
    selected_matched={}
    for condition,mask in conditions.items():
        # Compare 3-line bante to other BANTE, not to all roles.
        adj_role=condition not in {"bante","third","front"}
        for half in ("H1","H2"):
            x=t.loc[t.half==half]
            q,matched=match_metrics(x,mask.loc[x.index],with_role=adj_role)
            outputs.append({"condition":condition,"half":half,
                "adjusts_for_role":adj_role,**q})
            selected_matched[(condition,half)]=matched

    table=pd.DataFrame(outputs)
    train=table[(table.half=="H1") &
                (table.exposures>=200) &
                (table.matched_exposures>=0.75*table.exposures) &
                (table.adjusted_top3_lift_pp>0)].copy()
    train.sort_values(["adjusted_top3_lift_pp","exposures"],ascending=False,inplace=True)
    top=train.head(8).condition.tolist() # TRAIN ONLY selection, H2 not consulted
    table["selected_from_first_half"]=table.condition.isin(top)

    confidence=[]
    rng=np.random.default_rng(20261025)
    for cond in top:
        for half in ("H1","H2"):
            matched=selected_matched[(cond,half)].copy()
            matched["residual"]=matched.top3-matched["mean"]
            g=matched.groupby("rider_id").residual.agg(["sum","count"]).to_numpy()
            if len(g):
                ix=rng.integers(0,len(g),size=(600,len(g)))
                sample=g[ix].sum(axis=1)
                val=100*sample[:,0]/sample[:,1]
                lower,upper=np.quantile(val,[.025,.975])
            else:lower=upper=float("nan")
            confidence.append({"condition":cond,"half":half,
                  "candidate_distinct_riders":len(g),
                  "rider_cluster_bootstrap95_lift_pp_lower":round(float(lower),3),
                  "rider_cluster_bootstrap95_lift_pp_upper":round(float(upper),3)})

    subgroup=[]
    for cond in top:
        with_role=cond not in {"bante","third","front"}
        for cat in CLASSES:
            for half in ("H1","H2"):
                x=t[(t.category==cat)&(t.half==half)]
                met,_=match_metrics(x,conditions[cond].loc[x.index],with_role)
                subgroup.append({"condition":cond,"class":cat,"half":half,**met})
    t["date"]=t.date.dt.strftime("%Y-%m-%d")
    t[list(COLS)].to_csv(output/"2025_all_7car_LOW_underdog_exposures_and_PRIOR_FORM.csv",
                         index=False,encoding="utf-8-sig")
    table.to_csv(output/"all_23_conditions_H1_H2.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(confidence).to_csv(output/"top8_train_selected_rider_cluster_CI.csv",
                                 index=False,encoding="utf-8-sig")
    pd.DataFrame(subgroup).to_csv(output/"top8_train_selected_by_S_A12_A3.csv",
                                 index=False,encoding="utf-8-sig")
    summary={"year":2025,"research_only":True,
        "original_history_races":26004,"original_frozen_scored_races":24141,
        "primary_races_7car_LOW":t.race_id.nunique(),
        "primary_under_rank_exposures":len(t),
        "H1_under_rank_exposures":int((t.half=="H1").sum()),
        "H2_under_rank_exposures":int((t.half=="H2").sum()),
        "H1_podiums":int(t.loc[t.half=="H1","top3"].sum()),
        "H2_podiums":int(t.loc[t.half=="H2","top3"].sum()),
        "train_only_condition_selection_top8":top,
        "control":"Within same class, original URA 5/6/7 rank, LOW quartile, plus same line role for all nonrole-only candidates",
        "prestart_features":"Earlier 2025 days only; same-day and future finish positions never used; prior 2024 races unavailable in this analysis",
        "uncertainty":"rider-cluster bootstrap 600 trials; exploratory, 23 conditions examined, multiplicity not fully controlled",
        "source":"GitHub run 40177278424 full 2025 audited archive, no V3 final odds subset",
        "model_modified":False,"formation_modified":False,"production_web_modified":False,
        "limitations":["Model itself may have trained on 2025; not a genuine held-out prediction validation",
         "The baseline is conditional observed mean, NOT the original AI's probability forecast.",
         "Effect is association not causation and not proof of profitable betting.",
         "H1 candidate threshold/grid is research-designed, no full correction for search.",
         "2025-YTD rider history omits pre-2025 training periods; do not interpret empty YTD as no career history."]}
    (output/"study_manifest.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("2025_UNDERESTIMATED_TRAIN_TEST",json.dumps({
        "races":summary["primary_races_7car_LOW"],
        "under_rank_entries":len(t),"H1_top8":top,
        "H1_H2":table[table.condition.isin(top)].to_dict(orient="records"),
        "H2_subgroups":[x for x in subgroup if x["half"]=="H2"],
    },ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True)
    p.add_argument("--out",required=True)
    a=p.parse_args();create(a.input,a.out)
