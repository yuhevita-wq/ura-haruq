"""Compare 2025-Q4-locked rider effect strengths on 2026 only.

No candidate selection from any 2026 outcome. Three strategies are the
2025-Q4 best within rider_weight 0, 0.7 and 1.5. Even if each loses,
report every result without claiming profitability.
"""
import argparse,csv,json,hashlib,collections
from pathlib import Path
import numpy as np
from research.line_patterns_20261025.rider_formations_train2025_test2026 import (
    races_from_csv,load_payouts,audit_races,race_scores,writecsv,file_sha
)

WEIGHTS=(0.,.7,1.5)
SEALED_SHA="dc450209cb217f54f79d4fd4ef50f3ef968630c54a349035d7089cd4053fae55"

def main():
    p=argparse.ArgumentParser()
    for x in ("exposures","payouts","policy","grid","out"):p.add_argument("--"+x,required=True)
    a=p.parse_args()
    if file_sha(a.policy)!=SEALED_SHA:raise RuntimeError("2025 fitted policy SHA was altered")
    policy=json.loads(Path(a.policy).read_text())
    if policy["training_year"]!="2025":raise RuntimeError("Policy must only use 2025 results")
    model=policy["rider_model"]
    grid=[]
    with open(a.grid,encoding="utf8",newline="") as f:grid=list(csv.DictReader(f))
    chosen={}
    for w in WEIGHTS:
        rows=[r for r in grid if float(r["player_weight"])==w]
        if not rows:raise RuntimeError("No pre-2026 tuned 2025 grid candidate")
        rows.sort(key=lambda r:(-float(r["selection_quality"]),-int(r["hits"]),int(r["tickets_per_race"])))
        chosen[w]=rows[0]
    if chosen[0.]["tickets_per_race"]!=str(policy["strategy"]["tickets_per_race"]):
        raise RuntimeError("Frozen winner differs from Q4 grid")
    original=races_from_csv(a.exposures)
    if not original or any(r["day"][:4]!="2026" for r in original):
        raise RuntimeError("Only forward 2026 races allowed")
    settled,census=audit_races(original,load_payouts(a.payouts))
    result=[];sumgrp=collections.defaultdict(lambda:collections.Counter())
    for r in settled:
        bikes,idx,base,person,role,rare,line=race_scores(r,model)
        for w in WEIGHTS:
            q=chosen[w]
            scored=base+w*person+float(q["role_weight"])*role+float(q["rarity_weight"])*rare+float(q["line_weight"])*line
            slots=np.argsort(-scored,kind="stable")[:int(q["tickets_per_race"])]
            tickets=[tuple(int(v) for v in bikes[idx[ix]]) for ix in slots]
            stake=100*len(tickets)
            hit=r["winner"] in tickets
            paid=r["payout"] if hit else 0
            label=("No 2025 rider personalization" if w==0 else f"2025 rider personalization {w:g}")
            result.append({"race_id":r["id"],"date":r["day"],"category":r["cat"],
                "n":r["n"],"LOW":r["low"],"approach":label,
                "rider_weight":w,
                "tickets_per_race":len(tickets),
                "selected_tickets":";".join("-".join(map(str,t)) for t in tickets),
                "actual_combo":"-".join(map(str,r["winner"])),
                "hit":int(hit),"stake_yen":stake,"return_yen":paid,
                "payout_per_100":r["payout"],
                "2025_personalized_rider_count":sum(f"{r['cat']}|{e['id']}|1" in model["rider_lift"] for e in r["entrants"])})
            for key in (label,label+"|"+r["cat"],label+"|"+("LOW" if r["low"] else "HIGH"),label+"|"+str(r["n"])+"car"):
                x=sumgrp[key];x["races"]+=1;x["stake"]+=stake;x["payout"]+=paid;x["hits"]+=int(hit)
    for v in sumgrp.values():
        v["ROI_pct"]=round(100*v["payout"]/v["stake"],4) if v["stake"] else None
    out=Path(a.out);out.mkdir(exist_ok=True,parents=True)
    writecsv(out/"races_2026_THREE_2025_FROZEN_STRATEGIES.csv",result)
    obj={"from_2025_validation_only":True,"2026_tuning":False,
         "policy_sha256":file_sha(a.policy),
         "2025_substrategy_precommitted":{str(w):chosen[w] for w in WEIGHTS},
         "2026_census":census,"by_group":dict(sumgrp),
         "2026_races_scored":len(set(r["race_id"] for r in result)),
         "2026_all_method_rows":len(result),
         "2026_source_races":len(original),
         "research_only":True,"production_modified":False}
    out.joinpath("2026_3WAY_MONTHLY_VALIDATION_AUDIT.json").write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n")
    print("2026_FROZEN_RIDER_THREE_WAY",json.dumps({"source":len(original),"scored":obj["2026_races_scored"],
        "settled_methods":{k:v for k,v in sumgrp.items() if "|" not in k},
        "policy_hash":file_sha(a.policy)},ensure_ascii=False),flush=True)
if __name__=="__main__":main()
