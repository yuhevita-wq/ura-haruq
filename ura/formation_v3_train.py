"""V3 data-driven formation learning: 2023 discovery, 2024 selection, 2025 untouched evaluation.

The 7-bin lattice describes the 3 *ordered* finishing ranks. It is a new
experiment; original URA coefficients and previously catalogued types stay frozen.
Only 2023 and 2024 label/revenue tensors are exposed to candidate selection.
"""
import argparse
import csv
import itertools
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from ura.formation_selector import rank_tickets
from ura.core import low_rank_start

PER_YEN=100
TRAIN=2023
SELECT=2024
TEST=2025
SEED=20261024

def rank_bin(rank,n):
    if not 1<=rank<=n or n not in (7,8,9):raise ValueError("Bad ranking")
    return 1+(rank-1)*7//n

def key3(ranks,n):
    return tuple(rank_bin(i,n)-1 for i in ranks)

def read_year(input_dir,year):
    source=Path(input_dir)/f"year_{year}.csv"
    records=[]
    with source.open(encoding="utf8",newline="") as fp:
        for z in csv.DictReader(fp):
            if int(z["year"])!=year:raise ValueError("Cross-year source")
            ranks=tuple(int(t) for t in z["actual_ranks"].split("-"))
            n=int(z["n"])
            if n not in (7,8,9) or len(set(ranks))!=3 or len(ranks)!=3:
                raise ValueError("Malformed observed top three")
            pay=int(z["payout_yen"])
            if pay<100:raise ValueError("Invalid official settlement")
            records.append({"race_id":z["race_id"],"date":z["date"],"year":year,
                            "n":n,"winner_ranks":ranks,"winning_bin":key3(ranks,n),
                            "payout":pay})
    if not records:raise RuntimeError(f"No {year} labeled races")
    if len({z["race_id"] for z in records})!=len(records):
        raise RuntimeError(f"Repeated {year} race ID")
    return records

def odds_lattice(n):
    matrix=np.zeros((7,7,7),float)
    for p in itertools.permutations(range(1,n+1),3):
        if any(rank>=low_rank_start(n) for rank in p):
            matrix[key3(p,n)]+=1.
    top=low_rank_start(n)-1
    assert abs(matrix.sum()-(n*(n-1)*(n-2)-top*(top-1)*(top-2)))<1e-9
    return matrix

LATTICES={n:odds_lattice(n) for n in (7,8,9)}

def group_matrices(rows):
    cost=np.zeros((7,7,7),float)
    payout=np.zeros((7,7,7),float)
    hit=np.zeros((7,7,7),float)
    cnt=Counter(z["n"] for z in rows)
    for n,k in cnt.items():cost+=LATTICES[n]*k*PER_YEN
    for r in rows:
        if not any(v>=low_rank_start(r["n"]) for v in r["winner_ranks"]):
            continue   # Original URA policy: exclude tickets with no lower-rated riders
        slot=r["winning_bin"]
        payout[slot]+=r["payout"]
        hit[slot]+=1
    assert sum(cnt.values())==len(rows)
    return cost,payout,hit,cnt

def intervals():
    return [(a,b) for a in range(7) for b in range(a,7)]

def all_candidates():
    v=intervals()
    cand=np.array([x+y+z for x,y,z in itertools.product(v,v,v)],dtype=np.int32)
    if cand.shape!=(21952,6):raise RuntimeError("Shape enumeration mismatch")
    return cand

def cube_sum(matrix,cands):
    # Prefix-sum a padded (8,8,8) tensor; vectorize any number of cuboids.
    x=np.pad(matrix,((1,0),(1,0),(1,0))).cumsum(0).cumsum(1).cumsum(2)
    a=cands[:,0];b=cands[:,1]+1
    c=cands[:,2];d=cands[:,3]+1
    e=cands[:,4];f=cands[:,5]+1
    return (x[b,d,f]-x[a,d,f]-x[b,c,f]-x[b,d,e]
           +x[a,c,f]+x[a,d,e]+x[b,c,e]-x[a,c,e])

def candidate_occupancy(c):
    a,b,d,e,g,h=(int(v) for v in c)
    x=np.zeros((7,7,7),bool)
    x[a:b+1,d:e+1,g:h+1]=True
    return x

def stats(cost,payout,hits,candidates):
    paid=cube_sum(payout,candidates)
    stake=cube_sum(cost,candidates)
    cnt=cube_sum(hits,candidates).astype(int)
    roi=np.divide(paid,stake,out=np.zeros_like(paid),where=stake>0)*100.
    return stake,paid,cnt,roi

def candidate_score(a,b,cands,already,first=False):
    # Restrict to incremental, non-overlapping outcomes. The actual bought
    # combinations are unique by construction, including repeat rank-bin links.
    available=~already
    stake_a,paid_a,hits_a,roi_a=stats(*(x*available for x in a[:3]),cands)
    stake_b,paid_b,hits_b,roi_b=stats(*(x*available for x in b[:3]),cands)
    n_a=sum(a[3].values());n_b=sum(b[3].values())
    # Learn on 2023, tune on 2024. Each picked region must occur in both years.
    mean_points=(stake_a/100./n_a+stake_b/100./n_b)/2
    supported=(hits_a>=5)&(hits_b>=5)&(stake_a>0)&(stake_b>0)&(mean_points<=24)&(mean_points>=1)
    # Penalize noisy 1-2-hit jackpots; support in both years, robust across periods.
    score=np.minimum(roi_a,roi_b)
    score=np.where(supported,score,-np.inf)
    return score,mean_points,roi_a,roi_b,hits_a,hits_b,stake_a,stake_b,paid_a,paid_b

def choose_forms(train,dev,max_types=6):
    cands=all_candidates()
    selection=np.zeros((7,7,7),bool)
    chosen=[]
    ranked_first=[]
    for stage in range(max_types):
        score,points,r1,r2,h1,h2,s1,s2,p1,p2=candidate_score(train,dev,cands,selection)
        candidates=np.flatnonzero(np.isfinite(score))
        if stage==0:
            top=sorted(candidates,key=lambda i:(-score[i],-min(h1[i],h2[i]),points[i],i))[:20]
            ranked_first=[{"candidate_idx":int(i),"bin_bounds_1based":[int(cands[i,j]+(j%2==0)) for j in range(6)],
                           "roi_2023":float(r1[i]),"roi_2024":float(r2[i]),
                           "hits_2023":int(h1[i]),"hits_2024":int(h2[i]),
                           "avg_points":round(float(points[i]),2)} for i in top]
        if not len(candidates):break
        # Require incremental profitability in BOTH years; do not manufacture types.
        survivors=candidates[score[candidates]>=100.]
        if not len(survivors):break
        best=min(survivors,key=lambda i:(-score[i],-min(h1[i],h2[i]),points[i],i))
        region=candidate_occupancy(cands[best])
        newly=region&~selection
        if not newly.any():break
        row={"id":f"V3_{len(chosen)+1}",
             "candidate_idx":int(best),
             "bin_bounds_0based":cands[best].tolist(),
             "training_incremental_roi_2023":round(float(r1[best]),4),
             "selection_incremental_roi_2024":round(float(r2[best]),4),
             "training_incremental_hits_2023":int(h1[best]),
             "selection_incremental_hits_2024":int(h2[best]),
             "incremental_avg_points":round(float(points[best]),4),
             "tickets_touched_in_bin_space":int(newly.sum())}
        chosen.append(row)
        selection|=region
    return chosen,selection,ranked_first

def old_type_summary(rows):
    summary={}
    for tid in range(1,7):
        stake=0;paid=0;hits=0
        for z in rows:
            combos=rank_tickets(z["n"],tid)
            stake+=100*len(combos)
            if z["winner_ranks"] in combos:paid+=z["payout"];hits+=1
        summary[str(tid)]={"races":len(rows),"tickets":int(stake/100),
                           "hits":hits,"stake_yen":stake,"return_yen":paid,
                           "roi_percent":round(100*paid/stake,4) if stake else None}
    return summary

def evaluate_masks(rows,mask):
    sums=Counter();ledger=[]
    for z in rows:
        n=z["n"]
        points=int(np.sum(LATTICES[n]*mask))
        hit=bool(mask[z["winning_bin"]]) and any(v>=low_rank_start(n) for v in z["winner_ranks"])
        stake=100*points;ret=z["payout"] if hit else 0
        ledger.append({"race_id":z["race_id"],"date":z["date"],"year":z["year"],
                       "n":n,"actual_rank":"-".join(map(str,z["winner_ranks"])),
                       "tickets":points,"hit":int(hit),"invested_yen":stake,
                       "returned_yen":ret})
        sums["races"]+=1;sums["tickets"]+=points;sums["hits"]+=int(hit)
        sums["investment"]+=stake;sums["return"]+=ret
    sums=dict(sums);sums["roi_percent"]=round(100*sums["return"]/sums["investment"],4) if sums["investment"] else None
    if sums["investment"]:
        a=np.array([(r["invested_yen"],r["returned_yen"]) for r in ledger],float)
        gen=np.random.default_rng(SEED)
        vals=[]
        for _ in range(2000):
            v=a[gen.integers(len(a),size=len(a))].sum(axis=0)
            if v[0]>0:vals.append(100*v[1]/v[0])
        sums["roi_bootstrap_95pct"]=[round(float(v),2) for v in np.quantile(vals,[.025,.975])]
    return sums,ledger

def main_train(source,out):
    per_year={year:read_year(source,year) for year in (TRAIN,SELECT,TEST)}
    base={year:group_matrices(r) for year,r in per_year.items()}
    forms,chosen_mask,leaderboard=choose_forms(base[TRAIN],base[SELECT])
    record=[]
    # fixed model 2025 eval without checking its outcome during selection.
    comparisons={}
    for year,rows in per_year.items():
        val,details=evaluate_masks(rows,chosen_mask)
        comparisons[str(year)]={"new_v3_union":val,"old_types":old_type_summary(rows)}
        record.extend(details)
    meta={"system":"URA_FORMATION_V3_2023_START_RESEARCH",
          "research_only":True,"purchase_approved":False,
          "selection_protocol":"2023 discover; 2024 choose disjoint incremental ROI >100% with at least 5 hits in each year, 1-24 average new points; 2025 evaluate without label access until after selection",
          "type_formation_family":"axis-aligned cuboids over 7-bin normalized URA probability rank triplets, every ticket must include at least one original-URA lower-ranked rider", 
          "lower_rank_eligibility_enforced":True,
          "rank_to_bin_formula":"1 + floor(7*(rank-1)/field_size)",
          "candidate_count":21952,
          "rows_by_year":{str(yr):len(rows) for yr,rows in per_year.items()},
          "type_count":len(forms),"learned_types":forms,
          "training_candidate_leaderboard_2023_2024":leaderboard,
          "same_models_as_2026_v2":True,
          "original_model_sha256":"e70690f4538b77e927a042e7ae2db312f8a0a4f16bf72def37e7ccad3568c193",
          "frozen_ura_git_blob":"5a353643cfe8196ab695521d479237294c3f9b66",
          "comparison":comparisons,
          "warnings":["Each candidate and all three annual ROI calculations strictly exclude 3連単 tickets with no lower-ranked riders as defined by fixed URA.",
                      "Original HaruQ was originally trained with historic outcomes including possibly 2023-2025; its reproduced rank is not genuine walk-forward. (Audit required)",
                      "Original reconstructed pre-race features may include final odds/weather, current rider class or popularity; as-of audit pending.",
                      "Final odds cannot be assumed available at purchase deadline.",
                      "Mass candidate search creates multiple-testing risk even after temporal validation.",
                      "2025 was not used to choose V3 types but other project research has viewed 2025 before."]}
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    (root/"model_v3.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    with (root/"yearly_ticket_outcomes.csv").open("w",encoding="utf8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(record[0]));w.writeheader();w.writerows(record)
    print("V3_LEARNED_RESULT",json.dumps({"matched":meta["rows_by_year"],"types":meta["type_count"],
          "discovered":forms,"roi_by_year":{y:z["new_v3_union"]["roi_percent"] for y,z in comparisons.items()},
          "held_out_2025":comparisons["2025"]["new_v3_union"]},ensure_ascii=False),flush=True)
    return meta

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",type=Path,default=Path("v3_input"))
    ap.add_argument("--out",type=Path,default=Path("v3_learning"))
    x=ap.parse_args()
    main_train(x.input,x.out)

if __name__=="__main__":
    main()
