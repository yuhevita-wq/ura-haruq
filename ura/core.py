"""URA-HARUQ v1: ordered 3-finisher probabilities calibrated on frozen HaruQ scores.

No claims of model-score-to-probability calibration unless trained.
All race-level predictions are normalized over every possible ordered triple.
"""
import itertools
import numpy as np
from scipy.special import logsumexp

FEATURE_NAMES=("zscore","gap_times_zscore","rank_fraction","low_rank","gap_times_low_rank")
DEFAULT_COEFFICIENTS=np.array([1.0,0.,0.,0.,0.],dtype=float)

def low_rank_start(n):
    if not 5<=n<=9:
        raise ValueError("Eligible races have 5-9 distinct riders")
    return n//2+(2 if n%2 else 1)

def prepare(ranked):
    """ranked is list of dicts with bike_number and score; returns desc-sorted features."""
    if not 5<=len(ranked)<=9:raise ValueError("5-9 riders required")
    if len({int(x["bike_number"]) for x in ranked})!=len(ranked):
        raise ValueError("Duplicate bike numbers")
    riders=sorted(ranked,key=lambda t:(-float(t["score"]),int(t["bike_number"])))
    scores=np.asarray([float(x["score"]) for x in riders],dtype=float)
    if not np.isfinite(scores).all():raise ValueError("Non-finite model score")
    gap=float(scores[0]-scores[1])
    z=(scores-scores.mean())/max(float(scores.std()),0.001)
    fractions=np.arange(len(riders),dtype=float)/max(1,len(riders)-1)
    underdog=(np.arange(len(riders))+1>=low_rank_start(len(riders))).astype(float)
    features=np.column_stack((z,gap*z,fractions,underdog,gap*underdog))
    return riders,features,gap,underdog

def ordered_probabilities(features,coefficients):
    x=np.asarray(features,dtype=float)
    theta=np.asarray(coefficients,dtype=float)
    if x.ndim!=2 or x.shape[1]!=len(theta):
        raise ValueError("Incorrect score feature dimension")
    logits=x@theta
    if not np.isfinite(logits).all():raise ValueError("Non-finite trained utilities")
    results={}
    for i,j,k in itertools.permutations(range(len(logits)),3):
        remaining=np.delete(logits,i)
        third=np.delete(logits,[i,j])
        p=float(np.exp(logits[i]-logsumexp(logits)
                        +logits[j]-logsumexp(remaining)
                        +logits[k]-logsumexp(third)))
        results[(i,j,k)]=p
    total=sum(results.values())
    if abs(total-1)>1e-8:
        raise RuntimeError(f"3連単 probability mass mismatch: {total}")
    return results

def ordered_logloss_and_gradient(theta,races,regularization=0.):
    """races: iterable of (feature_matrix, indices_of_actual_1st_2nd_3rd)."""
    theta=np.asarray(theta,float)
    total=0.
    grad=np.zeros_like(theta)
    n=0
    for features,winners in races:
        x=np.asarray(features,float)
        y=tuple(int(v) for v in winners)
        if len(y)!=3 or len(set(y))!=3:
            raise ValueError("Invalid 1st-3rd finishing order")
        logits=x@theta
        remaining=np.ones(len(x),bool)
        for winner in y:
            if winner<0 or winner>=len(x) or not remaining[winner]:
                raise ValueError("Invalid actual winner index")
            v=logits[remaining]
            norm=logsumexp(v)
            p=np.exp(v-norm)
            total+=norm-logits[winner]
            grad+=p@x[remaining]-x[winner]
            remaining[winner]=False
        n+=1
    if not n:raise ValueError("No training races")
    diff=theta-DEFAULT_COEFFICIENTS
    return (float(total/n+regularization*diff@diff),
            grad/n+2*regularization*diff)

def select_candidates(ranked,coefficients,include_all=False):
    riders,x,gap,underdog=prepare(ranked)
    probs=ordered_probabilities(x,coefficients)
    out=[]
    for (i,j,k),p in probs.items():
        if include_all or (gap<1.0 and any(underdog[v] for v in (i,j,k))):
            out.append({"ticket":"-".join(str(riders[v]["bike_number"]) for v in (i,j,k)),
                        "probability":p,
                        "lower_rank_roles":[role for role,v in zip((1,2,3),(i,j,k)) if underdog[v]]})
    out.sort(key=lambda t:(-t["probability"],t["ticket"]))
    return {"confidence_gap":gap,"low_confidence":gap<1.0,
            "lower_rank_bikes":[int(riders[i]["bike_number"]) for i in range(len(riders)) if underdog[i]],
            "candidate_count":len(out),"candidates":out}
