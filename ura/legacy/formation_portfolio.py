#!/usr/bin/env python3
"""Independent multi-type portfolio selection, with frozen URA and frozen shape models.

The recommended combination is a RESEARCH formation only; it never places bets.
No race outcomes, settlements, final odds or held-out labels enter selection.
"""
import argparse
import json
from pathlib import Path

from ura.legacy.formation_selector import choose, load_fixed, FIXED_HARUQ_SHA256


def combine_types(single_type_output, max_types=2, max_unique_points=None, min_marginal_capture_per_point=0.0):
    """Greedily combine 1..6 learned formation types by marginal capture per unique point.

    Each individual type's calibrated capture probability comes from a previously
    trained (read-only) selector. Marginal weighting uses the fraction of NEW
    unique tickets inside that type; this is an approximation, NOT calibrated
    combined hit probability.
    """
    if type(max_types) is not int or not 1 <= max_types <= 6:
        raise ValueError("max_types must be an integer 1..6")
    if max_unique_points is not None and (
            type(max_unique_points) is not int or max_unique_points < 1):
        raise ValueError("max_unique_points must be a positive integer or None")
    if not 0 <= float(min_marginal_capture_per_point) <= 1:
        raise ValueError("Marginal threshold must be between 0 and 1")
    if single_type_output.get("selected_type") is None:
        return {**single_type_output, "system":"URA_HARUQ_MULTI_TYPE_RESEARCH_V1",
                "selected_types":[], "selected_type_ids":[],
                "unique_tickets":[], "unique_points":0,
                "research_only":True,"purchase_approved":False}

    candidates = [single_type_output["selected_type"], *single_type_output["other_types"]]
    if len({int(c["type_id"]) for c in candidates}) != len(candidates):
        raise ValueError("Duplicate type IDs")
    chosen = []
    occupied = set()
    while len(chosen) < max_types:
        offers=[]
        for t in candidates:
            if any(x["type_id"] == t["type_id"] for x in chosen):
                continue
            current=set(t["tickets"])
            if len(current) != t["points"]:
                raise ValueError("Duplicate ticket inside a shape")
            if len(current) == 0:
                continue
            newly=current-occupied
            if not newly:
                continue
            if max_unique_points is not None and len(occupied)+len(newly)>max_unique_points:
                continue
            pred=float(t["estimated_capture_probability"])
            if not 0 <= pred <= 1:
                raise ValueError("Invalid learned shape capture probability")
            # This does not claim a calibrated portfolio probability.
            heuristic_marginal=pred * len(newly)/len(current)
            priority=heuristic_marginal/len(newly)
            if priority < min_marginal_capture_per_point:
                continue
            offers.append((priority,heuristic_marginal,-len(newly),-t["type_id"],t,newly))
        if not offers:
            break
        _, gain, _, _, t, newly = max(offers)
        occupied.update(newly)
        chosen.append({"type_id":t["type_id"],"name":t["name"],
                       "formation_positions":t["rank_positions"],
                       "individual_points":int(t["points"]),
                       "incremental_unique_points":len(newly),
                       "individual_calibrated_capture_probability":float(t["estimated_capture_probability"]),
                       "approximate_marginal_capture_score":float(gain),
                       "added_tickets":sorted(newly)})

    return {
        "system":"URA_HARUQ_MULTI_TYPE_RESEARCH_V1",
        "race_id":single_type_output.get("race_id"),
        "date":single_type_output.get("date"),
        "gap":single_type_output.get("gap"),
        "ura_probability_order_bikes":single_type_output.get("ura_probability_order_bikes"),
        "selected_types":chosen,
        "selected_type_ids":[int(t["type_id"]) for t in chosen],
        "unique_tickets":sorted(occupied),
        "unique_points":len(occupied),
        "duplicated_points_removed":sum(t["individual_points"] for t in chosen)-len(occupied),
        "available_types":len(candidates),
        "max_types":max_types,
        "max_unique_points":max_unique_points,
        "min_marginal_capture_per_point":float(min_marginal_capture_per_point),
        "no_outcome_or_odds_used_to_select":True,
        "stage_one_fixed":True,
        "stage_two_one_type_models_fixed":True,
        "research_only":True,
        "purchase_approved":False,
        "caveat":"Marginal capture score is only a selection heuristic. Multiple type probabilities cannot be added as calibrated portfolio probabilities. Historical final-odds ROI does not establish tradable value."
    }


def analyze(source, ura, shape, max_types=2, max_unique_points=None, min_marginal_capture_per_point=0.0):
    if source.get("model_sha256") != FIXED_HARUQ_SHA256:
        raise ValueError("Frozen HaruQ model SHA mismatch")
    original = choose(source, ura, shape)
    return combine_types(original, max_types=max_types,
                         max_unique_points=max_unique_points,
                         min_marginal_capture_per_point=min_marginal_capture_per_point)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--prediction",required=True,type=Path)
    ap.add_argument("--ura",type=Path,default=Path("artifacts/ura_ai_v1.json"))
    ap.add_argument("--shape",type=Path,default=Path("artifacts/ura_shape_selector_v1.json"))
    ap.add_argument("--max-types",type=int,default=2)
    ap.add_argument("--max-unique-points",type=int)
    ap.add_argument("--min-marginal-capture-per-point",type=float,default=0.0)
    ap.add_argument("--out",type=Path,default=Path("ura_portfolio_research.json"))
    args=ap.parse_args()
    source=json.loads(args.prediction.read_text(encoding="utf-8"))
    fixed_ura=load_fixed(args.ura)
    shape=json.loads(args.shape.read_text(encoding="utf-8"))
    result=analyze(source,fixed_ura,shape,args.max_types,args.max_unique_points,
                   args.min_marginal_capture_per_point)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"race_id":result["race_id"],
                      "selected_types":result["selected_type_ids"],
                      "unique_points":result["unique_points"],
                      "purchase_approved":False},ensure_ascii=False))


if __name__=="__main__":
    main()
