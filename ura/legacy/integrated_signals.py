"""Unified Ura-HaruQ *read-only* signal output, preserving adopted V2.

New 2026-10-26 patterns are shown alongside V2. They do not add tickets
to its BET decision or change the frozen V1 six-ticket structure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ura.legacy.participant_v2 import decide as decide_v2
from ura.legacy.market_shapes_v1 import evaluate as evaluate_shapes

POLICY_ID = "URA_V2_WITH_INDEPENDENT_SHAPE_SIGNALS_20261026"
V2_FIELDS = (
    "predicted_bikes_by_rank",
    "line_groups_by_bike",
    "score_gap",
    "competition_scores_by_bike",
)


def evaluate_race(row):
    if not isinstance(row, dict) or "race_id" not in row:
        raise ValueError("each race must contain race_id")
    output = {
        "race_id": str(row["race_id"]),
        "combined_policy_id": POLICY_ID,
        "adopted_v2": None,
        "independent_new_shapes": None,
        "new_shape_status": "NOT_EVALUATED",
        "official_purchase_source": "adopted_v2_only",
        "new_shape_purchase_authorized": False,
    }
    if all(k in row for k in V2_FIELDS):
        output["adopted_v2"] = decide_v2(
            row["predicted_bikes_by_rank"],
            row["line_groups_by_bike"],
            row["score_gap"],
            row["competition_scores_by_bike"],
        )
    if "formation" in row and "trifecta_odds" in row:
        shapes = evaluate_shapes(row["formation"], row["trifecta_odds"], row.get("race_class"))
        output["independent_new_shapes"] = shapes
        output["new_shape_status"] = (
            "RESEARCH_CANDIDATE" if shapes["qualified_shapes"] else "NO_RESEARCH_SIGNAL"
        )
    new_shapes = output["independent_new_shapes"]
    output["new_shape_suggested_tickets"] = (
        new_shapes["unique_research_tickets"] if new_shapes else []
    )
    output["new_shape_suggested_stake_yen"] = (
        100 * len(output["new_shape_suggested_tickets"])
    )
    output["new_shape_decision"] = (
        "BET_CANDIDATE" if output["new_shape_suggested_tickets"] else "NO_SIGNAL"
    )
    # A candidate is a visible model selection, not a settled profitability
    # claim or an automatic purchase instruction. The adopted V2 is separate.
    output["official_tickets"] = (
        output["adopted_v2"]["tickets"] if output["adopted_v2"] else []
    )
    output["official_decision"] = (
        output["adopted_v2"]["decision"] if output["adopted_v2"] else "NOT_EVALUATED"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description="URA V2 + independent new shape signals")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(source, dict) or not isinstance(source.get("races"), list):
        raise ValueError('Expected {"races": [...]}')
    result = {
        "policy_id": POLICY_ID,
        "note": "V2 is unchanged; new patterns are separate research signals.",
        "races": [evaluate_race(row) for row in source["races"]],
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
