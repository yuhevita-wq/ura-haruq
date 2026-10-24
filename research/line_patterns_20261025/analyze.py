"""URA-HARUQ line-composition research ONLY.

Reads immutable 2023/24/25 V3 labels and historical line formations from the
read-only SQLite archive. The frozen baseline, URA coefficients and 2-2-2-1
policy are NEVER fitted or written here. Archived odds are final, not quotes.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from db.schema import get_connection
from ura.core import low_rank_start
from ura.formation_v3_train import (
    read_year, group_matrices, choose_forms, evaluate_masks, rank_bin,
)

YEARS = (2023, 2024, 2025)
CONTROL = "2-2-2-1"
MIN_TRAIN = 120
MIN_SCREEN = 120
SEED = 20261025


def get_groups(conn, race_id, table, expected):
    raw = conn.execute(
        f"SELECT bike_number,line_group,line_role FROM {table} WHERE race_id=?",
        (race_id,),
    ).fetchall()
    if len(raw) != len(expected):
        return None
    bikes = [int(r["bike_number"]) for r in raw if r["bike_number"] is not None]
    if len(bikes) != len(expected) or len(set(bikes)) != len(expected):
        return None
    if set(bikes) != expected:
        return None
    if any(r["line_group"] is None or not str(r["line_group"]).strip()
           for r in raw):
        return None
    result = defaultdict(list)
    for r in raw:
        result[str(r["line_group"]).strip()].append(int(r["bike_number"]))
    groups = [tuple(sorted(members)) for members in result.values()]
    if sum(map(len, groups)) != len(expected):
        return None
    return sorted(groups)


def load_lineup(conn, race_id, expected):
    entry = get_groups(conn, race_id, "entries", expected)
    result = get_groups(conn, race_id, "race_results", expected)
    if entry is not None and result is not None and entry != result:
        return None, "entries_results_disagree"
    if entry is not None:
        return entry, "entries"
    if result is not None:
        return result, "result_page_lineup"
    return None, "no_complete_lineup"


def low_band(gap):
    if not (0 <= gap < 1):
        raise ValueError("Outside fixed LOW (<1) threshold")
    return ("0.00-0.25" if gap < .25 else
            "0.25-0.50" if gap < .50 else
            "0.50-0.75" if gap < .75 else "0.75-1.00")


def enrich(conn, row):
    n = int(row["n"])
    predicted = tuple(map(int, row["ranked_bikes"].split("-")))
    finished = tuple(map(int, row["actual_bikes"].split("-")))
    ranks = tuple(map(int, row["actual_ranks"].split("-")))
    if n not in (7, 8, 9) or len(predicted) != n:
        raise ValueError("Invalid original V3 row")
    if len(set(predicted)) != n or len(set(finished)) != 3:
        raise ValueError("Repeated rider")
    if finished != tuple(predicted[r - 1] for r in ranks):
        raise ValueError("V3 ranking/outcome inconsistency")
    groups, source = load_lineup(conn, row["race_id"], set(predicted))
    if groups is None:
        return None, source
    bike_to_group = {b: i for i, group in enumerate(groups)
                     for b in group}
    pattern = "-".join(map(str, sorted((len(g) for g in groups), reverse=True)))
    rank_by_bike = {bike: i + 1 for i, bike in enumerate(predicted)}
    low_cut = low_rank_start(n)
    low_roles = [i + 1 for i, r in enumerate(ranks) if r >= low_cut]
    # "Weak-line proxy": the line with the WORST mean fixed-URA probability
    # ranking; not a claim about actual line strength. Tied mean ranks = unknown.
    mean_ranks = [sum(rank_by_bike[b] for b in g) / len(g) for g in groups]
    weakest = max(mean_ranks)
    ties = [i for i, avg in enumerate(mean_ranks) if abs(avg - weakest) < 1e-10]
    weak_roles = ([i + 1 for i, b in enumerate(finished)
                   if bike_to_group[b] == ties[0]] if len(ties) == 1 else [])
    same = [bike_to_group[b] for b in finished]
    data = {
        "race_id": row["race_id"], "date": row["date"], "year": int(row["year"]),
        "n": n, "pattern": pattern, "line_source": source,
        "gap_band": low_band(float(row["gap"])),
        "payout_yen": int(row["payout_yen"]),
        "lower_1": int(1 in low_roles), "lower_2": int(2 in low_roles),
        "lower_3": int(3 in low_roles), "lower_any": int(bool(low_roles)),
        "weak_mean_tied": int(len(ties) != 1),
        "weak_1": int(1 in weak_roles), "weak_2": int(2 in weak_roles),
        "weak_3": int(3 in weak_roles),
        "line_12": int(same[0] == same[1]),
        "line_23": int(same[1] == same[2]),
        "line_13": int(same[0] == same[2]),
        "line_all": int(len(set(same)) == 1),
        "actual_ranks": "-".join(map(str, ranks)),
        "grade": row.get("grade", ""),
    }
    return data, source


def read_original_csv(folder, year):
    with (Path(folder) / f"year_{year}.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or len(rows) != len({r["race_id"] for r in rows}):
        raise RuntimeError(f"{year}: missing or duplicate V3 records")
    return rows


def percentile(values, q):
    return round(float(np.quantile(np.asarray(values, dtype=float), q)), 2) if values else None


def describe(rows):
    n = len(rows)
    payouts = [r["payout_yen"] for r in rows]
    denominator = sum(1 - r["weak_mean_tied"] for r in rows)
    sums = {k: sum(r[k] for r in rows) for k in
            ("lower_1", "lower_2", "lower_3", "lower_any",
             "weak_1", "weak_2", "weak_3", "weak_mean_tied",
             "line_12", "line_23", "line_13", "line_all")}
    return {
        "races": n, "lineup_entries": sum(r["line_source"] == "entries" for r in rows),
        "lineup_result_page": sum(r["line_source"] == "result_page_lineup" for r in rows),
        "median_3rentan_yen": percentile(payouts, .5),
        "mean_3rentan_yen": round(sum(payouts) / n, 2) if n else None,
        "p90_3rentan_yen": percentile(payouts, .9),
        "underdog_any_rate": round(sums["lower_any"] / n, 5) if n else None,
        "same_line_first_second_rate": round(sums["line_12"] / n, 5) if n else None,
        "same_line_second_third_rate": round(sums["line_23"] / n, 5) if n else None,
        "underdog_finish_roles": {str(i): sums[f"lower_{i}"] for i in (1, 2, 3)},
        "weak_line_finish_roles": {str(i): sums[f"weak_{i}"] for i in (1, 2, 3)},
        "weak_line_role_denominator": denominator,
        "weak_line_proxy_ties": sums["weak_mean_tied"],
        "line_finish_counts": {k: sums[k] for k in
                               ("line_12", "line_23", "line_13", "line_all")},
    }


def bootstrap_daily(ledger, repeat=1500):
    day_rows = defaultdict(lambda: [0, 0])
    for x in ledger:
        a = day_rows[x["date"]]
        a[0] += x["invested_yen"]
        a[1] += x["returned_yen"]
    if not day_rows:
        return None
    values = np.asarray(list(day_rows.values()), dtype=float)
    rng = np.random.default_rng(SEED)
    ratios = []
    for _ in range(repeat):
        chosen = values[rng.integers(len(values), size=len(values))].sum(axis=0)
        if chosen[0]:
            ratios.append(100 * chosen[1] / chosen[0])
    return [percentile(ratios, .025), percentile(ratios, .975)] if ratios else None


def analyze(source, output):
    original = {y: read_original_csv(source, y) for y in YEARS}
    labels = {y: read_year(source, y) for y in YEARS}
    audit = Counter()
    decorated = []
    with get_connection() as conn:
        for year in YEARS:
            for row in original[year]:
                audit[f"{year}_v3_input"] += 1
                record, reason = enrich(conn, row)
                if record is None:
                    audit[f"{year}_{reason}"] += 1
                else:
                    decorated.append(record)
                    audit[f"{year}_lineup_{reason}"] += 1
    by_id = {z["race_id"]: z for z in decorated}
    if len(by_id) != len(decorated):
        raise RuntimeError("Duplicate enriched race IDs")
    joined = {y: defaultdict(list) for y in YEARS}
    for year in YEARS:
        for original_label in labels[year]:
            supplement = by_id.get(original_label["race_id"])
            if supplement:
                joined[year][supplement["pattern"]].append(original_label)
    patterns = sorted({p for year in YEARS for p in joined[year]})
    if not patterns:
        raise RuntimeError("No usable complete line compositions: see audit")
    overall = []
    band_report = []
    for pattern in patterns:
        for year in YEARS:
            observed = [r for r in decorated
                        if r["year"] == year and r["pattern"] == pattern]
            stats = describe(observed)
            overall.append({"pattern": pattern, "year": year, **stats})
            for band in ("0.00-0.25", "0.25-0.50", "0.50-0.75", "0.75-1.00"):
                subset = [r for r in observed if r["gap_band"] == band]
                band_report.append({"pattern": pattern, "year": year, "LOW_band": band,
                                    **describe(subset)})
    # Frozen 2-2-2-1 is a descriptive CONTROL ONLY. No attempt to reconstruct,
    # re-fit or modify the user's fixed 2-2-2-1 ticket policy.
    models = []
    ledger_all = []
    for pattern in patterns:
        cohorts = {y: joined[y][pattern] for y in YEARS}
        counts = {str(y): len(cohorts[y]) for y in YEARS}
        if pattern == CONTROL:
            models.append({"pattern": pattern, "status": "FROZEN_CONTROL_NOT_TRAINED",
                           "cohort_counts": counts})
            continue
        if len(cohorts[2023]) < MIN_TRAIN or len(cohorts[2024]) < MIN_SCREEN:
            models.append({"pattern": pattern, "status": "DESCRIPTIVE_ONLY_INSUFFICIENT_SUPPORT",
                           "cohort_counts": counts})
            continue
        discovery = group_matrices(cohorts[2023])
        screen = group_matrices(cohorts[2024])
        learned, mask, _ = choose_forms(discovery, screen, max_types=5)
        if not learned:
            models.append({"pattern": pattern, "status": "NO_REPRODUCIBLE_SHAPE_2023_2024",
                           "cohort_counts": counts, "learned_types": []})
            continue
        evaluation = {}
        for year in YEARS:
            if not cohorts[year]:
                evaluation[str(year)] = {"races": 0, "roi_percent": None}
                continue
            numbers, ledger = evaluate_masks(cohorts[year], mask)
            numbers["day_cluster_roi_95pct"] = bootstrap_daily(ledger)
            evaluation[str(year)] = numbers
            if year == 2025:
                for r in ledger:
                    ledger_all.append({"pattern": pattern, **r})
        models.append({"pattern": pattern, "status": "RESEARCH_ONLY_NOT_APPROVED",
                       "cohort_counts": counts, "learned_types": learned,
                       "evaluation": evaluation})
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "study": "URA_LINE_PATTERN_RESEARCH_20261025",
        "frozen_2221_unchanged": True, "original_haruq_and_ura_untouched": True,
        "purchase_approved": False, "research_only": True,
        "years": {str(y): {"original_v3_LOW_races": len(original[y]),
                          "usable_classified_lineups": sum(r["year"] == y for r in decorated)}
                  for y in YEARS},
        "lineup_audit": dict(audit),
        "thresholds_for_RESEARCH_only": {"train_2023": MIN_TRAIN,
                                         "screen_2024": MIN_SCREEN},
        "patterns": patterns, "models": models,
        "warnings": [
            "2-2-2-1 existing V1 rule is only a control; not re-trained or changed.",
            "Weak line here means largest MEAN fixed URA probability rank; ties excluded. This is a diagnostic proxy, not the original V1 definition.",
            "Result-page lineup is post-race archived evidence of a predicted lineup, not an independently timestamped pre-start snapshot.",
            "Past original-HaruQ features may leak final odds/popularity/weather; pre-start parity is NOT verified.",
            "Original HaruQ may have been trained on the 2023-2025 historical outcomes.",
            "2023 shape discovery and 2024 screening were used for selection; 2025 V3 shape held out, but previously inspected in wider project.",
            "Final odds cash settlement is not a reproducible pre-close quotation.",
            "Multiple testing over thousands of shapes and several line cohorts; bootstraps do not correct model selection bias.",
            "No forecast profitability, model improvement or operational inclusion is established by this audit.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    for filename, records in (
        ("lineup_races.csv", decorated),
        ("pattern_yearly.csv", overall),
        ("pattern_LOW_quartiles.csv", band_report),
        ("pattern_2025_formation_ledger.csv", ledger_all),
    ):
        if not records:
            continue
        with (out / filename).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print("URA_LINE_PATTERN_SUMMARY", json.dumps({
        "source_rows": {str(y): len(original[y]) for y in YEARS},
        "classified": len(decorated), "pattern_counts": {
            p: {str(y): len(joined[y][p]) for y in YEARS} for p in patterns
        }, "model_status": {m["pattern"]: m["status"] for m in models},
    }, ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.input, args.out)
