"""Independent URA research signals: 4-point L1xL3 cross and 2-point counterline.

These are the *frozen 2026-10-26 exploratory definitions*. They do not
replace, modify, or authorize purchases under the official V1/V2 selector.
All 210 trifecta prices and an as-of predicted lineup are required.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Mapping

POLICY_ID = "URA_INDEPENDENT_LINE_SHAPES_20261026_R1"
CROSS_ID = "L1_L3_HEAD_CROSS_4_GEOM_200_400"
COUNTER_ID = "L1_L3_COUNTERLINE_2_A12_SUPPORT_001_0025"
ALL_TICKETS = frozenset(itertools.permutations(range(1, 8), 3))
YEN_PER_TICKET = 100


def _ticket_key(raw):
    if isinstance(raw, (tuple, list)):
        nums = tuple(int(x) for x in raw)
    elif isinstance(raw, str):
        nums = tuple(int(x) for x in raw.replace("=", "-").split("-"))
    else:
        raise ValueError("ticket must be a three-number string or tuple")
    if nums not in ALL_TICKETS:
        raise ValueError("invalid three-car trifecta ticket")
    return nums


def _parse_odds(raw):
    if not isinstance(raw, Mapping):
        raise ValueError("trifecta_odds must map all 210 tickets to prices")
    result = {}
    for key, value in raw.items():
        ticket = _ticket_key(key)
        if ticket in result:
            raise ValueError("duplicate trifecta ticket")
        try:
            number = float(value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("invalid trifecta odds") from exc
        if not math.isfinite(number) or number <= 1:
            raise ValueError("odds must be finite and greater than 1")
        result[ticket] = number
    if set(result) != ALL_TICKETS:
        raise ValueError("all 210 complete trifecta odds are required")
    return result


def _parse_formation(raw):
    if not isinstance(raw, str):
        raise ValueError("formation must be a '/' separated string")
    try:
        lines = [tuple(int(b) for b in s.split("-")) for s in raw.split("/")]
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid line formation") from exc
    bikes = [b for ln in lines for b in ln]
    if (len(bikes) != 7 or set(bikes) != set(range(1, 8))
            or any(not ln for ln in lines)):
        raise ValueError("formation must assign bikes 1..7 once each")
    return lines


def _format_ticket(ticket):
    return "-".join(map(str, ticket))


def _a12(race_class):
    return str(race_class).strip() in ("A1/A2", "A級1・2班", "A級１・２班")


def _candidate(kind, tickets, odds, q, passing, criterion, value, market_lines):
    all_keys = [_format_ticket(ticket) for ticket in tickets]
    return {
        "theory_id": kind,
        "qualified": bool(passing),
        "status": "RESEARCH_MATCH" if passing else "OUT_OF_BAND",
        "condition": criterion,
        "condition_value": round(float(value), 7),
        "formation_tickets": all_keys,
        "selected_tickets": all_keys if passing else [],
        "points": len(tickets) if passing else 0,
        "hypothetical_stake_yen": len(tickets) * YEN_PER_TICKET if passing else 0,
        "market_support": round(sum(q[t] for t in tickets), 9),
        "ticket_odds": {_format_ticket(t): odds[t] for t in tickets},
        "line_rank_source": market_lines,
        "purchase_authorized": False,
    }


def evaluate(formation, trifecta_odds, race_class=None):
    """Evaluate independent market-line patterns without results or V1 ranker.

    Inputs:
      formation: predicted lineup "1-7/5-4-3/6-2"
      trifecta_odds: a complete 210-entry mapping, e.g. {"1-2-3": 51.2}
      race_class: A1/A2, S級, A3, etc. No grade inference from F1/F2.
    Output is a separate research signal, not an executable bet order.
    """
    lines = _parse_formation(formation)
    odds = _parse_odds(trifecta_odds)
    inverse = {t: 1.0 / price for t, price in odds.items()}
    total = sum(inverse.values())
    q = {t: val / total for t, val in inverse.items()}
    p1 = {b: 0.0 for b in range(1, 8)}
    for t, v in q.items():
        p1[t[0]] += v
    ranked = sorted(lines, key=lambda ln: (-sum(p1[b] for b in ln), -p1[ln[0]], ln))
    ranked_str = [_format_ticket(ln) if len(ln) == 3 else "-".join(map(str, ln)) for ln in ranked]
    result = {
        "policy_id": POLICY_ID,
        "market_ranked_lines": ranked_str,
        "source": "full_trifecta_market_normalized_inverse_odds",
        "research_only": True,
        "purchase_authorized": False,
        "candidates": [],
        "qualified_shapes": [],
        "unique_research_tickets": [],
        "overlap_tickets": [],
        "reason": None,
    }
    if len(ranked) < 3 or len(ranked[0]) < 2 or len(ranked[2]) < 2:
        result["reason"] = "L1_OR_L3_HAS_FEWER_THAN_TWO_RIDERS"
        return result
    A, a = ranked[0][:2]
    B, b = ranked[2][:2]
    four = ((A, B, a), (A, B, b), (B, A, a), (B, A, b))
    two = ((A, B, b), (B, A, a))
    geometric = math.exp(sum(math.log(odds[t]) for t in four) / 4.0)
    c4 = _candidate(
        CROSS_ID, four, odds, q, 200 <= geometric < 400,
        "200 <= geometric_mean_odds < 400", geometric, ranked_str)
    support = sum(q[t] for t in two)
    c2 = _candidate(
        COUNTER_ID, two, odds, q, _a12(race_class) and .010 <= support < .025,
        "class=A1/A2 AND .010 <= total_market_support < .025",
        support, ranked_str)
    result["candidates"] = [c4, c2]
    qualified = [c for c in result["candidates"] if c["qualified"]]
    result["qualified_shapes"] = [c["theory_id"] for c in qualified]
    occurrences = [ticket for c in qualified for ticket in c["selected_tickets"]]
    result["unique_research_tickets"] = list(dict.fromkeys(occurrences))
    result["overlap_tickets"] = sorted({t for t in occurrences if occurrences.count(t) > 1})
    result["reason"] = "MATCH" if qualified else "NO_PRICE_BAND_MATCH"
    return result
