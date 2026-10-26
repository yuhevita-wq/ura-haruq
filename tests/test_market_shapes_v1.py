"""Exact tests for the independent L1xL3 shape definitions and V2 isolation."""
import itertools
import math
import unittest

from ura.market_shapes_v1 import CROSS_ID, COUNTER_ID, evaluate
from ura.integrated_signals import evaluate_race
from ura.participant_v2 import decide as official_v2

LINES = "1-7/5-4/6-2/3"
FOUR = ("1-6-7", "1-6-2", "6-1-7", "6-1-2")
TWO = ("1-6-2", "6-1-7")


def make_book(pairs_odds=250, other_cross_odds=250):
    head_price = {1: 40., 7: 40., 5: 80., 4: 80.,
                  6: 140., 2: 140., 3: 200.}
    book = {
        "-".join(map(str, triple)): head_price[triple[0]]
        for triple in itertools.permutations(range(1, 8), 3)
    }
    for t in FOUR:
        book[t] = pairs_odds if t in TWO else other_cross_odds
    return book


class IndependentFormationTests(unittest.TestCase):
    def test_four_point_cross_geometric_band(self):
        r = evaluate(LINES, make_book(), "S級")
        self.assertEqual(r["market_ranked_lines"][:3],
                         ["1-7", "5-4", "6-2"])
        four, two = r["candidates"]
        self.assertEqual(four["theory_id"], CROSS_ID)
        self.assertEqual(four["selected_tickets"], list(FOUR))
        self.assertEqual(four["points"], 4)
        self.assertEqual(four["hypothetical_stake_yen"], 400)
        self.assertAlmostEqual(four["condition_value"], 250.0)
        self.assertFalse(two["qualified"])
        self.assertFalse(r["purchase_authorized"])

    def test_a12_two_point_support_band(self):
        r = evaluate(LINES, make_book(pairs_odds=35), "A1/A2")
        four, two = r["candidates"]
        self.assertFalse(four["qualified"])
        self.assertTrue(two["qualified"])
        self.assertEqual(two["theory_id"], COUNTER_ID)
        self.assertEqual(two["selected_tickets"], list(TWO))
        self.assertEqual(two["hypothetical_stake_yen"], 200)
        self.assertGreaterEqual(two["market_support"], .010)
        self.assertLess(two["market_support"], .025)

    def test_a12_grade_required_not_f1_or_s(self):
        for grade in ("F1", "S級", "A級チャレンジ", None):
            r = evaluate(LINES, make_book(pairs_odds=35), grade)
            self.assertFalse(r["candidates"][1]["qualified"])

    def test_incomplete_and_bad_odds_fail_closed(self):
        book = make_book()
        book.pop("1-2-3")
        with self.assertRaises(ValueError):
            evaluate(LINES, book, "A1/A2")
        book = make_book()
        book["1-2-3"] = float("nan")
        with self.assertRaises(ValueError):
            evaluate(LINES, book, "A1/A2")
        with self.assertRaises(ValueError):
            evaluate("1-7/1-5/6-2/3-4", make_book(), "A1/A2")

    def test_missing_two_valid_lines_returns_no_signal(self):
        # L3 is a singleton, even though seven distinct riders exist.
        r = evaluate("1-7-4/5-2-6/3", make_book(), "A1/A2")
        self.assertEqual(r["qualified_shapes"], [])
        self.assertEqual(r["reason"], "L1_OR_L3_HAS_FEWER_THAN_TWO_RIDERS")

    def test_integration_keeps_official_v2_exactly(self):
        row = {
            "race_id": "synthetic-001",
            "predicted_bikes_by_rank": [2, 6, 5, 1, 4, 3, 7],
            "line_groups_by_bike": {
                1: "A", 7: "A", 5: "B", 4: "B",
                6: "C", 2: "C", 3: "D",
            },
            "score_gap": 0.1,
            "competition_scores_by_bike": {
                1: 82., 2: 83., 3: 82.5, 4: 82.1,
                5: 84., 6: 83.5, 7: 82.7,
            },
            "formation": LINES,
            "trifecta_odds": make_book(),
            "race_class": "A1/A2",
        }
        expected = official_v2(
            row["predicted_bikes_by_rank"],
            row["line_groups_by_bike"],
            row["score_gap"],
            row["competition_scores_by_bike"],
        )
        new = evaluate_race(row)
        self.assertEqual(new["adopted_v2"], expected)
        self.assertEqual(new["official_tickets"], expected["tickets"])
        self.assertEqual(new["official_decision"], expected["decision"])
        self.assertEqual(len(new["official_tickets"]), 6)
        self.assertEqual(new["new_shape_status"], "RESEARCH_CANDIDATE")
        self.assertFalse(new["new_shape_purchase_authorized"])

    def test_shaped_only_input_requires_no_v1_ranking(self):
        out = evaluate_race({
            "race_id": "synthetic-002", "formation": LINES,
            "trifecta_odds": make_book(), "race_class": "S級",
        })
        self.assertIsNone(out["adopted_v2"])
        self.assertEqual(out["official_decision"], "NOT_EVALUATED")
        self.assertEqual(out["independent_new_shapes"]["qualified_shapes"], [CROSS_ID])


if __name__ == "__main__":
    unittest.main()
