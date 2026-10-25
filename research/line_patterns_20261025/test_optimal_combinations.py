"""Tests of exact LOW-conditioned portfolio optimization with inspectable shapes."""
import unittest
from collections import Counter
from research.line_patterns_20261025.optimal_combinations import (
    solve, get_inventory, format_summary, PERMITTED
)


def entry(name,n23,n24,h23,h24,pts23=20,pts24=20):
    return {"type":name,"2023_net_yen":n23,"2024_net_yen":n24,
            "2023_hits":h23,"2024_hits":h24,
            "2023_points":pts23,"2024_points":pts24}


def race(year,winner,tickets,payout):
    return {"date":year+"-01-01","winner_type":winner,
            "tickets":Counter(tickets),"payout":payout}


class CombinatorialPortfolioTests(unittest.TestCase):
    def test_profit_vs_coverage_can_use_loss_type(self):
        shapes=[entry("2H-3H-3B|23",1000,1000,2,2),
                entry("2B-3H-3T|23",-150,-150,6,6)]
        profit=solve(shapes,"profit")
        coverage=solve(shapes,"coverage")
        self.assertEqual(profit["selected"],["2H-3H-3B|23"])
        self.assertEqual(len(coverage["selected"]),2)
        self.assertGreaterEqual(coverage["train_profit_2023_yen"],0)
        self.assertGreaterEqual(coverage["train_profit_2024_yen"],0)
        self.assertGreater(coverage["train_hits"],profit["train_hits"])

    def test_no_arbitrary_four_type_limit(self):
        shapes=[entry(f"2H-3B-{i}|12",1000,1000,2,2) for i in range(9)]
        res=solve(shapes,"profit")
        self.assertEqual(len(res["selected"]),9)
        self.assertEqual(res["train_profit_2023_yen"],9000)

    def test_both_year_guard_rejects_overfit(self):
        shapes=[entry("2H-3H-3B|23",1000,-1000,9,9),
                entry("2B-3H-3T|23",100,100,1,1)]
        res=solve(shapes,"profit")
        self.assertEqual(res["selected"],["2B-3H-3T|23"])

    def test_empty_candidate_is_no_purchase(self):
        a=solve([],"coverage")
        self.assertEqual(a["selected"],[])
        self.assertEqual(a["objective_value"],0)

    def test_eligible_geometries_only_have_12_23_13(self):
        rows23=[race("2023","2H-3H-3B|23",{"2H-3H-3B|23":2,
                                               "3H-3B-3T|123":1},500)]
        rows24=[race("2024","2H-3H-3B|23",{"2H-3H-3B|23":2,
                                               "3H-3B-3T|123":1},500)]
        inv,eligible=get_inventory(rows23,rows24)
        self.assertEqual(inv,["2H-3H-3B|23"])
        self.assertEqual(len(eligible),1)

    def test_negative_result_handling_and_disjoint_tickets(self):
        rows=[race("2023","2H-3H-3B|23",{"2H-3H-3B|23":1,"2H-2B-3H|12":3},1200)]
        res=format_summary(rows,["2H-3H-3B|23","2H-2B-3H|12"])
        self.assertEqual(res["investment_yen"],400)
        self.assertEqual(res["payout_yen"],1200)
        self.assertEqual(res["net_yen"],800)
        self.assertEqual(res["hits"],1)


if __name__=="__main__":
    unittest.main()
