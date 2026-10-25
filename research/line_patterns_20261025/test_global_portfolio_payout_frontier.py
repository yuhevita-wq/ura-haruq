"""Whole-portfolio, not individual geometry, acceptance contract tests."""
import unittest
from collections import Counter
import numpy as np
from research.line_patterns_20261025.global_portfolio_payout_frontier import (
  cohort_vectors,exact_portfolio,inventory_from_training,portfolio_score
)


def make_record(winner,payout,tickets,day="2023-05-01"):
    return {"winner_type":winner,"payout":payout,"tickets":Counter(tickets),
            "date":day,"observed_structure":winner}

class GlobalPortfolioTests(unittest.TestCase):
    def test_subsidize_single_losing_shape_when_combination_profitable(self):
        names=["2H-3H-3B|23","3B-2H-3T|13"]
        rows=[make_record(names[0],500,dict(zip(names,[2,1]))),
              make_record(names[1],150,dict(zip(names,[2,1])))]
        yearly=cohort_vectors(rows,names,300)
        self.assertEqual(yearly["stake"].tolist(),[400,200])
        self.assertEqual(yearly["paid"].tolist(),[500,150])
        self.assertEqual(yearly["high"].tolist(),[1,0])
        self.assertEqual(yearly["low"].tolist(),[0,1])
        both={"2023":yearly,"2024":yearly}
        selection=exact_portfolio(names,both,0,True)
        self.assertEqual(set(selection["types"]),set(names))
        self.assertEqual(selection["train_each_year"]["2023"]["paid_yen"],650)
        self.assertEqual(selection["train_each_year"]["2023"]["investment_yen"],600)

    def test_global_median_rejects_low_only_type(self):
        names=["2H-3H-3B|23","3B-2H-3T|13"]
        rows=[make_record(names[0],1000,{names[0]:1,names[1]:1}),
              make_record(names[1],100,{names[0]:1,names[1]:1})]
        by=cohort_vectors(rows,names,400)
        pick=exact_portfolio(names,{"2023":by,"2024":by},0.0,True)
        self.assertEqual(len(pick["types"]),2)
        self.assertGreaterEqual(by["high"].sum(),by["low"].sum())

    def test_no_past_profit_means_empty_portfolio(self):
        names=["2H-3H-3B|23"]
        rows=[make_record(names[0],150,{names[0]:2})]
        yy={"2023":cohort_vectors(rows,names,200),
            "2024":cohort_vectors(rows,names,200)}
        selected=exact_portfolio(names,yy,0.0,True)
        self.assertEqual(selected["types"],[])
        self.assertEqual(selected["training_hits"],0)

    def test_never_filter_any_shape_on_its_own_history(self):
        names=["2H-3H-3B|23","3B-2H-3T|13"]
        a=[make_record(names[0],500,{names[0]:1,names[1]:1})]
        b=[make_record(names[1],200,{names[0]:1,names[1]:1})]
        result=inventory_from_training(a,b)
        self.assertEqual(result,names)

    def test_total_portfolio_hit_payout_break_even(self):
        name="2H-3H-3B|23"
        rows=[make_record(name,1000,{name:2}),
              make_record("3B-2H-3T|13",0,{name:2})]
        s=portfolio_score(rows,[name],200)
        self.assertEqual(s["hits"],1)
        self.assertEqual(s["points"],4)
        self.assertEqual(s["payout_median_yen"],1000)
        self.assertEqual(s["needed_mean_payout_to_break_even_yen"],400)
        self.assertEqual(s["roi_pct"],250)

if __name__=="__main__":unittest.main()
