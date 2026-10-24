import json
import unittest
from pathlib import Path

from ura.formation_portfolio import analyze, combine_types
from ura.formation_selector import load_fixed, FIXED_HARUQ_SHA256


class PortfolioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ura = load_fixed("artifacts/ura_ai_v1.json")
        cls.shape = json.loads(Path("artifacts/ura_shape_selector_v1.json").read_text())
        cls.source = {
            "race_id":"unit-test-race",
            "date":"2026-10-24",
            "model_sha256":FIXED_HARUQ_SHA256,
            "ranked":[{"bike":i,"score":0.2*(8-i)} for i in range(1,8)],
            "confidence":0.2,
        }

    def test_choose_two_independent_types(self):
        r=analyze(self.source,self.ura,self.shape,max_types=2)
        self.assertEqual(len(r["selected_types"]),2)
        self.assertEqual(r["unique_points"],len(r["unique_tickets"]))
        self.assertEqual(r["unique_points"],len(set(r["unique_tickets"])))
        self.assertTrue(r["stage_one_fixed"])
        self.assertTrue(r["stage_two_one_type_models_fixed"])
        self.assertFalse(r["purchase_approved"])
        self.assertTrue(r["no_outcome_or_odds_used_to_select"])

    def test_multiple_choices_include_all_unique_union(self):
        r=analyze(self.source,self.ura,self.shape,max_types=6)
        self.assertEqual(len(r["selected_types"]),6)
        expected=set().union(*(set(t["added_tickets"]) for t in r["selected_types"]))
        self.assertEqual(set(r["unique_tickets"]),expected)
        self.assertEqual(sum(t["incremental_unique_points"] for t in r["selected_types"]),len(expected))

    def test_optional_budget_does_not_exceed_limit(self):
        r=analyze(self.source,self.ura,self.shape,max_types=6,max_unique_points=16)
        self.assertLessEqual(r["unique_points"],16)
        self.assertEqual(sum(t["incremental_unique_points"] for t in r["selected_types"]),r["unique_points"])

    def test_high_confidence_still_ignored(self):
        src={**self.source,"confidence":2.0,
             "ranked":[{"bike":i,"score":(3 if i==1 else .2*(7-i))} for i in range(1,8)]}
        out=analyze(src,self.ura,self.shape,max_types=6)
        self.assertEqual(out["unique_points"],0)
        self.assertEqual(out["selected_types"],[])
        self.assertFalse(out["purchase_approved"])

    def test_overlap_is_not_charged_twice(self):
        synthetic={
            "race_id":"test","gap":0.4,
            "selected_type":{"type_id":1,"name":"A","rank_positions":[],
                "points":2,"tickets":["1-2-3","1-3-2"],
                "estimated_capture_probability":0.2},
            "other_types":[
                {"type_id":2,"name":"B","rank_positions":[],
                 "points":2,"tickets":["1-2-3","4-2-3"],
                 "estimated_capture_probability":0.1}
            ]
        }
        result=combine_types(synthetic,max_types=2)
        self.assertEqual(result["unique_points"],3)
        self.assertEqual(result["duplicated_points_removed"],1)

    def test_marginal_cutoff_allows_zero_or_one_types(self):
        zero=analyze(self.source,self.ura,self.shape,max_types=6,
                     min_marginal_capture_per_point=1.0)
        self.assertEqual(zero["selected_type_ids"],[])
        self.assertEqual(zero["unique_points"],0)
        self.assertFalse(zero["purchase_approved"])

    def test_reject_invalid_limits(self):
        with self.assertRaises(ValueError):combine_types({},max_types=0)
        with self.assertRaises(ValueError):combine_types({},max_types=7)
        with self.assertRaises(ValueError):combine_types({},max_unique_points=0)
        with self.assertRaises(ValueError):combine_types({},max_unique_points=True)


if __name__=="__main__":
    unittest.main()
