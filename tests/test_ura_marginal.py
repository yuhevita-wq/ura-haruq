import json
import unittest
from pathlib import Path
from ura.formation_marginal import analyze, predict_marginals, marginal_choices, frontier
from ura.formation_selector import load_fixed, FIXED_HARUQ_SHA256

class MarginalLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ura=load_fixed("artifacts/ura_ai_v1.json")
        cls.shape=json.loads(Path("artifacts/ura_shape_selector_v1.json").read_text())
        cls.marginal=json.loads(Path("artifacts/ura_marginal_v1.json").read_text())
        cls.source={"race_id":"unit-test","model_sha256":FIXED_HARUQ_SHA256,
                    "date":"2026-10-24","confidence":0.2,
                    "ranked":[{"bike":i,"score":0.2*(8-i)} for i in range(1,8)]}

    def test_seven_class_probability_mass_normalized(self):
        p=predict_marginals(self.source,self.ura,self.shape,self.marginal)
        self.assertEqual(p["decision"],"RESEARCH")
        self.assertAlmostEqual(sum(p["probabilities_by_outcome"].values()),1.,places=10)
        self.assertEqual(len(p["shapes"]),6)

    def test_marginal_new_points_do_not_overlap(self):
        p=predict_marginals(self.source,self.ura,self.shape,self.marginal)
        value=marginal_choices(p,(1,))
        occupied=set(p["shapes"][1]["tickets"])
        self.assertEqual(value["existing_points"],11)
        self.assertTrue(all(not (occupied & set(x["new_tickets"])) for x in value["additional_options"]))
        self.assertEqual(len(value["additional_options"]),5)
        self.assertTrue(all(x["added_points"]==len(x["new_tickets"]) for x in value["additional_options"]))

    def test_pareto_frontier_is_monotonic(self):
        p=predict_marginals(self.source,self.ura,self.shape,self.marginal)
        f=frontier(p)
        self.assertEqual(f[0]["types"],[])
        self.assertTrue(all(a["points"]<b["points"] and a["expected_capture_probability"]<b["expected_capture_probability"]
                            for a,b in zip(f,f[1:])))
        self.assertEqual(len(f[-1]["types"]),6)

    def test_budget_is_scenario_not_purchase(self):
        a=analyze(self.source,self.ura,self.shape,self.marginal,(1,),max_points=16)
        self.assertFalse(a["purchase_approved"])
        self.assertTrue(a["research_only"])
        self.assertLessEqual(a["best_within_scenario"]["points"],16)
        self.assertTrue(a["no_outcomes_or_payouts_are_model_inputs"])

    def test_already_selected_type_is_never_discarded(self):
        out=analyze(self.source,self.ura,self.shape,self.marginal,(1,),max_types=3,max_points=24)
        self.assertTrue(all(1 in z["types"] for z in out["pareto_frontier"]))
        if out["best_within_scenario"]:
            self.assertIn(1,out["best_within_scenario"]["types"])
        self.assertEqual(out["already_selected"],[1])

    def test_five_six_rejected_due_to_overlap(self):
        for n in (5,6):
            source={"race_id":"test","model_sha256":FIXED_HARUQ_SHA256,
                    "confidence":0.2,
                    "ranked":[{"bike":i,"score":0.2*(n+1-i)} for i in range(1,n+1)]}
            p=predict_marginals(source,self.ura,self.shape,self.marginal)
            self.assertEqual(p["decision"],"UNSUPPORTED_5_6_RIDER_OVERLAP")
            self.assertFalse(p["purchase_approved"])

    def test_unchanged_frozen_model(self):
        self.assertEqual(self.marginal["frozen_ura_git_blob"],"5a353643cfe8196ab695521d479237294c3f9b66")
        self.assertEqual(self.marginal["fit_races"],219)
        self.assertFalse(self.marginal["purchase_approved"])
        with self.assertRaises(ValueError):
            predict_marginals({**self.source,"model_sha256":"incorrect"},self.ura,self.shape,self.marginal)

    def test_nine_vehicle_works(self):
        source={"race_id":"test","model_sha256":FIXED_HARUQ_SHA256,
                "confidence":0.2,
                "ranked":[{"bike":i,"score":0.2*(10-i)} for i in range(1,10)]}
        p=predict_marginals(source,self.ura,self.shape,self.marginal)
        self.assertEqual(p["decision"],"RESEARCH")
        self.assertEqual(p["n"],9)

if __name__=="__main__":
    unittest.main()
