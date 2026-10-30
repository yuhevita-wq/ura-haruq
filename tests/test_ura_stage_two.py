import json
import unittest
from pathlib import Path
from ura.legacy.formation_selector import CATALOG, rank_tickets, load_fixed, choose, FIXED_HARUQ_SHA256

class TwoStageTests(unittest.TestCase):
    def test_catalog_is_six_frozen_types(self):
        self.assertEqual([n for n,_,_ in CATALOG],[1,2,3,4,5,6])
        self.assertEqual(CATALOG[0][2],("tail","top1","mid2_5"))

    def test_seven_and_nine_exact_point_counts(self):
        self.assertEqual([len(rank_tickets(7,i)) for i in range(1,7)],[11,11,9,12,3,11])
        self.assertEqual([len(rank_tickets(9,i)) for i in range(1,7)],[16,16,12,16,3,16])

    def test_never_duplicate_a_vehicle(self):
        for n in range(5,10):
            for tid in range(1,7):
                combos=rank_tickets(n,tid)
                self.assertEqual(len(combos),len(set(combos)))
                self.assertTrue(all(len(set(x))==3 for x in combos))

    def test_no_retraining_and_real_model_locked(self):
        src=load_fixed(Path("artifacts/ura_ai_v1.json"))
        shape=json.loads(Path("artifacts/ura_shape_selector_v1.json").read_text())
        self.assertFalse(shape["base_stage_reestimated"])
        self.assertFalse(shape["purchase_approved"])
        self.assertEqual(shape["training_split"]["races"],242)
        self.assertEqual(shape["temporal_comparison"]["races"],180)
        source={"race_id":"dummy","date":"2026-10-24","model_sha256":FIXED_HARUQ_SHA256,
                "ranked":[{"bike":i,"score":0.2*(8-i)} for i in range(1,8)],"confidence":0.2}
        actual=choose(source,src,shape)
        self.assertTrue(actual["research_only"])
        self.assertFalse(actual["purchase_approved"])
        self.assertEqual(len(actual["other_types"]),5)
        selected=actual["selected_type"]
        self.assertEqual(len(selected["tickets"]),selected["points"])
        self.assertTrue(0.<=selected["estimated_capture_probability"]<=1.)

    def test_reject_source_model_fingerprint_mismatch(self):
        src=load_fixed("artifacts/ura_ai_v1.json")
        shape=json.loads(Path("artifacts/ura_shape_selector_v1.json").read_text())
        source={"model_sha256":"bad","ranked":[],"confidence":0}
        with self.assertRaises(ValueError):choose(source,src,shape)

    def test_high_gap_is_not_a_bet(self):
        src=load_fixed("artifacts/ura_ai_v1.json")
        shape=json.loads(Path("artifacts/ura_shape_selector_v1.json").read_text())
        scores=[3,1.5,0,0,0,0,0]
        source={"race_id":"test","model_sha256":FIXED_HARUQ_SHA256,
                "ranked":[{"bike":i,"score":scores[i-1]} for i in range(1,8)],
                "confidence":1.5}
        result=choose(source,src,shape)
        self.assertEqual(result["decision"],"NOT_ELIGIBLE")
        self.assertIsNone(result["selected_type"])
        self.assertFalse(result["purchase_approved"])

if __name__=="__main__":
    unittest.main()
