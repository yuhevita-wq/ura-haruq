import json
import unittest
from pathlib import Path
from ura.legacy.formation_v3_predict import research_prediction,bin_rank
from ura.legacy.formation_selector import load_fixed,FIXED_HARUQ_SHA256
from ura.core import low_rank_start

class TestV3Predict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ura=load_fixed("artifacts/ura_ai_v1.json")
        cls.v3=json.loads(Path("artifacts/ura_formation_v3_2023_compliant.json").read_text())

    def synthetic(self,n=7,gap=.2):
        return {"race_id":"synthetic", "model_sha256":FIXED_HARUQ_SHA256,
                "date":"2026-10-24",
                "confidence":gap,
                "ranked":[{"bike":i,"score":(7-i)*.2 if i>1 else (7-1)*.2+(gap-.2)}
                          for i in range(1,n+1)]}

    def test_7_8_9_field_all_tickets_are_unique_and_genuine_underdogs(self):
        for n in (7,8,9):
            source=self.synthetic(n)
            out=research_prediction(source,self.ura,self.v3)
            self.assertEqual(out["decision"],"RESEARCH_ONLY")
            self.assertFalse(out["purchase_approved"])
            self.assertEqual(out["points"],len(set(out["tickets"])))
            self.assertEqual(len(out["formations"]),5)
            rank={int(b):i+1 for i,b in enumerate(out["ura_probability_rank_bikes"])}
            for ticket in out["tickets"]:
                bike=[int(x) for x in ticket.split("-")]
                self.assertEqual(len(set(bike)),3)
                self.assertTrue(any(rank[v]>=low_rank_start(n) for v in bike))

    def test_no_stakes_on_high_confidence(self):
        source=self.synthetic()
        source["ranked"][0]["score"]+=1.
        source["confidence"]=1.2
        out=research_prediction(source,self.ura,self.v3)
        self.assertEqual(out["points"],0)
        self.assertEqual(out["decision"],"NOT_ELIGIBLE")

    def test_reject_tampered_model_fingerprints(self):
        with self.assertRaises(ValueError):
            research_prediction(self.synthetic(),self.ura,{**self.v3,"full_source_model_v3_json_sha256":"fake"})

    def test_rider_rank_bins(self):
        self.assertEqual([bin_rank(i,7) for i in range(1,8)],list(range(1,8)))
        self.assertEqual(bin_rank(9,9),7)
        self.assertEqual(self.v3["combined"]["2025"]["roi_percent"],87.8204)

if __name__=="__main__":
    unittest.main()
