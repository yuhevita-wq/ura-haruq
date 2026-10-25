import unittest
from research.line_patterns_20261025.evaluate_frozen_4_20_julaug import (
  audit_policy, observed_result, payout_index, canonical_id, clean_name, summarize,
)

def policy():
    return {
       "policy_id":"URA_322_HIGH_FIRST_4_20_FROZEN_20261025",
       "research_only":True,"selection_uses_period_for_future_test":False,
       "period_for_future_test":{"start":"2026-07-01","end_exclusive":"2026-08-31"},
       "LOW_types":{"0.00-0.25":[],"0.25-0.50":["a","b","c","d"],
                    "0.50-0.75":[],"0.75-1.00":[str(x) for x in range(20)]}
    }

class SeparatePeriodTests(unittest.TestCase):
    def test_4_and_20_policy_is_unchanged(self):
        self.assertEqual(list(map(len,audit_policy(policy()).values())),[0,4,0,20])

    def test_future_leakage_switch_rejected(self):
        p=policy();p["selection_uses_period_for_future_test"]=True
        with self.assertRaises(ValueError):audit_policy(p)

    def test_cannot_replace_4_with_5(self):
        p=policy();p["LOW_types"]["0.25-0.50"].append("e")
        with self.assertRaises(ValueError):audit_policy(p)

    def test_duplicate_types_rejected(self):
        p=policy();p["LOW_types"]["0.25-0.50"][1]="a"
        with self.assertRaises(ValueError):audit_policy(p)

    def test_dead_heat_not_fake_unique_places(self):
        winner=observed_result(
             [{"finish_position":str(i),"car_no":str(i)}
              for i in [1,2,2,4,5,6,7]],7)
        self.assertIsNone(winner)

    def test_unique_positions_return_exact_win(self):
        winner=observed_result(
             [{"finish_position":str(i),"car_no":str(8-i)}
              for i in range(1,8)],7)
        self.assertEqual(winner,"7-6-5")

    def test_multiple_ticket_wins_not_choose_best_after_result(self):
        a=[{"race_id":"r","ticket_type":"3連単","status":"paid",
            "combination":"7-6-5","payout_yen":"12340"},
           {"race_id":"r","ticket_type":"3連単","status":"paid",
            "combination":"7-6-4","payout_yen":"3000"}]
        self.assertEqual(len(payout_index(a)["r"]),2)

    def test_archive_race_id_conversion(self):
        self.assertEqual(canonical_id("2520260701010006"),"202607012506")

    def test_name_spaces(self):
        self.assertEqual(clean_name("熊崎 麻人"),"熊崎麻人")

    def test_no_tickets_produce_no_invented_roi(self):
        out=summarize([{"date":"2026-07-01","points":0,"hit":0,"returned_yen":0}])
        self.assertIsNone(out["roi_pct"])

    def test_hit_metric_math(self):
        arr=[{"date":"2026-07-01","points":2,"hit":1,"returned_yen":1500},
             {"date":"2026-07-02","points":2,"hit":0,"returned_yen":0}]
        out=summarize(arr)
        self.assertEqual(out["points"],4)
        self.assertEqual(out["roi_pct"],375.0)
        self.assertEqual(out["break_even_avg_payout_per_hit_yen"],400)
if __name__=="__main__":unittest.main()
