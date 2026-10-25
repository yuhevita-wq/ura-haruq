"""Tests: payout/occurrence stability screen never cheats with future results."""
import unittest
from collections import Counter

from research.line_patterns_20261025.high_payout_stability import (
    annual_profile, payout_reference_2023, frequency_ratio, profile_and_gate
)

HIGH="2H-3H-3B|23"
CHEAP="2H-2B-3B|12"

def sample(shape,payout,year="2023",eligible=True, points=None):
    if points is None:points={HIGH:2,CHEAP:1}
    return {"date":year+"-01-01",
            "observed_structure":shape,
            "winner_type":shape if eligible else None,
            "payout":payout,"tickets":Counter(points)}


class TypeMedianAndStabilityTest(unittest.TestCase):
    def setUp(self):
        self.a=[sample(HIGH,10000) for _ in range(3)]+[sample(CHEAP,500) for _ in range(3)]
        self.b=[sample(HIGH,12000,"2024") for _ in range(3)]+[sample(CHEAP,600,"2024") for _ in range(3)]

    def test_reference_from_actual_observed_payout_not_mean(self):
        p=payout_reference_2023(self.a)
        self.assertEqual(p["cutoff_yen"],5250)
        self.assertEqual(p["reference_payout_n"],6)

    def test_predeclared_cutoff_keeps_high_and_discard_cheap(self):
        ref,catalog,retained=profile_and_gate(self.a,self.b)
        self.assertEqual(ref["cutoff_yen"],5250)
        self.assertEqual([x["type"] for x in retained],[HIGH])
        low=next(x for x in catalog if x["type"]==CHEAP)
        self.assertIn("2023_low_actual_median_payout",low["exclusion_reasons"])
        self.assertIn("2024_low_actual_median_payout",low["exclusion_reasons"])

    def test_points_count_all_race_not_only_wins(self):
        p=annual_profile(self.a,HIGH)
        self.assertEqual(p["required_total_points"],12)
        self.assertEqual(p["required_mean_points_per_race"],2)
        self.assertEqual(p["observed_wins_all"],3)
        self.assertEqual(p["payout_median_all_yen"],10000)
        self.assertEqual(p["paid_yen"],30000)
        self.assertEqual(p["roi_pct"],2500)

    def test_realized_low_payout_cannot_be_deleted_after_selection(self):
        x=[sample(HIGH,15000),sample(HIGH,15000),sample(HIGH,15000),
           sample(HIGH,100),sample(CHEAP,500),sample(CHEAP,500)]
        prof=annual_profile(x,HIGH)
        self.assertEqual(prof["paid_yen"],45100)
        self.assertEqual(prof["eligible_hits"],4)
        self.assertGreater(prof["payout_median_all_yen"],5000)

    def test_uneven_year_frequency_is_rejected(self):
        a=[sample(HIGH,10000) for _ in range(4)]+[sample(CHEAP,1000) for _ in range(4)]
        b=[sample(HIGH,11000,"2024") for _ in range(3)]+[sample(CHEAP,1000,"2024") for _ in range(30)]
        ref,catalog,retained=profile_and_gate(a,b)
        high=next(x for x in catalog if x["type"]==HIGH)
        self.assertIn("hit_frequency_unstable_2023_to_2024",high["exclusion_reasons"])
        self.assertNotIn(HIGH,[x["type"] for x in retained])

    def test_missing_2024_hits_does_not_use_2025_data(self):
        a=[sample(HIGH,10000) for _ in range(4)]+[sample(CHEAP,500) for _ in range(4)]
        b=[sample(HIGH,12000,"2024",eligible=False) for _ in range(4)]
        _,catalog,retained=profile_and_gate(a,b)
        hi=next(x for x in catalog if x["type"]==HIGH)
        self.assertIn("2024_fewer_than_3_eligible_hits",hi["exclusion_reasons"])
        self.assertNotIn(HIGH,[x["type"] for x in retained])

if __name__=="__main__":
    unittest.main()
