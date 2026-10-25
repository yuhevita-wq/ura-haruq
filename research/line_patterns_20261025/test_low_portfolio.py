"""Non-regression tests for LOW-conditional multi-type geometry portfolio."""
import unittest
from collections import Counter
from research.line_patterns_20261025.low_portfolio import (
    select_band, aggregate, group_for, summary, BANDS
)


def case(year, band, shape, tickets, payout, win=True):
    return {"race_id":year+"x","date":year+"-02-01","split":year,"LOW_band":band,
            "tickets":Counter(tickets),"winner_type":shape if win else None,
            "observed_structure":shape,"payout":payout}


class LowPortfolioTests(unittest.TestCase):
    def test_underdog_disjoint_type_union(self):
        rows=[case("2023",BANDS[0],"a",{"a":2,"b":3},550),
              case("2023",BANDS[0],"b",{"a":2,"b":3},650)]
        a=aggregate(rows,["a"]);b=aggregate(rows,["b"])
        union=aggregate(rows,["a","b"])
        self.assertEqual(union["points"],a["points"]+b["points"])
        self.assertEqual(union["payout_yen"],a["payout_yen"]+b["payout_yen"])
        self.assertEqual(union["hits"],a["hits"]+b["hits"])
        self.assertEqual(union["net_yen"],a["net_yen"]+b["net_yen"])

    def test_all_types_losing_cannot_form_profit(self):
        rows=[case("2023",BANDS[0],"a",{"a":2,"b":1},100),
              case("2023",BANDS[0],"b",{"a":2,"b":1},100)]
        p=aggregate(rows,["a","b"])
        self.assertLess(p["roi_pct"],100)
        self.assertLess(aggregate(rows,["a"])["roi_pct"],100)
        self.assertLess(aggregate(rows,["b"])["roi_pct"],100)

    def test_2023_2024_select_near_profit_loss_types(self):
        a=[case("2023",BANDS[1],"a",{"a":1,"b":1},80),
           case("2023",BANDS[1],"a",{"a":1,"b":1},90),
           case("2023",BANDS[1],"b",{"a":1,"b":1},80),
           case("2023",BANDS[1],"b",{"a":1,"b":1},90)]
        b=[case("2024",BANDS[1],"a",{"a":1,"b":1},70),
           case("2024",BANDS[1],"a",{"a":1,"b":1},75),
           case("2024",BANDS[1],"b",{"a":1,"b":1},70),
           case("2024",BANDS[1],"b",{"a":1,"b":1},75)]
        shapes=select_band(a,b)["selected_types"]
        self.assertEqual(set(shapes),{"a","b"})

    def test_screen_with_zero_hits_not_adopted(self):
        a=[case("2023",BANDS[0],"a",{"a":1},500) for _ in range(3)]
        b=[case("2024",BANDS[0],"a",{"a":1},0,win=False) for _ in range(3)]
        self.assertEqual(select_band(a,b)["selected_types"],[])

    def test_year_split_and_no_label_time_crossing(self):
        self.assertEqual(group_for({"year":"2026","date":"2026-03-31"}),"2026Q1")
        self.assertEqual(group_for({"year":"2026","date":"2026-04-01"}),"2026Q2")
        with self.assertRaises(ValueError):
            group_for({"year":"2026","date":"2026-07-01"})

    def test_empty_band_returns_unpurchased_not_fabricated(self):
        self.assertEqual(select_band([],[])["selected_types"],[])
        s=summary([],[])
        self.assertEqual(s["races"],0)
        self.assertIsNone(s["roi_pct"])


if __name__=="__main__":
    unittest.main()
