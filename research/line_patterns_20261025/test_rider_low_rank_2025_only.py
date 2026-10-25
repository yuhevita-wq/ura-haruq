import unittest
from research.line_patterns_20261025.rider_low_rank_2025_only import (
    wilson_lower, stats, make_exposures, per_rider, split_by_band
)

class RiderOnly2025Tests(unittest.TestCase):
    def test_interval_wilson(self):
        self.assertEqual(wilson_lower(0,0),0)
        self.assertEqual(wilson_lower(0,100),0)
        self.assertTrue(.1<wilson_lower(1,1)<.3)
        self.assertLess(wilson_lower(1,1),wilson_lower(10,10))
        self.assertLess(wilson_lower(5,10),.5)

    def test_exactly_three_reduced_rank_riders_per_race(self):
        players=[{"bike_number":i,"rider_id":str(i),
                  "official_name":str(i),"rider_name":str(i),"prefecture":"A",
                  "line_role":"自力"} for i in range(1,8)]
        a=make_exposures([{"race_id":"202501010101","date":"2025-01-01",
               "gap":0.33,"LOW_band":"0.25-0.50","is_322":False,
               "ranks":[1,2,3,4,5,6,7],"finish":[5,6,1],
               "entries":players,"lookup":None}])
        self.assertEqual(len(a),3)
        self.assertEqual([x["ura_rank"] for x in a],[5,6,7])
        self.assertEqual([x["finish_position"] for x in a],[1,2,""])
        self.assertEqual(sum(x["top3"] for x in a),2)
        self.assertEqual(sum(x["second"] for x in a),1)

    def test_exposure_denominator_avoids_1of1_advantage(self):
        def cell(pid,i,n,hit):
            return {"rider_id":pid,"name":pid,"prefecture":"",
                    "ura_rank":5,"top3":int(hit),"first":0,
                    "second":int(hit),"third":0,"line_role":"2B",
                    "date":"2025-02-01","half":"H1","LOW_band":"0.25-0.50"}
        rs=[cell("one",0,1,True)]+[cell("many",i,12,i<9) for i in range(12)]
        out=per_rider(rs,"all_7car_LOW")
        one=next(x for x in out if x["rider_id"]=="one")
        many=next(x for x in out if x["rider_id"]=="many")
        self.assertEqual(one["low_rank_starts"],1)
        self.assertEqual(many["low_rank_starts"],12)
        self.assertGreater(many["top3_wilson95_lower_pct"],one["top3_wilson95_lower_pct"])

    def test_finish_position_is_separate_from_all_placements(self):
        a=[{"ura_rank":i,"top3":int(i!=7),
            "first":int(i==5),"second":int(i==6),"third":0,
            "half":"H1","date":"2025-01-01","LOW_band":"0.00-0.25",
            "line_role":"3B"} for i in (5,6,7)]
        x=stats(a)
        self.assertEqual((x["low_rank_starts"],x["top3_count"]), (3,2))
        self.assertEqual((x["first_count"],x["second_count"],x["third_count"]), (1,1,0))

if __name__=="__main__":unittest.main()
