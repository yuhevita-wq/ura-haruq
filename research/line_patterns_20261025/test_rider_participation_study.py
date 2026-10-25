import unittest
from research.line_patterns_20261025.rider_participation_study import (
    past_starts_band,miss_reasons,evidence_group,rider_frequency
)

class RiderDiagnosisTests(unittest.TestCase):
    def test_start_bins(self):
        self.assertEqual([past_starts_band(x) for x in (0,4,5,9,10,19,20)],
                         ["00-04","00-04","05-09","05-09","10-19","10-19","20+"])

    def test_misses_partition_with_no_hidden_results(self):
        rows=[{"LOW_band":"0.25-0.50","points":6,"hit":0,"winner_rank_has_5to7":0},
              {"LOW_band":"0.25-0.50","points":6,"hit":0,"winner_rank_has_5to7":1},
              {"LOW_band":"0.75-1.00","points":32,"hit":1,"winner_rank_has_5to7":1},
              {"LOW_band":"0.50-0.75","points":0,"hit":0,"winner_rank_has_5to7":1}]
        audit,hits=miss_reasons(rows)
        self.assertEqual(audit["active_low_band_races"],3)
        self.assertEqual(audit["paused_low_band_races"],1)
        self.assertEqual(audit["missing_geometry_but_eligible_lower_rank_podium"],1)
        self.assertEqual(audit["excluded_due_to_no_lower_rank_in_podium"],1)
        self.assertEqual(audit["matched_eligible_win"],1)
        self.assertEqual(len(hits),1)

    def test_player_start_frequency_not_raw_top3_count(self):
        rs=[{"rider_id":"aa","rider_name":"A","prefecture":"F",
             "date":"2026-07-01","ura_rank":5,"in_top3":1,"in_1st":0,
             "in_2nd":1,"in_3rd":0,"prior180_starts":7,"prior180_top3":3},
            {"rider_id":"aa","rider_name":"A","prefecture":"F",
             "date":"2026-08-01","ura_rank":5,"in_top3":0,"in_1st":0,
             "in_2nd":0,"in_3rd":0,"prior180_starts":8,"prior180_top3":3}]
        s=rider_frequency(rs)
        self.assertEqual(s[0]["rank5to7_top3_pct"],50.0)
        self.assertEqual(s[0]["starts_in_exact_LOW_322_sample"],2)

    def test_exposure_group_correct_denominator(self):
        a=[{"ura_rank":i,"in_top3":int(i==5),"in_1st":int(i==5),
            "in_2nd":0,"in_3rd":0,"prior180_starts":10,
            "prior180_top3":5} for i in (5,5,5,5)]
        out=evidence_group(a,"rider",lambda x:"R")
        self.assertEqual(out[0]["top3_rate_pct"],100)
        self.assertEqual(out[0]["avg_prior_180day_top3_rate_pct"],50.0)

if __name__=="__main__":unittest.main()
