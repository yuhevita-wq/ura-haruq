import unittest
import numpy as np
from scipy.optimize import approx_fprime
from ura.core import (DEFAULT_COEFFICIENTS,low_rank_start,ordered_probabilities,
                      ordered_logloss_and_gradient,prepare,select_candidates)
def scored(n,gap=0.3):
    return [{"bike_number":i+1,
             "score":(gap if i==0 else 0.0)-0.025*max(i-1,0)}
            for i in range(n)]
class UraCoreTests(unittest.TestCase):
    def test_lower_rank_seven(self):self.assertEqual(low_rank_start(7),5)
    def test_lower_rank_nine(self):self.assertEqual(low_rank_start(9),6)
    def test_all_probabilities_sum_one(self):
        _,x,_,_=prepare(scored(7))
        p=ordered_probabilities(x,DEFAULT_COEFFICIENTS)
        self.assertEqual(len(p),210)
        self.assertAlmostEqual(sum(p.values()),1,places=9)
    def test_low_race_all_186_tail_combinations(self):
        p=select_candidates(scored(7),DEFAULT_COEFFICIENTS)
        self.assertTrue(p["low_confidence"])
        self.assertEqual(p["candidate_count"],186)
        self.assertEqual(len({x["ticket"] for x in p["candidates"]}),186)
        self.assertTrue(all(x["lower_rank_roles"] for x in p["candidates"]))
    def test_high_race_not_targeted(self):
        p=select_candidates(scored(7,gap=3.0),DEFAULT_COEFFICIENTS)
        self.assertFalse(p["low_confidence"])
        self.assertEqual(p["candidate_count"],0)
    def test_nine_car_tail_combinations(self):
        p=select_candidates(scored(9),DEFAULT_COEFFICIENTS)
        self.assertEqual(p["candidate_count"],444)
    def test_duplicate_bikes_rejected(self):
        rows=scored(7);rows[2]["bike_number"]=1
        with self.assertRaises(ValueError):prepare(rows)
    def test_invalid_rider_number_rejected(self):
        with self.assertRaises(ValueError):prepare(scored(4))
    def test_parameter_gradient(self):
        _,x,_,_=prepare(scored(7))
        group=[(x,(0,3,5)),(x,(6,1,0))]
        t=DEFAULT_COEFFICIENTS.copy()+np.array([0.12,0.02,-0.06,0.21,0.07])
        analytic=ordered_logloss_and_gradient(t,group,regularization=0.1)[1]
        numeric=approx_fprime(t,lambda v:ordered_logloss_and_gradient(
            v,group,regularization=0.1)[0],1e-6)
        self.assertTrue(np.allclose(analytic,numeric,rtol=2e-4,atol=2e-4))
    def test_probability_mass_for_five_car(self):
        _,x,_,_=prepare(scored(5))
        p=ordered_probabilities(x,DEFAULT_COEFFICIENTS)
        self.assertEqual(len(p),60)
        self.assertAlmostEqual(sum(p.values()),1,places=9)
if __name__=="__main__":unittest.main()
