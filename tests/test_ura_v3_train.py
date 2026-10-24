import unittest
import numpy as np
from ura.formation_v3_train import (all_candidates,candidate_occupancy,
       choose_forms,cube_sum,group_matrices,rank_bin,LATTICES)

class TestV3FormationLearning(unittest.TestCase):
    def test_all_cuboids_and_normalization(self):
        self.assertEqual(all_candidates().shape,(21952,6))
        self.assertEqual([rank_bin(i,7) for i in range(1,8)],list(range(1,8)))
        self.assertEqual(rank_bin(9,9),7)
        for n in (7,8,9):
            self.assertEqual(int(LATTICES[n].sum()),n*(n-1)*(n-2))

    def test_integral_box_sums_exactly(self):
        rng=np.random.default_rng(9)
        a=rng.uniform(size=(7,7,7))
        cands=all_candidates()
        byprefix=cube_sum(a,cands)
        for j in (0,1,210,1598,20000,21951):
            x=cands[j]
            brute=float(a[x[0]:x[1]+1,x[2]:x[3]+1,x[4]:x[5]+1].sum())
            self.assertAlmostEqual(float(byprefix[j]),brute,places=8)

    def test_candidate_region_has_same_cost(self):
        c=all_candidates()[358]
        mask=candidate_occupancy(c)
        self.assertEqual(int(mask.sum()),int((c[1]-c[0]+1)*(c[3]-c[2]+1)*(c[5]-c[4]+1)))

    def test_training_does_not_require_test_labels(self):
        # Synthetic payout-only 2023 and 2024 records; 2025 does not enter choose_forms.
        def data(year):
            return [{"race_id":f"{year}-{i}","date":f"{year}-01-01","year":year,"n":7,
                     "winner_ranks":(2,1,7),"winning_bin":(1,0,6),
                     "payout":5000} for i in range(20)]
        a=group_matrices(data(2023))
        b=group_matrices(data(2024))
        forms,mask,_=choose_forms(a,b)
        self.assertGreaterEqual(len(forms),1)
        self.assertTrue(mask[1,0,6])
        self.assertTrue(all(t["training_incremental_hits_2023"]>=5
                            and t["selection_incremental_hits_2024"]>=5 for t in forms))
        self.assertEqual(len({t["candidate_idx"] for t in forms}),len(forms))

if __name__=="__main__":
    unittest.main()
