"""New active policy tests. All prices/scores here are synthetic fixtures."""
import copy,itertools,unittest
from unittest.mock import patch
from ura.manual_formations import (FIXED_MODEL_SHA,load_probability_model,expand,
    build_formations,evaluate_set,evaluate_race,parse_lines)


def fixture(n=7):
    bikes=list(range(1,n+1))
    return {'race_id':'synthetic-test','grade':'F1','race_class':'A級1・2班',
        'model_sha256':FIXED_MODEL_SHA,'confidence':.1,
        'ranked':[{'bike':b,'score':round(.8-i*.1,4)} for i,b in enumerate(bikes)],
        'formation':'1-2-3/4-5/6-7' if n==7 else '1-2-3/4-5-6/7-8-9',
        'trifecta_odds':{'-'.join(map(str,t)):100000. for t in itertools.permutations(bikes,3)},
        'start_at':'2026-11-01T12:00:00+09:00','prediction_at':'2026-11-01T11:30:00+09:00',
        'lineup_at':'2026-11-01T11:29:00+09:00','odds_observed_at':'2026-11-01T11:31:00+09:00',
        'odds_kind':'pre_start_snapshot'}


class ManualPolicyTests(unittest.TestCase):
    def test_manual_exact_counts_and_union(self):
        active=range(1,10)
        self.assertEqual(len(expand([[1],[2,4,5,9],[2,4,5,9]],active)),12)
        self.assertEqual(len(expand([[1,7],[1,7],[2,3,5,9]],active)),8)
        self.assertEqual(len(expand([[5],active,[1]],active)),7)
        box=expand([[1,3,5,7]]*3,active)
        self.assertEqual(len(box|expand([[5],active,[1]],active)),29)

    def test_shapes_are_whole_and_include_lower_rider(self):
        lines=[[1,2,3],[4,5],[6,7]]
        cs=build_formations(range(1,8),lines,{5,6,7})
        self.assertTrue({'first_fixed','second_fixed','third_fixed','pair_reverse',
            'ordered_pair','sandwich','three_rider_box','multiple_scenarios','pair_plus_sandwich'} <= {c['family'] for c in cs})
        self.assertGreater(len({c['points'] for c in cs}),3)
        for c in cs:
            full=set(t for b in c['blocks'] for t in expand(b['fields'],range(1,8)))
            self.assertEqual(full,set(c['tickets']))
            self.assertTrue(all(set(t)&{5,6,7} for t in full))

    def test_full_set_ev_keeps_negative_component(self):
        a,b=(1,2,3),(1,3,2)
        m=evaluate_set([a,b],{a:.05,b:.01},{a:60,b:10})
        self.assertAlmostEqual(m['estimated_roi'],1.55)
        self.assertAlmostEqual(m['estimated_net_yen'],110)
        self.assertEqual(m['investment_yen'],200)

    def test_active_replacement_works_outside_old_2221_and_six_points(self):
        x=fixture();x['competition_scores_by_bike']={b:50+b*20 for b in range(1,8)}
        with patch('ura.legacy.frozen_v1.formation',side_effect=AssertionError('old called')):
            out=evaluate_race(x)
        self.assertEqual(out['decision'],'BET')
        self.assertFalse(out['legacy_formations_used'])
        self.assertNotEqual(out['points'],6)
        self.assertEqual(out['stake_yen'],100*out['points'])
        self.assertEqual(out['tickets'],out['selected']['tickets'])
        self.assertFalse(out['automatic_purchase'])

    def test_nine_riders(self):
        out=evaluate_race(fixture(9));self.assertGreater(out['candidate_count'],0)
        self.assertTrue(all(any(int(b)>=6 for b in t.split('-')) for t in out['tickets']))

    def test_no_odds_does_not_select_by_probability_alone(self):
        x=fixture();x.pop('trifecta_odds')
        out=evaluate_race(x)
        self.assertEqual(out['reason'],'MISSING_ODDS');self.assertIsNone(out['selected'])
        self.assertGreater(len(out['candidates']),0);self.assertEqual(out['tickets'],[])

    def test_no_whole_set_edge(self):
        x=fixture();x['trifecta_odds']={k:1.1 for k in x['trifecta_odds']}
        self.assertEqual(evaluate_race(x)['reason'],'NO_WHOLE_FORMATION_PRICE_EDGE')

    def test_rank_only_input_has_no_old_fallback(self):
        x=fixture();x.pop('ranked');x['predicted_bikes_by_rank']=list(range(1,8))
        self.assertEqual(evaluate_race(x)['reason'],'FULL_SCORE_SNAPSHOT_REQUIRED_NO_LEGACY_RANK_FALLBACK')

    def test_timing_and_price_provenance(self):
        for key in ['prediction_at','lineup_at','odds_observed_at']:
            x=fixture();x[key]=x['start_at']
            self.assertEqual(evaluate_race(x)['reason'],'INPUT_NOT_CAPTURED_BEFORE_START')
        x=fixture();x.pop('odds_observed_at')
        self.assertEqual(evaluate_race(x)['reason'],'MISSING_PRESTART_TIMESTAMPS')
        x=fixture();x['odds_kind']='final'
        self.assertEqual(evaluate_race(x)['reason'],'FINAL_OR_UNVERIFIED_ODDS_NOT_PURCHASE_TIME_PRICES')

    def test_missing_rider_price_lineup_and_model_stop(self):
        x=fixture();x['trifecta_odds'].pop('1-2-3')
        with self.assertRaises(ValueError):evaluate_race(x)
        x=fixture();x['formation']='1-2/3-4/5-6'
        with self.assertRaises(ValueError):evaluate_race(x)
        x=fixture();x['model_sha256']='wrong'
        with self.assertRaises(ValueError):evaluate_race(x)
        x=fixture();bad=copy.deepcopy(load_probability_model());bad['parameters'][0]+=1
        with self.assertRaises(ValueError):evaluate_race(x,bad)

    def test_old_production_calls_are_retired(self):
        from ura.participant_v2 import decide
        from ura.frozen_v1 import formation
        from ura.market_shapes_v1 import evaluate
        for f in [decide,formation,evaluate]:
            with self.assertRaises(RuntimeError):f()

    def test_integration_uses_manual_as_only_official_source(self):
        from ura.integrated_signals import evaluate_race as integrated
        out=integrated(fixture())
        self.assertEqual(out['official_purchase_source'],'manual_formations_only')
        self.assertEqual(out['official_tickets'],out['selected']['tickets'])
        self.assertNotIn('adopted_v2',out)

class IndependentProbabilityCompatibilityTests(unittest.TestCase):
    def test_independent_model_never_uses_original_coefficients(self):
        x=fixture();x['model_role']='INDEPENDENT_CHRONOLOGICAL_2026';x['model_sha256']='independent-test-sha'
        out=evaluate_race(x)
        self.assertEqual(out['reason'],'MISSING_COMPATIBLE_INDEPENDENT_PROBABILITIES')
        self.assertGreater(len(out['candidates']),0)
        self.assertEqual(out['tickets'],[])

    def test_compatible_joint_snapshot_and_changed_scores(self):
        import hashlib,json
        x=fixture();x['model_role']='INDEPENDENT_CHRONOLOGICAL_2026';x['model_sha256']='independent-test-sha'
        x['trifecta_probabilities']={k:1/210 for k in x['trifecta_odds']}
        rank_hash=hashlib.sha256(json.dumps(x['ranked'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        x['probability_provenance']={'race_id':x['race_id'],'source_model_sha256':x['model_sha256'],
            'ranked_scores_sha256':rank_hash,'probability_model_id':'synthetic-test-only',
            'trained_on_results':True,'odds_used_as_teacher':False}
        self.assertEqual(evaluate_race(x)['decision'],'BET')
        x['ranked'][-1]['score']-=.01
        with self.assertRaises(ValueError):evaluate_race(x)
