import copy,itertools,unittest
from tests.test_manual_formations import fixture
from ura.manual_formations import evaluate_race,load_probability_model,compare_priced_formations,evaluate_set
from ura.replay_manual_2023 import forecast,settle,netid

class AnnualReplayContracts(unittest.TestCase):
    def test_shared_selection_matches_live_but_not_timestamps(self):
        source=fixture();row={'ranked':source['ranked'],'score_gap':source['confidence'],
            'line_formation':[[1,2,3],[4,5],[6,7]]}
        live=evaluate_race(source);replay=forecast(row,source['trifecta_odds'],load_probability_model())
        self.assertEqual(live['tickets'],replay['tickets'])
        self.assertEqual(live['points'],replay['points'])
        self.assertEqual(live['selected']['metrics'],replay['selected']['metrics'])
        self.assertFalse(replay['counts_as_live_purchase'])
        source['odds_kind']='final'
        self.assertEqual(evaluate_race(source)['reason'],'FINAL_OR_UNVERIFIED_ODDS_NOT_PURCHASE_TIME_PRICES')
    def test_no_result_in_selection_input(self):
        source=fixture();row={'ranked':source['ranked'],'score_gap':.1,'line_formation':[[1,2,3],[4,5],[6,7]]}
        first=forecast(row,source['trifecta_odds'],load_probability_model())
        row['actual']=[7,6,5];row['payout']=1000000
        self.assertEqual(first,forecast(row,source['trifecta_odds'],load_probability_model()))
    def test_missing_payout_is_not_a_loss(self):
        d={'decision':'BET','tickets':['1-2-3'],'stake_yen':100}
        results=[{'bike_number':i,'finish_position':i} for i in (1,2,3)]
        self.assertIsNone(settle(d,results,{}, {'1-2-3':20})['returned_yen'])
        m=settle(d,results,{'sanrentan_combo':'1-2-3','sanrentan_payout':2000},{'1-2-3':30})
        self.assertEqual(m['status'],'final_quote_payout_mismatch')
        self.assertIsNone(m['investment_yen'])
    def test_source_calendar_id(self):
        self.assertEqual(netid('1120230427020006'),'202304281106')
