import unittest
from ura.roi_holdout import odds_race_id

class OddsIDTests(unittest.TestCase):
    def test_first_day(self):
        self.assertEqual(odds_race_id("1120260620010006"),"202606201106")
    def test_second_day(self):
        self.assertEqual(odds_race_id("1120260620020006"),"202606211106")
    def test_third_day(self):
        self.assertEqual(odds_race_id("1120260620030006"),"202606221106")
    def test_month_boundary(self):
        self.assertEqual(odds_race_id("1220260630030012"),"202607021212")
    def test_invalid_day(self):
        with self.assertRaises(ValueError):odds_race_id("1120260620000006")

if __name__=="__main__":unittest.main()
