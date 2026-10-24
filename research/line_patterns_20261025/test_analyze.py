"""Safety and consistency tests for the independent line-pattern study."""
import sqlite3
import unittest

from research.line_patterns_20261025.analyze import (
    CONTROL, enrich, get_groups, load_lineup, low_band, describe,
)


def db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    for name in ("entries", "race_results"):
        conn.execute(f"""CREATE TABLE {name} (
            race_id TEXT,bike_number INTEGER,line_group TEXT,line_role TEXT
        )""")
    return conn


class LinesTest(unittest.TestCase):
    def setUp(self):
        self.conn = db()
        self.row = {
            "race_id": "202301010001", "date": "2023-01-01", "year": "2023",
            "n": "7", "ranked_bikes": "1-2-3-4-5-6-7",
            "actual_bikes": "6-1-3", "actual_ranks": "6-1-3",
            "gap": ".22", "payout_yen": "12340", "grade": "F1",
        }

    def tearDown(self):
        self.conn.close()

    def insert(self, table, groups):
        for i, bikes in enumerate(groups, 1):
            for pos, bike in enumerate(bikes):
                self.conn.execute(
                    f"INSERT INTO {table} VALUES(?,?,?,?)",
                    (self.row["race_id"], bike, str(i),
                     ("自力", "番手", "3番手")[min(pos, 2)]),
                )

    def test_control_never_fitted(self):
        self.assertEqual(CONTROL, "2-2-2-1")

    def test_partition_and_roles(self):
        self.insert("entries", [(1, 2, 3), (4, 5), (6, 7)])
        groups, source = load_lineup(self.conn, self.row["race_id"], set(range(1, 8)))
        self.assertEqual(sorted(map(len, groups), reverse=True), [3, 2, 2])
        self.assertEqual(source, "entries")
        x, _ = enrich(self.conn, self.row)
        self.assertEqual(x["pattern"], "3-2-2")
        self.assertEqual((x["lower_1"], x["lower_2"], x["lower_3"]), (1, 0, 0))
        self.assertEqual(x["lower_any"], 1)
        self.assertEqual(x["line_12"], 0)
        self.assertEqual(x["line_23"], 1)
        self.assertEqual(x["weak_1"], 1)
        self.assertEqual(x["weak_mean_tied"], 0)
        self.assertEqual(x["gap_band"], "0.00-0.25")

    def test_2221_is_classified_for_control_only(self):
        self.insert("entries", [(1, 2), (3, 4), (5, 6), (7,)])
        x, _ = enrich(self.conn, self.row)
        self.assertEqual(x["pattern"], "2-2-2-1")
        self.assertEqual(x["lower_1"], 1)

    def test_result_fallback_marked_separately(self):
        self.insert("race_results", [(1, 2, 3), (4, 5), (6, 7)])
        x, source = enrich(self.conn, self.row)
        self.assertEqual(x["line_source"], "result_page_lineup")
        self.assertEqual(source, "result_page_lineup")

    def test_conflicting_lineup_excluded(self):
        self.insert("entries", [(1, 2, 3), (4, 5), (6, 7)])
        self.insert("race_results", [(1, 2), (3, 4, 5), (6, 7)])
        x, why = enrich(self.conn, self.row)
        self.assertIsNone(x)
        self.assertEqual(why, "entries_results_disagree")

    def test_incomplete_lineup_never_guessed(self):
        self.insert("entries", [(1, 2, 3), (4, 5), (6,)])
        x, why = enrich(self.conn, self.row)
        self.assertIsNone(x)
        self.assertEqual(why, "no_complete_lineup")

    def test_predictions_not_derived_from_finisher(self):
        self.insert("entries", [(1, 2, 3), (4, 5), (6, 7)])
        modified = dict(self.row, actual_bikes="1-6-3")
        with self.assertRaisesRegex(ValueError, "inconsistency"):
            enrich(self.conn, modified)

    def test_low_ranges_exact(self):
        self.assertEqual(low_band(.25), "0.25-0.50")
        self.assertEqual(low_band(.5), "0.50-0.75")
        self.assertEqual(low_band(.75), "0.75-1.00")
        with self.assertRaises(ValueError):
            low_band(1.0)

    def test_zero_subset_statistics(self):
        self.assertEqual(describe([])["races"], 0)
        self.assertIsNone(describe([])["median_3rentan_yen"])


if __name__ == "__main__":
    unittest.main()
