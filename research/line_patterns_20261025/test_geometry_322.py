"""3-2-2 symmetric geometry tests. No operational HaruQ models touched."""
import sqlite3
import unittest

from research.line_patterns_20261025.geometry_322 import (
    role_code, load_322, same_line_mark, structural_key,
)

class Geometry322(unittest.TestCase):
    def setUp(self):
        c=sqlite3.connect(":memory:")
        c.row_factory=sqlite3.Row
        c.execute("CREATE TABLE entries (race_id TEXT,bike_number INTEGER,line_group TEXT,line_role TEXT)")
        self.conn=c
        self.rid="202301010001"
        for g,items in [("1",[(1,"自力"),(2,"番手"),(3,"3番手")]),
                        ("2",[(4,"自力"),(5,"番手")]),
                        ("3",[(6,"自力"),(7,"番手")])]:
            for b,role in items:
                c.execute("INSERT INTO entries VALUES (?,?,?,?)",(self.rid,b,g,role))

    def tearDown(self):self.conn.close()

    def test_roles(self):
        self.assertEqual(role_code("3番手"),"T")
        self.assertEqual(role_code("番手"),"B")
        self.assertEqual(role_code("自力"),"H")
        self.assertIsNone(role_code(""))

    def test_322_detected(self):
        lk,status=load_322(self.conn,self.rid,set(range(1,8)))
        self.assertEqual(status,"ok")
        self.assertEqual(lk[3][1],"3T")
        self.assertEqual(lk[7][1],"2B")

    def test_2line_swaps_are_symmetric(self):
        lookup,status=load_322(self.conn,self.rid,set(range(1,8)))
        self.assertEqual(structural_key((4,5,1),lookup),
                         structural_key((6,7,1),lookup))
        self.assertEqual(structural_key((4,6,1),lookup),
                         structural_key((6,4,1),lookup))
        self.assertNotEqual(structural_key((4,5,1),lookup),
                            structural_key((4,7,1),lookup))

    def test_geometry_is_not_ai_rank_based(self):
        lk,_=load_322(self.conn,self.rid,set(range(1,8)))
        self.assertEqual(structural_key((2,4,5),lk),"3B-2H-2B|23")
        self.assertEqual(structural_key((4,2,5),lk),"2H-3B-2B|13")
        self.assertEqual(structural_key((1,2,3),lk),"3H-3B-3T|123")

    def test_2line_three_way_same_is_impossible(self):
        self.assertEqual(same_line_mark(("a","a","b")),"12")
        self.assertEqual(same_line_mark(("b","a","a")),"23")
        self.assertEqual(same_line_mark(("a","b","a")),"13")
        self.assertEqual(same_line_mark(("a","b","c")),"X")

    def test_exact_7_rider_roster_required(self):
        lk,status=load_322(self.conn,self.rid,set(range(2,9)))
        self.assertIsNone(lk)
        self.assertEqual(status,"roster_disagrees")

if __name__=="__main__":
    unittest.main()
