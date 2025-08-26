import unittest
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from crt_core import crt_scan


def make_df(rows):
    idx = pd.date_range("2024-01-05", periods=len(rows), freq="W-FRI")
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"])  # type: ignore
    df.index = idx
    return df


class TestC3Confirmation(unittest.TestCase):
    def test_bullish_close_rule_not_confirmed_when_no_close_above_c1h(self):
        # C1: [90, 110]; C2: sweep low and close back in range
        rows = [
            [100, 105, 80, 100],        # 0 - noise (so i=1 does NOT qualify)
            [100, 110, 90, 100],        # 1 - C1
            [95, 105, 85, 95],          # 2 - C2 (bullish: L < C1L and close back in)
            [95, 109, 95, 100],         # 3 - no close > C1H
            [100, 108, 88, 89],         # 4 - close < C1L (should NOT confirm bullish 'close>C1H')
        ]
        df = make_df(rows)

        setups = crt_scan(
            df,
            lookback_bars=10,
            confirm_within=3,
            confirm_method="close",
            directions=("bullish",),
        )

        self.assertEqual(len(setups), 1, "Expected one bullish setup detected")
        rec = setups[0]

        self.assertEqual(rec["confirm_rule"], "close>C1H in 3")
        self.assertFalse(rec["confirmed"], "Should not confirm without Close>C1H")
        self.assertFalse(rec["c3_happened"], "No C3 anywhere (no Close>C1H at all)")

    def test_bullish_close_rule_confirmed_when_close_above_c1h_within_n(self):
        # C1: [90, 110]; C2: sweep low and close back in range
        # Then Close 112 (> C1H=110) within 2 bars
        rows = [
            [100, 105, 80, 100],        # 0 - noise (so i=1 does NOT qualify)
            [100, 110, 90, 100],        # 1 - C1
            [95, 105, 85, 95],          # 2 - C2 (bullish)
            [100, 113, 98, 112],        # 3 - C3 by close (Close > C1H)
            [100, 108, 100, 100],       # 4 - trailing bar (does NOT form new setup)
        ]
        df = make_df(rows)

        setups = crt_scan(
            df,
            lookback_bars=10,
            confirm_within=2,
            confirm_method="close",
            directions=("bullish",),
        )

        self.assertEqual(len(setups), 1)
        rec = setups[0]

        self.assertEqual(rec["confirm_rule"], "close>C1H in 2")
        self.assertTrue(rec["confirmed"]) 
        self.assertTrue(rec["c3_happened"]) 


if __name__ == "__main__":
    unittest.main()
