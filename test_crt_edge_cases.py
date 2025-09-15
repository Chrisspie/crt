# -*- coding: utf-8 -*-
import pandas as pd
from crt_core import crt_scan, EPS

def _mk(rows):
    idx = pd.date_range("2024-01-05", periods=len(rows), freq="W-FRI")
    df = pd.DataFrame(rows, columns=["Open","High","Low","Close"], index=idx)
    return df

def test_uppercase_confirm_method_ok():
    rows = [
        [100, 110, 90, 100],  # C1
        [95, 105, 85, 95],    # C2 bull (sweep low, close back in range)
        [100, 113, 98, 112],  # C3 confirm by close > C1H
    ]
    df = _mk(rows)
    rec = crt_scan(df, lookback_bars=10, confirm_within=2, confirm_method="CLOSE", directions=("bullish",))
    assert rec and rec[0]["confirmed"] is True

def test_dual_sweep_skipped_by_default():
    rows = [
        [100, 110, 90, 100],  # C1
        [95, 115, 85, 100],   # C2: L<C1L and H>C1H, Close in range
    ]
    df = _mk(rows)
    recs = crt_scan(df, lookback_bars=5, confirm_within=0, directions=("bullish","bearish"))
    assert recs == [], "Ambiguous dual-sweep should be skipped by default"

def test_epsilon_edges_in_range_and_sweep():
    eps = EPS
    rows = [
        [100, 110, 90, 100],                # C1
        [95, 105, 90.0 + eps/2, 110.0],     # C2 close == C1H (in-range with eps)
    ]
    df = _mk(rows)
    out = crt_scan(df, lookback_bars=5, confirm_within=0, directions=("bearish",))
    assert out == []

def test_c1_window_multi_bar_range():
    rows = [
        [100, 120, 90, 110],  # i-3
        [110, 115, 85, 100],  # i-2
        [100, 105, 80, 100],  # i-1 (C1 window end)
        [95, 104, 79, 100],   # i (C2): L<80, close 100 within [80,120] -> BULL
    ]
    df = _mk(rows)
    recs = crt_scan(df, lookback_bars=10, confirm_within=0, directions=("bullish",), c1_window_bars=3)
    assert len(recs) == 1 and recs[0]["direction"] == "BULL"
