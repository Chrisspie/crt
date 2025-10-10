
# -*- coding: utf-8 -*-
import math, pandas as pd, pytest
from datetime import datetime, timedelta

crt_core = pytest.importorskip("crt_core")
crt_scan = crt_core.crt_scan

# Helpers
def mk_idx(start="2024-03-01", n=12, freq="H"):
    return pd.date_range(start, periods=n, freq=freq)

def df_from_rows(rows, start="2024-03-01", freq="H"):
    idx = mk_idx(start=start, n=len(rows), freq=freq)
    return pd.DataFrame(rows, columns=["Open","High","Low","Close"], index=idx)

# ----------------------------
# 1) INSIDE BAR boost after C1
# ----------------------------
@pytest.mark.xfail(reason="Inside Bar detection + probability boost not implemented yet")
def test_inside_bar_series_boost_flag():
    # C1 wide range; then 3 Inside Bars fully inside C1; then sweep & revert (C2)
    c1 = [100, 110, 90, 100]
    ib1 = [100, 105, 95, 100]
    ib2 = [100, 104, 96, 100]
    ib3 = [100, 103, 97, 100]
    # C2 bullish: sweep below C1L and close back in the range
    c2  = [98, 102, 89, 99]
    rows = [[100,101,99,100], c1, ib1, ib2, ib3, c2, [100,105,95,102]]
    df = df_from_rows(rows, freq="H")
    recs = crt_scan(df, lookback_bars=20, directions=("bullish",))
    # Expect a record with an explicit IB boost flag / counter
    r = next((r for r in recs if r.get("direction")=="bullish"), None)
    assert r, "Expected a bullish CRT record"
    assert r.get("ib_series_len", 0) >= 3
    assert r.get("prob_boost_inside_bar") is True

# ----------------------------
# 2) HTF POI (OB/FVG/iFVG/BPR)
# ----------------------------
@pytest.mark.xfail(reason="HTF POI (OB/FVG/iFVG/BPR) filter not implemented")
def test_htf_poi_filter_increases_confidence():
    # Minimal HTF/LTF surrogate: pass HTF POI via kwargs or via crt_core API once available.
    # Expect: rec['poi_htf']==True and rec['prob_boost_poi']==True
    rows = [
        [100,110,90,100],   # C1
        [95,105,89,99],     # C2 (bullish sweep, close inside)
        [100,104,96,101],   # noise
    ]
    df = df_from_rows(rows, freq="H")
    recs = crt_scan(df, lookback_bars=10, directions=("bullish",), htf_poi_hint=True)
    r = next((r for r in recs if r.get("direction")=="bullish"), None)
    assert r and r.get("poi_htf") is True and r.get("prob_boost_poi") is True

# ----------------------------
# 3) Premium / Discount context
# ----------------------------
@pytest.mark.xfail(reason="Premium/Discount swing context not implemented")
@pytest.mark.parametrize("direction,in_premium", [("bearish", True), ("bullish", False)])
def test_premium_discount_required_zone(direction, in_premium):
    # Build a swing and ensure the C2 occurs in premium (for shorts) or discount (for longs)
    # Expect: flags and a scoring bump
    base = 100.0
    c1 = [base, base+10, base-10, base]  # C1
    if direction == "bearish":
        c2 = [base+5, base+11, base-1, base+6]  # sweep above C1H
    else:
        c2 = [base-5, base+1, base-11, base-6]  # sweep below C1L
    df = df_from_rows([[100,101,99,100], c1, c2, [100,105,95,101]], freq="H")
    recs = crt_scan(df, lookback_bars=10, directions=(direction,), premium_discount_required=True)
    r = next((r for r in recs if r.get("direction")==direction), None)
    assert r and r.get("in_premium")==in_premium if direction=="bearish" else r.get("in_discount")== (not in_premium)
    assert r.get("prob_boost_premium_discount") is True

# ----------------------------
# 4) Time-based filters (Killing Zones)
# ----------------------------
@pytest.mark.xfail(reason="Time-based filter (1-5-9 / 2-6-10) not implemented")
def test_time_killing_zone_flag_and_boost():
    start = "2024-03-01 00:00:00"
    rows = [
        [100,110,90,100],   # 00:00 C1
        [95,105,89,99],     # 01:00 C2 -> in 1-5-9 zone (Forex)
        [100,104,96,101],
    ]
    df = df_from_rows(rows, start=start, freq="H")
    recs = crt_scan(df, lookback_bars=10, directions=("bullish",), killing_zone_profile="1-5-9")
    r = next((r for r in recs if r.get("direction")=="bullish"), None)
    assert r and r.get("in_killing_zone") is True and r.get("prob_boost_time") is True
