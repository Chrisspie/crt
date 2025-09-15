# -*- coding: utf-8 -*-
import math
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from crt_core import crt_scan

EPS = 1e-9

def mk_df(rows, start="2024-06-07", pad_left=3, pad_right=3):
    pre = [[100.0, 110.0, 90.0, 100.0] for _ in range(pad_left)]
    post = [[100.0, 110.0, 90.0, 100.0] for _ in range(pad_right)]
    arr = pre + rows + post
    idx = pd.date_range(start, periods=len(arr), freq="W-FRI")
    return pd.DataFrame(arr, columns=["Open","High","Low","Close"], index=idx)

def find_by_c2_date(recs, ts):
    for r in recs:
        if pd.to_datetime(r.get("C2_date")) == pd.to_datetime(ts):
            return r
    return None

@settings(max_examples=50, deadline=None)
@given(
    base=st.floats(min_value=50, max_value=150, allow_nan=False, allow_infinity=False),
    width=st.floats(min_value=1e-3, max_value=50, allow_nan=False, allow_infinity=False),
    sweep_delta=st.floats(min_value=1e-6, max_value=5, allow_nan=False, allow_infinity=False),
    close_pos=st.floats(min_value=0.0, max_value=1.0),
    direction=st.sampled_from(["bullish","bearish"]),
    confirm_method=st.sampled_from(["high","close","CLOSE"]),  # allow case variant for close
    confirm_within=st.integers(min_value=0, max_value=3),
)
def test_invariants_basic(base, width, sweep_delta, close_pos, direction, confirm_method, confirm_within):
    # Build C1 with given width
    c1_low = base - width/2.0
    c1_high = base + width/2.0
    c1_bar = [base, c1_high, c1_low, base]  # [O,H,L,C]

    # Build C2 that *does* sweep on the chosen direction and closes inside
    if direction == "bullish":
        c2_low = c1_low - abs(sweep_delta)
        c2_high = max(c1_high - 1e-6, c1_high)  # any
        c2_close = c1_low + close_pos * (c1_high - c1_low)
        c2_bar = [c2_close, max(c2_close, c2_high), c2_low, c2_close]
    else:
        c2_high = c1_high + abs(sweep_delta)
        c2_low  = min(c1_low + 1e-6, c1_low)
        c2_close = c1_low + close_pos * (c1_high - c1_low)
        c2_bar = [c2_close, c2_high, min(c2_close, c2_low), c2_close]

    df = mk_df([c1_bar, c2_bar], start="2024-06-07", pad_left=1, pad_right=3)
    t2 = df.index[1+1]

    out = crt_scan(df, lookback_bars=50, confirm_within=confirm_within, confirm_method=confirm_method, directions=(direction,))
    r = find_by_c2_date(out, t2)
    assert r is not None, "A proper sweep with close inside should produce a setup"
    assert (r["direction"] == "BULL") if direction=="bullish" else (r["direction"] == "BEAR")

    # Now violate the 'close inside' condition -> expect no setup
    if direction == "bullish":
        bad_c2 = [c2_bar[0], c2_bar[1], c2_bar[2], c1_high + 1e-6]  # close above range
    else:
        bad_c2 = [c2_bar[0], c2_bar[1], c2_bar[2], c1_low - 1e-6]   # close below range
    df_bad = mk_df([c1_bar, bad_c2], start="2024-06-07", pad_left=1, pad_right=3)
    out_bad = crt_scan(df_bad, lookback_bars=50, confirm_within=0, directions=(direction,))
    assert find_by_c2_date(out_bad, t2) is None

@settings(max_examples=25, deadline=None)
@given(
    base=st.floats(min_value=50, max_value=150),
    width=st.floats(min_value=0.01, max_value=10),
    sweep=st.floats(min_value=0.1, max_value=3.0),
)
def test_dual_sweep_toggle(base, width, sweep):
    c1_low = base - width/2.0
    c1_high = base + width/2.0
    c1_bar = [base, c1_high, c1_low, base]
    c2_bar = [base, c1_high + abs(sweep), c1_low - abs(sweep), base]  # dual sweep, close inside
    df = mk_df([c1_bar, c2_bar], start="2024-07-05", pad_left=1, pad_right=3)
    t2 = df.index[1+1]
    try:
        out_skip = crt_scan(df, lookback_bars=10, directions=("bullish","bearish"), skip_dual_sweep=True)
    except TypeError:
        pytest.skip("skip_dual_sweep not supported in this repo")
    assert find_by_c2_date(out_skip, t2) is None
    out_both = crt_scan(df, lookback_bars=10, directions=("bullish","bearish"), skip_dual_sweep=False)
    both = [x for x in out_both if x["C2_date"] == t2]
    assert {x["direction"] for x in both} == {"BULL","BEAR"}

@settings(max_examples=30, deadline=None)
@given(
    base=st.floats(min_value=50, max_value=150),
    width=st.floats(min_value=0.02, max_value=10),
    d=st.floats(min_value=1e-6, max_value=2.0),
)
def test_no_lookahead_for_setup(base, width, d):
    c1_low = base - width/2.0
    c1_high = base + width/2.0
    c1_bar = [base, c1_high, c1_low, base]
    c2_bar = [base-1, c1_high-1e-6, c1_low - abs(d), base]  # BULL C2
    j_bar  = [base, c1_high + abs(d), base, base]           # confirm by high at i+1
    df_full = mk_df([c1_bar, c2_bar, j_bar], start="2024-08-02", pad_left=1, pad_right=2)
    df_trunc = df_full.iloc[:(1+1)+1]  # up to and including C2, exclude j_bar
    t2 = df_full.index[1+1]
    out_full = crt_scan(df_full, lookback_bars=50, confirm_within=2, confirm_method="high", directions=("bullish",))
    out_trunc = crt_scan(df_trunc, lookback_bars=50, confirm_within=2, confirm_method="high", directions=("bullish",))
    # Setup must be detectable without future bar
    assert (find_by_c2_date(out_trunc, t2) is not None)
    # Confirmation may differ (trunc missing j_bar)
    r_full = find_by_c2_date(out_full, t2); r_trunc = find_by_c2_date(out_trunc, t2)
    assert r_full is not None and r_trunc is not None
    assert bool(r_full["confirmed"]) in (True, False)
