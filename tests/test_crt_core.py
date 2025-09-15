# -*- coding: utf-8 -*-
import math
import pandas as pd, numpy as np, pytest
from crt_core import crt_scan, get_key_level_and_confluence, midline


def _weekly_df():
    idx = pd.date_range("2024-01-05", periods=8, freq="W-FRI")
    data = {
        "Open": [10, 10.5, 10, 10.2, 9.8, 9.4, 9.6, 9.7],
        "High": [11, 11.2, 10.8, 10.6, 10, 9.9, 10.2, 10.1],
        "Low": [9, 10, 9.6, 9.7, 9.2, 9.0, 9.3, 9.4],
        "Close": [10, 10.1, 10.2, 10, 9.9, 9.6, 9.8, 9.9],
    }
    return pd.DataFrame(data, index=idx)


def _make_weekly(rows, start="2024-01-05"):
    idx = pd.date_range(start, periods=len(rows), freq="W-FRI")
    df = pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)
    return df


def test_midline():
    assert midline(10, 12) == 11.0


def test_crt_scan_small_lookback_returns_list():
    d = _weekly_df()
    setups = crt_scan(d, lookback_bars=3, confirm_within=2, confirm_method="high")
    assert isinstance(setups, list)


def test_crt_scan_detects_bullish_setup_with_confirmation():
    rows = [
        [10, 11, 9.5, 10.5],  # noise
        [10, 12, 10, 11],  # C1
        [9.2, 11.5, 8.8, 11.0],  # C2 sweep low, close in range
        [11.2, 12.6, 10.8, 12.4],  # C3 high > C1H (confirmation)
        [11.0, 11.8, 10.9, 11.2],  # trailing bar without new sweep
    ]
    df = _make_weekly(rows)

    setups = crt_scan(
        df,
        lookback_bars=10,
        confirm_within=2,
        confirm_method="high",
        directions=("bullish",),
    )

    assert len(setups) == 1
    rec = setups[0]
    assert rec["direction"] == "BULL"
    assert rec["swept_side"] == "LOW"
    assert rec["confirmed"] is True
    assert rec["c3_happened"] is True
    assert rec["confirm_rule"] == "high>C1H in 2"
    assert rec["C3_date_within"] == df.index[3]
    assert rec["C3_date_any"] == df.index[3]
    assert pytest.approx(rec["C1_mid"], rel=0.01) == 11.0
    assert pytest.approx(rec["C2_position_in_range"], rel=0.01) == 0.5


def test_crt_scan_bearish_c3_outside_window_marks_any_only():
    rows = [
        [110, 112, 105, 111],
        [112, 125, 110, 118],  # C1
        [118, 130, 115, 120],  # C2 sweeps high, closes inside
        [117, 123, 108, 111],  # inside window but no confirm (close >= C1L)
        [110, 121, 103, 108],  # confirmation happens after window
        [108, 115, 95, 96],  # drop below C1L -> C3_any only
    ]
    df = _make_weekly(rows)

    setups = crt_scan(
        df,
        lookback_bars=10,
        confirm_within=1,
        confirm_method="close",
        directions=("bearish",),
    )

    assert len(setups) == 1
    rec = setups[0]
    assert rec["direction"] == "BEAR"
    assert rec["swept_side"] == "HIGH"
    assert rec["confirmed"] is False
    assert rec["c3_happened"] is True
    assert rec["confirm_rule"] == "close<C1L in 1"
    assert pd.isna(rec["C3_date_within"])
    assert rec["C3_date_any"] == df.index[4]


def test_crt_scan_respects_midline_and_strict_flags():
    rows = [
        [9.0, 9.8, 8.7, 9.5],  # filler
        [9.5, 10.2, 9.1, 9.9],  # filler
        [10, 12, 10, 11],  # C1
        [9, 11, 8.5, 10.4],  # C2 close below midline (11)
        [9.5, 12.5, 9.3, 12.3],
    ]
    df = _make_weekly(rows)

    setups_default = crt_scan(df, lookback_bars=5, directions=("bullish",))
    assert len(setups_default) == 1

    setups_midline = crt_scan(
        df,
        lookback_bars=5,
        directions=("bullish",),
        require_midline=True,
    )
    assert setups_midline == []

    # Same data but tweak C2 close to fail strict_vs_c1open when enabled
    rows_strict = [
        [9.0, 9.8, 8.7, 9.5],
        [9.5, 10.2, 9.1, 9.9],
        [12, 12.5, 10, 12],  # C1 with open above C2 close
        [9.5, 11.5, 9, 11.0],  # C2 close < C1 open
        [11.5, 13.0, 10.8, 12.8],
    ]
    df_strict = _make_weekly(rows_strict)
    setups_relaxed = crt_scan(df_strict, lookback_bars=5, directions=("bullish",))
    assert len(setups_relaxed) == 1
    setups_strict = crt_scan(
        df_strict,
        lookback_bars=5,
        directions=("bullish",),
        strict_vs_c1open=True,
    )
    assert setups_strict == []


def test_crt_scan_lookback_limits_old_setups():
    rows = [
        [10, 12, 10, 11],
        [9, 11, 8.5, 11],  # first setup candidate
        [11, 13, 11, 12],
        [10, 11, 9, 10],  # later C1
        [8.5, 10.5, 8, 10.2],  # later C2 within lookback
    ]
    df = _make_weekly(rows)

    setups_all = crt_scan(df, lookback_bars=10, directions=("bullish",))
    assert len(setups_all) == 2

    setups_short = crt_scan(df, lookback_bars=2, directions=("bullish",))
    assert len(setups_short) == 1
    assert setups_short[0]["C2_date"] == df.index[4]


def test_key_level_touch_bull():
    idx = pd.date_range("2023-01-31", periods=12, freq="ME")
    htf = pd.DataFrame(
        {
            "Open": np.linspace(10, 9, len(idx)),
            "High": np.linspace(11, 12, len(idx)),
            "Low": np.linspace(9, 8, len(idx)),
            "Close": np.linspace(10, 9.5, len(idx)),
        },
        index=idx,
    )
    tf, kl, kd, conf = get_key_level_and_confluence(
        htf,
        pd.to_datetime("2023-12-29"),
        "BULL",
        9.2,
        10.0,
        9.0,
        9.9,
        6,
        "C1 lub C2",
        "touch (≤, ≥)",
        "1mo",
    )
    assert tf == "1M"
    assert isinstance(kl, float)
    assert kd in htf.index
    assert conf is True


def test_key_level_strict_directional_filter():
    idx = pd.date_range("2023-01-31", periods=4, freq="ME")
    htf = pd.DataFrame(
        {
            "Open": [100, 101, 102, 103],
            "High": [105, 106, 107, 108],
            "Low": [98, 99.5, 100.5, 101],
            "Close": [101, 101.5, 102.5, 103],
        },
        index=idx,
    )
    tf, key_val, _, conf = get_key_level_and_confluence(
        htf,
        idx[-1],
        "BULL",
        101.6,
        104.2,
        100.0,
        104.0,
        6,
        "Tylko C1",
        "strict (<, >)",
        "1mo",
    )
    assert tf == "1M"
    assert math.isclose(key_val, 101.5, rel_tol=1e-6)
    assert conf is False  # fails directional rule (C1 low > key)


def test_key_level_prefers_closest_candidate_between_c1_and_c2():
    idx = pd.date_range("2023-01-31", periods=5, freq="ME")
    htf = pd.DataFrame(
        {
            "Open": [300, 280, 260, 240, 220],
            "High": [305, 285, 265, 245, 225],
            "Low": [295, 275, 255, 100, 90],
            "Close": [298, 278, 258, 102, 95],
        },
        index=idx,
    )
    tf, key_val, _, conf = get_key_level_and_confluence(
        htf,
        idx[-1],
        "BULL",
        310.0,
        330.0,
        99.0,
        118.0,
        9,
        "C1 lub C2",
        "touch (≤, ≥)",
        "1mo",
    )
    assert tf == "1M"
    # The closest HTF level to the C2 low (99) is ~100 from recent bars
    assert math.isclose(key_val, 100.0, rel_tol=1e-6)
    assert conf is True


def test_key_level_handles_missing_history():
    tf, key_val, key_date, conf = get_key_level_and_confluence(
        pd.DataFrame(),
        pd.to_datetime("2024-01-01"),
        "BULL",
        0,
        0,
        0,
        0,
        6,
        "C1 lub C2",
        "touch (≤, ≥)",
        "1mo",
    )
    assert tf == "1M"
    assert math.isnan(key_val)
    assert pd.isna(key_date)
    assert conf is False
